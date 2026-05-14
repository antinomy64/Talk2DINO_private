from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import importlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_joint_with_part_anchoraudit import DinoClipJointDataset, joint_collate_fn
from src.loss_stage3_gw import (
    Stage3GWLoss,
    build_stage2_visual_prototypes,
    build_class_part_blocks_from_dataset,
    safe_normalize,
    pairwise_cosine_distance,
    hard_bijective_gw_match,
    hard_gw_struct_objective,
)


# Your current fixed dataset fields. Change here if your pth keys change.
OBJ_FEATURE_NAME = "avg_self_attn_out"
PART_FEATURE_NAME = "cropaug_patch_tokens"
OBJ_TEXT_NAME = "ann_feats"
PART_TEXT_NAME = "part_ann_feats"
RESIZE_DIM = 448
CROP_DIM = 448
PATCH_SIZE = 14

# Debug fake-text settings.
AFFINE_SCALE = 0.05
AFFINE_BIAS = 0.0
MIN_BLOCK_PARTS = 3       # skip k=2, because 2-point structures are permutation-ambiguous
PRINT_EVERY = 1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DebugProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            m = min(in_dim, out_dim)
            self.proj.weight[:m, :m] = torch.eye(m)

    def project_clip_txt(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


def load_model(config_path: str, ckpt_path: str, device: torch.device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"].get("model_class", "ProjectionLayer")
    Model = getattr(importlib.import_module("src.model"), model_name)
    model = Model.from_config(cfg["model"]).to(device)

    if ckpt_path:
        print(f"[load] {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model, cfg


def build_dataset(dataset_path: str, cfg: dict):
    min_obj_area_ratio = float(cfg.get("dataset", {}).get("min_obj_area_ratio", 0.0))
    return DinoClipJointDataset(
        dataset_path,
        obj_feature_name=OBJ_FEATURE_NAME,
        part_feature_name=PART_FEATURE_NAME,
        obj_text_name=OBJ_TEXT_NAME,
        part_text_name=PART_TEXT_NAME,
        resize_dim=RESIZE_DIM,
        crop_dim=CROP_DIM,
        patch_size=PATCH_SIZE,
        with_background=False,
        is_wds=".tar" in dataset_path,
        path_prefix=None,
        min_obj_area_ratio=min_obj_area_ratio,
    )


@torch.no_grad()
def make_fake_blocks(class_blocks, visual_proto, fake_text_dim: int):
    """V[part_ids] -> fake CLIP text, then shuffle rows inside every block."""
    device = visual_proto.device
    visual_dim = visual_proto.shape[-1]

    # Cross-dim semi-orthogonal projection: [visual_dim, fake_text_dim]
    q, _ = torch.linalg.qr(torch.randn(visual_dim, fake_text_dim, device=device))
    A = q[:, :fake_text_dim]

    fake_global = visual_proto.float() @ A
    scale = 1.0 + AFFINE_SCALE * torch.randn(fake_text_dim, device=device)
    bias = AFFINE_BIAS * torch.randn(fake_text_dim, device=device)
    fake_global = safe_normalize(fake_global * scale[None, :] + bias[None, :], dim=-1)

    fake_blocks = []
    true_match = {}

    for block in class_blocks:
        part_ids = block["part_ids"].to(device).long()
        k = part_ids.numel()
        if k < MIN_BLOCK_PARTS:
            continue

        fake_local = fake_global[part_ids]       # [K, fake_text_dim]
        perm = torch.randperm(k, device=device) # row i comes from local visual column perm[i]

        new_block = dict(block)
        new_block["part_ids"] = part_ids.detach()
        new_block["part_text"] = fake_local[perm].detach()
        new_block["row_part_ids"] = part_ids[perm].detach()
        fake_blocks.append(new_block)
        true_match[int(block["category_id"])] = perm.detach()

    return fake_blocks, true_match


@torch.no_grad()
def evaluate(criterion, projector, true_match, gw_max_iter: int, num_restarts: int):
    total = 0
    hit = 0
    gw_vals, pre_v_vals, post_v_vals, prepost_vals = [], [], [], []

    for block in criterion.gw_blocks:
        cat_id = int(block["category_id"])
        if cat_id not in true_match:
            continue

        T = block["part_text"].float()              # fake CLIP text, [K, 512]
        V = safe_normalize(block["visual"].float(), dim=-1)  # visual, [K, 768]
        target = true_match[cat_id].long().to(V.device)
        k = T.shape[0]

        Z = safe_normalize(projector.project_clip_txt(T), dim=-1)
        C_t = pairwise_cosine_distance(T)
        C_z = pairwise_cosine_distance(Z)
        C_v = pairwise_cosine_distance(V)

        result = hard_bijective_gw_match(
            C_z, C_v,
            num_iters=gw_max_iter,
            num_restarts=num_restarts,
            include_identity=True,
        )
        pred = result[0] if isinstance(result, tuple) else result
        pred = pred.to(V.device).long()

        hit += int((pred == target).sum().item())
        total += int(k)

        C_v_true = C_v[target][:, target]
        gw_vals.append(hard_gw_struct_objective(C_z, C_v, pred).detach())
        pre_v_vals.append(F.mse_loss(C_t, C_v_true).detach())
        post_v_vals.append(F.mse_loss(C_z, C_v_true).detach())
        prepost_vals.append(F.mse_loss(C_z, C_t).detach())

    return {
        "Hacc": hit / max(total, 1),
        "gw_struct": torch.stack(gw_vals).mean().item(),
        "preV": torch.stack(pre_v_vals).mean().item(),
        "postV": torch.stack(post_v_vals).mean().item(),
        "prepost": torch.stack(prepost_vals).mean().item(),
        "num_parts": total,
    }


@torch.no_grad()
def fit_block_pca_basis(V, Z, out_dim=2):
    X = torch.cat([V.detach().float().cpu(), Z.detach().float().cpu()], dim=0)
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    basis = Vh[:out_dim].T.contiguous()
    return mean, basis


@torch.no_grad()
def project_2d(x, mean, basis):
    x = x.detach().float().cpu()
    return (x - mean) @ basis


@torch.no_grad()
def hard_bijective_cosine_match(Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Greedy one-to-one cosine matching.

    Process Z rows in order. For each Z_i, choose the most similar V_j
    that has not been used by previous rows. This is intentionally NOT
    Hungarian/global-optimal matching; it is a row-order greedy cosine
    bijection baseline for visualization.
    """
    sim = (Z @ V.T).detach().float().cpu()  # [K, K]
    k = sim.shape[0]

    pred = torch.empty(k, dtype=torch.long)
    used_v = torch.zeros(k, dtype=torch.bool)

    for i in range(k):
        order = torch.argsort(sim[i], descending=True)
        chosen = None
        for j in order.tolist():
            if not used_v[j]:
                chosen = j
                break
        if chosen is None:
            # Should not happen for square KxK matching, but keep it safe.
            remaining = torch.where(~used_v)[0]
            chosen = int(remaining[0].item())

        pred[i] = chosen
        used_v[chosen] = True

    return pred


def set_square_limits(ax, V2, Z2, pad_ratio=0.08):
    """Use one square coordinate range for V/Z plots in the same block."""
    pts = np.concatenate([V2, Z2], axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min)
    half = max(half * (1.0 + pad_ratio), 1e-6)

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


@torch.no_grad()
def save_all_block_triplet_viz(criterion, projector, true_match, step, save_root, gw_max_iter, num_restarts):
    save_root = Path(save_root)
    step_dir = save_root / f"step_{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    for block_idx, block in enumerate(criterion.gw_blocks):
        cat_id = int(block["category_id"])
        if cat_id not in true_match:
            continue

        V = safe_normalize(block["visual"].float(), dim=-1)
        T = block["part_text"].float()
        Z = safe_normalize(projector.project_clip_txt(T), dim=-1)

        v_part_ids = block["part_ids"].detach().cpu().tolist()
        z_part_ids = block.get("row_part_ids", block["part_ids"]).detach().cpu().tolist()

        C_z = pairwise_cosine_distance(Z)
        C_v = pairwise_cosine_distance(V)
        result = hard_bijective_gw_match(
            C_z, C_v,
            num_iters=gw_max_iter,
            num_restarts=num_restarts,
            include_identity=True,
        )
        pred = result[0] if isinstance(result, tuple) else result
        pred = pred.detach().cpu().long()
        target = true_match[cat_id].detach().cpu().long()

        # Greedy feature-level cosine bijection: row-order greedy one-to-one Z_i -> V_j assignment.
        # This is only for visualization, used to compare against hard-GW.
        cos_bij = hard_bijective_cosine_match(Z, V).detach().cpu().long()

        mean, basis = fit_block_pca_basis(V, Z)
        V2 = project_2d(V, mean, basis).numpy()
        Z2 = project_2d(Z, mean, basis).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax1, ax2, ax3 = axes

        label_box = dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.2)

        ax1.scatter(V2[:, 0], V2[:, 1], s=80, c="blue")
        for i, _pid in enumerate(v_part_ids):
            ax1.text(
                V2[i, 0], V2[i, 1], f"{i}",
                fontsize=11, color="black", fontweight="bold",
                ha="center", va="center", bbox=label_box,
            )
        ax1.set_title("V: true local id")
        set_square_limits(ax1, V2, Z2)

        ax2.scatter(Z2[:, 0], Z2[:, 1], s=80, c="red")
        for i, _pid in enumerate(z_part_ids):
            true_local_idx = int(target[i].item())
            ax2.text(
                Z2[i, 0], Z2[i, 1], f"{true_local_idx}",
                fontsize=11, color="black", fontweight="bold",
                ha="center", va="center", bbox=label_box,
            )
        ax2.set_title("Projected T: mapped-back true local id")
        set_square_limits(ax2, V2, Z2)

        ax3.scatter(V2[:, 0], V2[:, 1], s=80, c="blue", label="V")
        ax3.scatter(Z2[:, 0], Z2[:, 1], s=80, c="red", label="Projected T")
        for i, _pid in enumerate(v_part_ids):
            ax3.text(
                V2[i, 0], V2[i, 1], f"{i}",
                fontsize=11, color="black", fontweight="bold",
                ha="center", va="center", bbox=label_box,
            )
        for i, _pid in enumerate(z_part_ids):
            true_local_idx = int(target[i].item())
            ax3.text(
                Z2[i, 0], Z2[i, 1], f"{true_local_idx}",
                fontsize=11, color="black", fontweight="bold",
                ha="center", va="center", bbox=label_box,
            )

        # Black lines: hard-GW bijective matching.
        # Correct match rule: the black number at the red endpoint equals
        # the black number at the blue endpoint.
        n_correct = 0
        for i in range(len(Z2)):
            j = int(pred[i].item())
            true_j = int(target[i].item())
            n_correct += int(j == true_j)
            ax3.plot(
                [Z2[i, 0], V2[j, 0]],
                [Z2[i, 1], V2[j, 1]],
                linewidth=1.4,
                c="black",
                alpha=0.75,
                label="hard-GW match" if i == 0 else None,
            )

        # Blue dashed lines: feature-level cosine bijection.
        # Correct match rule is the same: the number at the red endpoint
        # should equal the number at the blue endpoint.
        n_cos_correct = 0
        for i in range(len(Z2)):
            j = int(cos_bij[i].item())
            true_j = int(target[i].item())
            n_cos_correct += int(j == true_j)
            ax3.plot(
                [Z2[i, 0], V2[j, 0]],
                [Z2[i, 1], V2[j, 1]],
                linewidth=1.1,
                c="blue",
                alpha=0.60,
                linestyle="--",
                label="greedy cosine" if i == 0 else None,
            )

        acc = n_correct / max(len(Z2), 1)
        cos_acc = n_cos_correct / max(len(Z2), 1)
        ax3.set_title(f"Overlay: black=hard-GW acc={acc:.3f}, blue dashed=greedy-cos acc={cos_acc:.3f}")
        ax3.legend()
        set_square_limits(ax3, V2, Z2)

        class_name = block.get("class_name", str(cat_id))
        fig.suptitle(f"step={step} | block={block_idx} | cat_id={cat_id} | class={class_name}")
        plt.tight_layout()
        out_path = step_dir / f"block_{block_idx:03d}_cat_{cat_id}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--init_weights", required=True)
    parser.add_argument("--visual_source", default="anchor", choices=["anchor", "zpart"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--viz_dir", type=str, default="stage3_debug_viz")
    parser.add_argument("--viz_block_idx", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    base_model, cfg = load_model(args.model_config, args.init_weights, device)
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {})

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 128))
    num_parts = int(train_cfg.get("num_parts", 116))
    min_proto_count = int(train_cfg.get("min_proto_count", 1))
    patch_temperature = float(train_cfg.get("patch_temperature", 0.07))
    em_iters = int(train_cfg.get("em_iters", 1))
    lambda_gw = float(train_cfg.get("lambda_gw", 1.0))
    lambda_struct = float(train_cfg.get("lambda_struct", 1.0))
    gw_max_iter = int(train_cfg.get("gw_max_iter", 20))
    num_restarts = int(train_cfg.get("sinkhorn_iter", 50))
    fake_text_dim = int(model_cfg.get("clip_embed_dim", 512))

    dataset = build_dataset(args.train_dataset, cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=joint_collate_fn)

    print(f"[Stage2] build visual prototypes: {args.visual_source}")
    proto = build_stage2_visual_prototypes(
        model=base_model,
        dataloader=loader,
        num_parts=num_parts,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        visual_source=args.visual_source,
    )
    visual_proto = proto["visual_proto"].detach().to(device)
    proto_count = proto["proto_count"].detach().to(device)
    visual_dim = visual_proto.shape[-1]
    print("[V]", tuple(visual_proto.shape), "valid", int((proto_count >= min_proto_count).sum().item()))

    class_blocks = build_class_part_blocks_from_dataset(dataset, device=device)
    fake_blocks, true_match = make_fake_blocks(class_blocks, visual_proto, fake_text_dim)
    print(f"[blocks] {len(fake_blocks)} blocks, fake text dim={fake_text_dim}, visual dim={visual_dim}")

    projector = DebugProjector(fake_text_dim, visual_dim).to(device)
    criterion = Stage3GWLoss(
        sim_model=projector,
        visual_proto=visual_proto,
        class_blocks=fake_blocks,
        lambda_obj=0.0,
        lambda_gw=lambda_gw,
        lambda_struct=lambda_struct,
        gw_max_iter=gw_max_iter,
        sinkhorn_iter=num_restarts,
        min_proto_count=min_proto_count,
        proto_count=proto_count,
    ).to(device)
    opt = torch.optim.AdamW(projector.parameters(), lr=1e-4)

    def print_row(tag, losses=None):
        m = evaluate(criterion, projector, true_match, gw_max_iter, num_restarts)
        if losses is None:
            print(f"{tag} Hacc={m['Hacc']:.3f} preV={m['preV']:.3e} postV={m['postV']:.3e} prepost={m['prepost']:.3e} parts={m['num_parts']}")
        else:
            print(f"{tag} total={losses['total'].item():.6f} gw={losses['gw'].item():.6f} struct={losses['struct'].item():.6f} "
                  f"Hacc={m['Hacc']:.3f} preV={m['preV']:.3e} postV={m['postV']:.3e} prepost={m['prepost']:.3e}")

    print_row("[before]")
    save_all_block_triplet_viz(criterion, projector, true_match, 0, args.viz_dir, gw_max_iter, num_restarts)

    for step in tqdm(range(args.steps), desc="minimal-stage3-debug"):
        losses = criterion(batch=None, do_anchor_audit=False, do_structure_audit=False)

        if step % PRINT_EVERY == 0 or step == args.steps - 1:
            print_row(f"[step {step} before_update]", losses)

        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()

        if step % PRINT_EVERY == 0 or step == args.steps - 1:
            print_row(f"[step {step} after_update]", None)
            save_all_block_triplet_viz(
                criterion,
                projector,
                true_match,
                step + 1,
                args.viz_dir,
                gw_max_iter,
                num_restarts,
            )


if __name__ == "__main__":
    main()
