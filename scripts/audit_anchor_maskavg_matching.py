from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint_with_part_anchoraudit import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss
from src.loss_stage3_gw import (
    build_stage2_visual_prototypes,
    build_class_part_blocks_from_dataset,
    safe_normalize,
)

# IMPORTANT:
# Inline implementation copied from the user's metric.py::structure_retrieval.
# The repository does not need metric.py to exist.
from scipy.optimize import linear_sum_assignment as linear_assignment


def structure_retrieval_inline(feat_1, feat_2, ret_sim=False, use_HM=False, ret_idx=False):
    """
    Same logic as metric.py::structure_retrieval.

    Args:
        feat_1: [N, D1]
        feat_2: [N, D2]
        D1 and D2 do not need to be equal.

    Returns:
        retrieval ratio by default.
    """
    feat_1_ = feat_1 / feat_1.norm(dim=-1, keepdim=True)
    feat_2_ = feat_2 / feat_2.norm(dim=-1, keepdim=True)
    sim_1 = feat_1_ @ feat_1_.transpose(1, 0)
    sim_2 = feat_2_ @ feat_2_.transpose(1, 0)

    N = feat_1_.shape[0]

    sim_1 = sim_1[~torch.eye(N, dtype=torch.bool, device=sim_1.device)].view(N, -1)
    sim_2 = sim_2[~torch.eye(N, dtype=torch.bool, device=sim_2.device)].view(N, -1)

    sim_1 = sim_1 - sim_1.mean(-1).unsqueeze(1)
    sim_2 = sim_2 - sim_2.mean(-1).unsqueeze(1)
    sim_1_norm = sim_1 / sim_1.norm(dim=-1, keepdim=True)
    sim_2_norm = sim_2 / sim_2.norm(dim=-1, keepdim=True)

    sim_1_2 = sim_1_norm @ sim_2_norm.transpose(1, 0)

    if not use_HM:
        idx = sim_1_2.argmax(0).detach().cpu().numpy()
        ret_flag = (idx == list(range(len(sim_1_2))))
        retrieval_structure = ret_flag.sum() / len(sim_1_2)

        if ret_idx:
            import numpy as np
            return retrieval_structure, np.where(ret_flag)[0]
        elif ret_sim:
            return sim_1_2
        return retrieval_structure
    else:
        m = linear_assignment(1 - sim_1_2.detach().cpu().numpy())
        retrieval_ratio_HM = (m[1] == list(range(len(sim_1_2)))).sum() / len(sim_1_2)

        if ret_idx:
            return retrieval_ratio_HM, m[1]
        return retrieval_ratio_HM


def scalar_float(x: Any) -> float:
    """Convert scalar tensor / python number / tuple first item to float."""
    if isinstance(x, (tuple, list)):
        x = x[0]
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


@torch.no_grad()
def load_projector(model_config: str, init_weights: str, device: torch.device):
    with open(model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"].get("model_class", "ProjectionLayer")
    Model = getattr(importlib.import_module("src.model"), model_name)
    model = Model.from_config(cfg["model"]).to(device)

    print(f"[load projector] {init_weights}")
    ckpt = torch.load(init_weights, map_location="cpu")
    msg = model.load_state_dict(ckpt, strict=False)
    print("  missing keys   :", getattr(msg, "missing_keys", []))
    print("  unexpected keys:", getattr(msg, "unexpected_keys", []))

    model.eval()
    return model, cfg


def build_joint_dataset(args, cfg):
    min_obj_area_ratio = float(cfg.get("dataset", {}).get("min_obj_area_ratio", 0.0))
    return DinoClipJointDataset(
        args.dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=".tar" in args.dataset,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
    )


@torch.no_grad()
def build_maskavg_visual_prototypes(
    dataloader: DataLoader,
    num_parts: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build GT-mask averaged part prototypes.

    For each valid part instance:
        local_proto = mean(normalize(cropaug_patch_tokens) inside part_gt_mask_patch)
    Then average local_proto by global part id into 116 prototypes.
    """
    proto_sum = None
    proto_count = torch.zeros(num_parts, device=device)

    for batch in tqdm(dataloader, total=len(dataloader), desc="Build GT-maskavg prototypes"):
        # IMPORTANT:
        # Keep this exactly consistent with the previous standalone
        # audit_t_v_structure_retrieval_anchor_maskavg.py:
        #   - do NOT normalize patch tokens before mask averaging;
        #   - average raw patch tokens inside each GT part mask;
        #   - average by global part id;
        #   - normalize only the final global prototypes.
        patch_tokens = batch["patch_tokens"].to(device, dtype=torch.float32)   # [B, N, D]
        part_masks = batch["part_gt_mask_patch"].to(device).bool()             # [B, K, N]
        part_ids = batch["part_category_id"].to(device).long()                 # [B, K]
        part_valid = batch["part_valid_mask"].to(device).bool()                # [B, K]

        if proto_sum is None:
            proto_sum = torch.zeros(num_parts, patch_tokens.shape[-1], device=device)

        B, K, _ = part_masks.shape
        for b in range(B):
            for k in range(K):
                if not bool(part_valid[b, k]):
                    continue

                pid = int(part_ids[b, k].item())
                if pid < 0 or pid >= num_parts:
                    continue

                mask = part_masks[b, k]
                if int(mask.sum().item()) <= 0:
                    continue

                local_proto = patch_tokens[b, mask].mean(dim=0)

                proto_sum[pid] += local_proto
                proto_count[pid] += 1.0

    if proto_sum is None:
        raise RuntimeError("No GT-maskavg prototype accumulated. Check dataset fields.")

    proto = proto_sum / proto_count.clamp_min(1.0)[:, None]
    proto = safe_normalize(proto, dim=-1)

    return {
        "visual_proto": proto.detach(),
        "proto_count": proto_count.detach(),
        "visual_source": "maskavg",
    }


@torch.no_grad()
def cosine_cost(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    src = safe_normalize(src.float(), dim=-1)
    tgt = safe_normalize(tgt.float(), dim=-1)
    return (1.0 - src @ tgt.T).clamp_min(0.0)


@torch.no_grad()
def hungarian_match(cost: torch.Tensor) -> Tuple[torch.Tensor, float]:
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    perm = torch.empty(cost.shape[0], dtype=torch.long, device=cost.device)
    perm[torch.as_tensor(row_ind, dtype=torch.long, device=cost.device)] = torch.as_tensor(
        col_ind, dtype=torch.long, device=cost.device
    )
    mean_dist = cost[torch.arange(cost.shape[0], device=cost.device), perm].mean().item()
    return perm, float(mean_dist)


@torch.no_grad()
def greedy_edge_match(cost: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Greedy one-to-one matching by sorted edges.

    This is a heuristic baseline. It is not guaranteed to beat Hungarian.
    """
    K = cost.shape[0]
    flat_order = torch.argsort(cost.flatten(), descending=False)

    used_row = torch.zeros(K, dtype=torch.bool, device=cost.device)
    used_col = torch.zeros(K, dtype=torch.bool, device=cost.device)
    perm = torch.full((K,), -1, dtype=torch.long, device=cost.device)

    chosen = 0
    for flat_idx in flat_order.tolist():
        i = flat_idx // K
        j = flat_idx % K
        if used_row[i] or used_col[j]:
            continue

        perm[i] = j
        used_row[i] = True
        used_col[j] = True
        chosen += 1

        if chosen == K:
            break

    if (perm < 0).any():
        remaining_cols = torch.where(~used_col)[0].tolist()
        for i in torch.where(perm < 0)[0].tolist():
            perm[i] = int(remaining_cols.pop(0))

    mean_dist = cost[torch.arange(K, device=cost.device), perm].mean().item()
    return perm, float(mean_dist)


def get_part_name(block: Dict[str, Any], local_idx: int) -> str:
    # Different helper versions may or may not include names.
    for key in ("part_names", "part_class_names", "part_class_name"):
        names = block.get(key, None)
        if isinstance(names, (list, tuple)) and local_idx < len(names):
            return str(names[local_idx])
    return ""


def part_rows_for_block(
    block: Dict[str, Any],
    part_ids: torch.Tensor,
    cost: torch.Tensor,
    hung_perm: torch.Tensor,
    greedy_perm: torch.Tensor,
) -> List[Dict[str, Any]]:
    rows = []
    part_ids_cpu = part_ids.detach().cpu().long()
    hung_cpu = hung_perm.detach().cpu().long()
    greedy_cpu = greedy_perm.detach().cpu().long()
    K = int(part_ids.numel())

    for i in range(K):
        h = int(hung_cpu[i].item())
        g = int(greedy_cpu[i].item())

        identity_dist = float(cost[i, i].detach().cpu().item())
        hung_dist = float(cost[i, h].detach().cpu().item())
        greedy_dist = float(cost[i, g].detach().cpu().item())

        rows.append(
            {
                "class_name": str(block.get("class_name", "")),
                "category_id": int(block["category_id"]),
                "anchor_local_idx": int(i),
                "anchor_part_id": int(part_ids_cpu[i].item()),
                "anchor_part_name": get_part_name(block, i),

                "identity_maskavg_local_idx": int(i),
                "identity_maskavg_part_id": int(part_ids_cpu[i].item()),
                "identity_maskavg_part_name": get_part_name(block, i),
                "identity_dist": identity_dist,
                "identity_sim": 1.0 - identity_dist,

                "hungarian_maskavg_local_idx": h,
                "hungarian_maskavg_part_id": int(part_ids_cpu[h].item()),
                "hungarian_maskavg_part_name": get_part_name(block, h),
                "hungarian_dist": hung_dist,
                "hungarian_sim": 1.0 - hung_dist,

                "greedy_maskavg_local_idx": g,
                "greedy_maskavg_part_id": int(part_ids_cpu[g].item()),
                "greedy_maskavg_part_name": get_part_name(block, g),
                "greedy_dist": greedy_dist,
                "greedy_sim": 1.0 - greedy_dist,
            }
        )

    return rows


@torch.no_grad()
def compute_structure_metrics(
    model,
    block: Dict[str, Any],
    V_anchor_block: torch.Tensor,
    V_mask_block: torch.Tensor,
    hung_perm: torch.Tensor,
    structure_retrieval_fn,
) -> Dict[str, float]:
    """
    T_pre  = original part text features, [K, text_dim]
    T_post = projector(T_pre), [K, visual_dim]
    A      = anchor prototypes, [K, visual_dim]
    M      = maskavg prototypes, [K, visual_dim]

    structure_retrieval_fn is loaded from metric.py.
    """
    T_pre = block["part_text"].float().to(V_anchor_block.device)
    T_post = model.project_clip_txt(T_pre).float()

    A = V_anchor_block.float()
    M = V_mask_block.float()
    M_hung = M[hung_perm]

    return {
        "str_pre_anchor": scalar_float(structure_retrieval_fn(T_pre, A)),
        "str_post_anchor": scalar_float(structure_retrieval_fn(T_post, A)),

        "str_pre_mask": scalar_float(structure_retrieval_fn(T_pre, M)),
        "str_post_mask": scalar_float(structure_retrieval_fn(T_post, M)),

        "str_anchor_mask": scalar_float(structure_retrieval_fn(A, M)),
        "str_anchor_mask_hung": scalar_float(structure_retrieval_fn(A, M_hung)),

        "str_pre_post": scalar_float(structure_retrieval_fn(T_pre, T_post)),
    }


def mean_of(rows: List[Dict[str, Any]], key: str):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def sum_of(rows: List[Dict[str, Any]], key: str):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) if vals else 0.0



def safe_div(num: float, den: float):
    return float(num) / float(den) if float(den) > 0 else None


def get_batch_category_id(batch: Dict[str, Any], b: int) -> int:
    """Robustly read category_id from a collated batch."""
    if "category_id" in batch:
        cat = batch["category_id"]
        if torch.is_tensor(cat):
            return int(cat[b].item())
        if isinstance(cat, (list, tuple)):
            return int(cat[b])

    meta = batch.get("metadata", None)
    if isinstance(meta, (list, tuple)) and b < len(meta):
        if isinstance(meta[b], dict) and "category_id" in meta[b]:
            return int(meta[b]["category_id"])

    raise KeyError("Cannot find category_id in batch. Expected batch['category_id'] or metadata[b]['category_id'].")


def init_counter():
    return {
        "total": 0.0,
        "identity_hits": 0.0,
        "hungarian_hits": 0.0,
        "greedy_hits": 0.0,

        # Pseudo-label segmentation IoU counters.
        # These are accumulated as intersection / union over patch-level
        # pseudo labels produced by the Stage-1 anchor-prototype EM routine.
        "identity_inter": 0.0,
        "identity_union": 0.0,
        "hungarian_inter": 0.0,
        "hungarian_union": 0.0,
        "greedy_inter": 0.0,
        "greedy_union": 0.0,
    }


@torch.no_grad()
def compute_anchor_hit_stats(
    model,
    dataloader: DataLoader,
    block_perm_by_cat: Dict[int, Dict[str, torch.Tensor]],
    patch_temperature: float,
    em_iters: int,
    device: torch.device,
) -> Tuple[Dict[int, Dict[str, float]], Dict[Tuple[int, int], Dict[str, float]], Dict[str, float]]:
    """
    Compute anchor hit rate and pseudo-label IoU before/after permutation.

    This mirrors JointObjPartLoss._anchor_proto_em_pool as closely as possible:
      - project part text;
      - compute part-patch scores inside obj_mask_patch;
      - compute relative scores with JointObjPartLoss._compute_relative_scores;
      - greedily assign one unique anchor patch per valid part.

    Difference from the training helper:
      The repo helper returns aggregate identity anchor_hit_rate only.
      Here we need the selected anchor patch index so that we can test hits
      against the original target mask and against the Hungarian/greedy matched
      target mask. Therefore the small anchor-index extraction loop is local.

    identity hit:
      anchor patch selected for source part i falls inside GT mask of same part i.

    hungarian hit:
      anchor patch selected for source part i falls inside GT mask of
      mask part perm_hungarian[i].

    greedy hit:
      anchor patch selected for source part i falls inside GT mask of
      mask part perm_greedy[i].

    pseudo-label IoU:
      After selecting one anchor per valid part, run the same EM assignment loop
      as JointObjPartLoss._anchor_proto_em_pool to assign each object patch to
      a pseudo part label. Then compare the predicted patch mask of each source
      part with:
        - its same-id GT part mask;
        - its Hungarian-matched GT part mask;
        - its greedy-matched GT part mask.
    """
    model.eval()

    anchor_helper = JointObjPartLoss(
        sim_model=model,
        obj_ltype="infonce",
        lambda_obj=0.0,
        lambda_inst=0.0,
        lambda_overlap=0.0,
        lambda_spear=0.0,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    ).to(device)
    anchor_helper.eval()

    block_stats = defaultdict(init_counter)
    part_stats = defaultdict(init_counter)
    global_stats = init_counter()

    for batch in tqdm(dataloader, total=len(dataloader), desc="Compute anchor hit stats"):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        batch = moved

        part_text_feat = batch["part_text_feat"].float()            # [B, K, Dt]
        patch_tokens = batch["patch_tokens"].float()                # [B, N, Dv]
        obj_mask_patch = batch["obj_mask_patch"].bool()             # [B, N]
        part_valid_mask = batch["part_valid_mask"].bool()           # [B, K]
        part_gt_mask_patch = batch["part_gt_mask_patch"].bool()     # [B, K, N]
        part_category_id = batch["part_category_id"].long()         # [B, K]

        part_proj = model.project_clip_txt(part_text_feat)
        part_proj = anchor_helper._safe_normalize(part_proj, dim=-1)
        patch_tokens_norm = anchor_helper._safe_normalize(patch_tokens, dim=-1)

        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens_norm) / float(patch_temperature)
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        B = int(abs_logits.shape[0])
        for b in range(B):
            cat = get_batch_category_id(batch, b)
            if cat not in block_perm_by_cat:
                continue

            perm_info = block_perm_by_cat[cat]
            block_part_ids = perm_info["part_ids"].to(device).long()
            hung_perm = perm_info["hung_perm"].to(device).long()
            greedy_perm = perm_info["greedy_perm"].to(device).long()

            block_pid_to_local = {
                int(pid.item()): int(i)
                for i, pid in enumerate(block_part_ids)
            }

            sample_part_ids = part_category_id[b]
            sample_pid_to_idx = {
                int(sample_part_ids[k].item()): int(k)
                for k in range(sample_part_ids.numel())
                if bool(part_valid_mask[b, k])
            }

            valid_patch_mask = obj_mask_patch[b]
            valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)

            if valid_part_idx.numel() == 0 or int(valid_patch_mask.sum().item()) == 0:
                continue

            valid_patch_idx_global = torch.nonzero(valid_patch_mask, as_tuple=False).squeeze(1)
            local_scores = abs_logits[b][valid_part_idx][:, valid_patch_mask]
            Kb, Mb = local_scores.shape
            if Kb == 0 or Mb == 0:
                continue

            # Same relative-score and unique-anchor assignment as JointObjPartLoss.
            rel_scores = anchor_helper._compute_relative_scores(local_scores)
            flat_scores = rel_scores.reshape(-1)
            sorted_idx = torch.argsort(flat_scores, descending=True)

            anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=device)
            patch_taken = torch.zeros((Mb,), dtype=torch.bool, device=device)
            assigned_parts = 0

            for flat_id in sorted_idx:
                p_local = torch.div(flat_id, Mb, rounding_mode="floor")
                n_local = flat_id % Mb
                if anchor_idx_local[p_local] != -1:
                    continue
                if patch_taken[n_local]:
                    continue

                anchor_idx_local[p_local] = n_local
                patch_taken[n_local] = True
                assigned_parts += 1

                if assigned_parts == Kb:
                    break

            unassigned = torch.nonzero(anchor_idx_local < 0, as_tuple=False).squeeze(1)
            if unassigned.numel() > 0:
                local_best = rel_scores.argmax(dim=1)
                anchor_idx_local[unassigned] = local_best[unassigned]

            anchor_idx_global = valid_patch_idx_global[anchor_idx_local]

            # Build Stage-1 pseudo labels over object-internal patches by following
            # JointObjPartLoss._anchor_proto_em_pool exactly:
            # initialize C from selected anchor patches, then alternately assign
            # every object patch to the nearest prototype and update prototypes.
            valid_patch_tokens = patch_tokens_norm[b][valid_patch_mask]  # [Mb, D]
            C = valid_patch_tokens[anchor_idx_local]                     # [Kb, D]
            assign = None
            for _ in range(max(int(em_iters), 1)):
                assign_scores = valid_patch_tokens @ C.T                 # [Mb, Kb]
                assign = assign_scores.argmax(dim=1)                     # [Mb]
                assign[anchor_idx_local] = torch.arange(Kb, device=assign.device)

                onehot = torch.nn.functional.one_hot(assign, num_classes=Kb).float()
                count = onehot.sum(dim=0).clamp_min(1.0)
                proto_sum = onehot.T @ valid_patch_tokens
                C = proto_sum / count[:, None]
                C = anchor_helper._safe_normalize(C, dim=-1)

            if assign is None:
                continue

            for p_local in range(Kb):
                src_sample_idx = int(valid_part_idx[p_local].item())
                src_pid = int(sample_part_ids[src_sample_idx].item())

                if src_pid not in block_pid_to_local:
                    continue

                src_block_local = block_pid_to_local[src_pid]
                hung_pid = int(block_part_ids[hung_perm[src_block_local]].item())
                greedy_pid = int(block_part_ids[greedy_perm[src_block_local]].item())

                # Original identity target.
                identity_target_idx = src_sample_idx

                # Permuted targets. If missing, count it as a valid anchor but a miss.
                hung_target_idx = sample_pid_to_idx.get(hung_pid, None)
                greedy_target_idx = sample_pid_to_idx.get(greedy_pid, None)

                patch_idx = int(anchor_idx_global[p_local].item())

                identity_hit = bool(part_gt_mask_patch[b, identity_target_idx, patch_idx].item())
                hung_hit = (
                    bool(part_gt_mask_patch[b, hung_target_idx, patch_idx].item())
                    if hung_target_idx is not None else False
                )
                greedy_hit = (
                    bool(part_gt_mask_patch[b, greedy_target_idx, patch_idx].item())
                    if greedy_target_idx is not None else False
                )

                # Pseudo-label patch mask for this source part.
                pred_mask_local = (assign == p_local)  # [Mb], over valid object patches

                def _target_gt_mask_local(target_idx):
                    if target_idx is None:
                        return torch.zeros_like(pred_mask_local, dtype=torch.bool)
                    return part_gt_mask_patch[b, target_idx, valid_patch_mask].bool()

                def _inter_union(target_idx):
                    gt_mask_local = _target_gt_mask_local(target_idx)
                    inter = (pred_mask_local & gt_mask_local).sum().item()
                    union = (pred_mask_local | gt_mask_local).sum().item()
                    return float(inter), float(union)

                id_inter, id_union = _inter_union(identity_target_idx)
                h_inter, h_union = _inter_union(hung_target_idx)
                g_inter, g_union = _inter_union(greedy_target_idx)

                for stats in (
                    global_stats,
                    block_stats[cat],
                    part_stats[(cat, src_pid)],
                ):
                    stats["total"] += 1.0
                    stats["identity_hits"] += float(identity_hit)
                    stats["hungarian_hits"] += float(hung_hit)
                    stats["greedy_hits"] += float(greedy_hit)

                    stats["identity_inter"] += id_inter
                    stats["identity_union"] += id_union
                    stats["hungarian_inter"] += h_inter
                    stats["hungarian_union"] += h_union
                    stats["greedy_inter"] += g_inter
                    stats["greedy_union"] += g_union

    return dict(block_stats), dict(part_stats), global_stats


def finalize_hit_stats(counter: Dict[str, float]) -> Dict[str, Any]:
    total = float(counter.get("total", 0.0))
    id_hits = float(counter.get("identity_hits", 0.0))
    h_hits = float(counter.get("hungarian_hits", 0.0))
    g_hits = float(counter.get("greedy_hits", 0.0))

    id_inter = float(counter.get("identity_inter", 0.0))
    id_union = float(counter.get("identity_union", 0.0))
    h_inter = float(counter.get("hungarian_inter", 0.0))
    h_union = float(counter.get("hungarian_union", 0.0))
    g_inter = float(counter.get("greedy_inter", 0.0))
    g_union = float(counter.get("greedy_union", 0.0))

    id_iou = safe_div(id_inter, id_union)
    h_iou = safe_div(h_inter, h_union)
    g_iou = safe_div(g_inter, g_union)

    return {
        "anchor_total": total,
        "anchor_identity_hits": id_hits,
        "anchor_hungarian_hits": h_hits,
        "anchor_greedy_hits": g_hits,
        "anchor_hit_identity": safe_div(id_hits, total),
        "anchor_hit_hungarian": safe_div(h_hits, total),
        "anchor_hit_greedy": safe_div(g_hits, total),
        "anchor_hit_delta_hungarian": (
            safe_div(h_hits, total) - safe_div(id_hits, total)
            if total > 0 else None
        ),
        "anchor_hit_delta_greedy": (
            safe_div(g_hits, total) - safe_div(id_hits, total)
            if total > 0 else None
        ),

        # Pseudo-label IoU. For a per-part row this is that part's IoU.
        # For a block/global counter this is a micro IoU over accumulated patches.
        "pseudo_identity_inter": id_inter,
        "pseudo_identity_union": id_union,
        "pseudo_hungarian_inter": h_inter,
        "pseudo_hungarian_union": h_union,
        "pseudo_greedy_inter": g_inter,
        "pseudo_greedy_union": g_union,
        "pseudo_iou_identity": id_iou,
        "pseudo_iou_hungarian": h_iou,
        "pseudo_iou_greedy": g_iou,
        "pseudo_iou_delta_hungarian": (
            h_iou - id_iou if h_iou is not None and id_iou is not None else None
        ),
        "pseudo_iou_delta_greedy": (
            g_iou - id_iou if g_iou is not None and id_iou is not None else None
        ),
    }


def fmt_rate(x):
    return "nan" if x is None else f"{float(x):.4f}"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset", required=True, help="pth/tar dataset used to build both anchor and maskavg prototypes")
    parser.add_argument("--init_weights", required=True, help="trained projector checkpoint")

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_parts", type=int, default=None)
    parser.add_argument("--min_proto_count", type=int, default=None)

    parser.add_argument("--patch_temperature", type=float, default=None)
    parser.add_argument("--em_iters", type=int, default=None)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_json", default="", help="optional path to save detailed results")
    parser.add_argument("--print_parts", action="store_true", help="print per-part matching rows")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("[device]", device)

    model, cfg = load_projector(args.model_config, args.init_weights, device)
    structure_retrieval_fn = structure_retrieval_inline
    print("[metric] using inline metric.py::structure_retrieval logic")

    train_cfg = cfg.get("train", {})
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 128))
    num_parts = int(args.num_parts or train_cfg.get("num_parts", 116))
    min_proto_count = int(args.min_proto_count or train_cfg.get("min_proto_count", 1))
    patch_temperature = float(args.patch_temperature if args.patch_temperature is not None else train_cfg.get("patch_temperature", 0.07))
    em_iters = int(args.em_iters if args.em_iters is not None else train_cfg.get("em_iters", 1))

    dataset = build_joint_dataset(args, cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )

    print("[1/3] Build V_anchor by repo Stage2 anchor routine")
    anchor_pack = build_stage2_visual_prototypes(
        model=model,
        dataloader=dataloader,
        num_parts=num_parts,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        visual_source="anchor",
    )
    V_anchor = anchor_pack["visual_proto"].detach().to(device)
    cnt_anchor = anchor_pack["proto_count"].detach().to(device)

    dataloader2 = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )

    print("[2/3] Build V_maskavg from GT part masks")
    mask_pack = build_maskavg_visual_prototypes(
        dataloader=dataloader2,
        num_parts=num_parts,
        device=device,
    )
    V_mask = mask_pack["visual_proto"].detach().to(device)
    cnt_mask = mask_pack["proto_count"].detach().to(device)

    print("[3/3] Build object-class part blocks by repo helper")
    class_blocks = build_class_part_blocks_from_dataset(dataset, device=device)

    block_results: List[Dict[str, Any]] = []
    part_results: List[Dict[str, Any]] = []
    block_perm_by_cat: Dict[int, Dict[str, torch.Tensor]] = {}

    for block in class_blocks:
        cat = int(block["category_id"])
        class_name = str(block.get("class_name", ""))
        part_ids = block["part_ids"].to(device).long()
        K = int(part_ids.numel())

        # Match the previous structure-retrieval audit script:
        # structure retrieval is only defined/reported for K >= 3.
        # This excludes bottle and pottedplant, giving valid classes = 13 for VOC116.
        if K < 3:
            continue
        if (cnt_anchor[part_ids] < min_proto_count).any():
            continue
        if (cnt_mask[part_ids] < min_proto_count).any():
            continue

        A = V_anchor[part_ids].float()
        M = V_mask[part_ids].float()
        cost = cosine_cost(A, M)

        identity_perm = torch.arange(K, device=device)
        identity_dist = float(cost[identity_perm, identity_perm].mean().item())

        hung_perm, hung_dist = hungarian_match(cost)
        greedy_perm, greedy_dist = greedy_edge_match(cost)

        block_perm_by_cat[cat] = {
            "part_ids": part_ids.detach().clone(),
            "hung_perm": hung_perm.detach().clone(),
            "greedy_perm": greedy_perm.detach().clone(),
        }

        struct = compute_structure_metrics(model, block, A, M, hung_perm, structure_retrieval_fn)
        rows = part_rows_for_block(block, part_ids, cost, hung_perm, greedy_perm)
        part_results.extend(rows)

        block_results.append(
            {
                "class_name": class_name,
                "category_id": cat,
                "num_parts": K,
                "part_ids": [int(x) for x in part_ids.detach().cpu().tolist()],
                "anchor_min_count": int(cnt_anchor[part_ids].min().item()),
                "mask_min_count": int(cnt_mask[part_ids].min().item()),

                "identity_mean_dist": identity_dist,
                "hungarian_mean_dist": float(hung_dist),
                "greedy_mean_dist": float(greedy_dist),
                "identity_mean_sim": 1.0 - identity_dist,
                "hungarian_mean_sim": 1.0 - float(hung_dist),
                "greedy_mean_sim": 1.0 - float(greedy_dist),

                **struct,
                "parts": rows,
            }
        )

    dataloader3 = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )

    print("[4/4] Compute anchor hit rate before/after permutation")
    block_hit_stats, part_hit_stats, global_hit_stats = compute_anchor_hit_stats(
        model=model,
        dataloader=dataloader3,
        block_perm_by_cat=block_perm_by_cat,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        device=device,
    )

    # Attach hit-rate stats to block-level and per-part results.
    for block_result in block_results:
        cat = int(block_result["category_id"])
        block_result.update(finalize_hit_stats(block_hit_stats.get(cat, init_counter())))

        for row in block_result["parts"]:
            key = (int(row["category_id"]), int(row["anchor_part_id"]))
            hit_info = finalize_hit_stats(part_hit_stats.get(key, init_counter()))
            row.update(hit_info)

            # Explicit aliases for "before/after permutation" distance.
            row["before_perm_maskavg_part_id"] = row["identity_maskavg_part_id"]
            row["before_perm_maskavg_part_name"] = row["identity_maskavg_part_name"]
            row["before_perm_dist"] = row["identity_dist"]
            row["before_perm_sim"] = row["identity_sim"]

            row["after_hungarian_maskavg_part_id"] = row["hungarian_maskavg_part_id"]
            row["after_hungarian_maskavg_part_name"] = row["hungarian_maskavg_part_name"]
            row["after_hungarian_dist"] = row["hungarian_dist"]
            row["after_hungarian_sim"] = row["hungarian_sim"]

            row["after_greedy_maskavg_part_id"] = row["greedy_maskavg_part_id"]
            row["after_greedy_maskavg_part_name"] = row["greedy_maskavg_part_name"]
            row["after_greedy_dist"] = row["greedy_dist"]
            row["after_greedy_sim"] = row["greedy_sim"]

        # Block-level part mIoU: average per-part IoU inside this object block.
        block_result["pseudo_miou_identity"] = mean_of(block_result["parts"], "pseudo_iou_identity")
        block_result["pseudo_miou_hungarian"] = mean_of(block_result["parts"], "pseudo_iou_hungarian")
        block_result["pseudo_miou_greedy"] = mean_of(block_result["parts"], "pseudo_iou_greedy")
        block_result["pseudo_miou_delta_hungarian"] = (
            block_result["pseudo_miou_hungarian"] - block_result["pseudo_miou_identity"]
            if block_result["pseudo_miou_hungarian"] is not None and block_result["pseudo_miou_identity"] is not None
            else None
        )
        block_result["pseudo_miou_delta_greedy"] = (
            block_result["pseudo_miou_greedy"] - block_result["pseudo_miou_identity"]
            if block_result["pseudo_miou_greedy"] is not None and block_result["pseudo_miou_identity"] is not None
            else None
        )

    # Rebuild flat part_results after rows have been updated.
    part_results = []
    for block_result in block_results:
        part_results.extend(block_result["parts"])

    print("=" * 210)
    print("Anchor prototype vs GT-maskavg prototype matching + structure retrieval")
    print(f"dataset       : {args.dataset}")
    print(f"ckpt          : {args.init_weights}")
    print(f"num_parts     : {num_parts}")
    print(f"min_proto_cnt : {min_proto_count}")
    print("lower distance is better; str_* uses inline metric.py::structure_retrieval; higher is better")
    print("=" * 210)
    print(
        f"{'class':<22} {'cat':<5} {'K':<4} "
        f"{'anc_min':<8} {'mask_min':<8} "
        f"{'identity':<10} {'hung':<10} {'greedy':<10} "
        f"{'str_pA':<8} {'str_zA':<8} {'str_pM':<8} {'str_zM':<8} "
        f"{'str_AM':<8} {'str_AMh':<8} {'str_pz':<8}"
    )

    for block_result in block_results:
        print(
            f"{block_result['class_name']:<22} {int(block_result['category_id']):<5d} {int(block_result['num_parts']):<4d} "
            f"{int(block_result['anchor_min_count']):<8d} {int(block_result['mask_min_count']):<8d} "
            f"{block_result['identity_mean_dist']:<10.6f} {block_result['hungarian_mean_dist']:<10.6f} {block_result['greedy_mean_dist']:<10.6f} "
            f"{block_result['str_pre_anchor']:<8.4f} {block_result['str_post_anchor']:<8.4f} "
            f"{block_result['str_pre_mask']:<8.4f} {block_result['str_post_mask']:<8.4f} "
            f"{block_result['str_anchor_mask']:<8.4f} {block_result['str_anchor_mask_hung']:<8.4f} "
            f"{block_result['str_pre_post']:<8.4f}"
        )

    print("-" * 210)

    global_hit = finalize_hit_stats(global_hit_stats)
    if block_results:
        print(
            f"{'MEAN':<49} "
            f"{mean_of(block_results, 'identity_mean_dist'):<10.6f} "
            f"{mean_of(block_results, 'hungarian_mean_dist'):<10.6f} "
            f"{mean_of(block_results, 'greedy_mean_dist'):<10.6f} "
            f"{mean_of(block_results, 'str_pre_anchor'):<8.4f} "
            f"{mean_of(block_results, 'str_post_anchor'):<8.4f} "
            f"{mean_of(block_results, 'str_pre_mask'):<8.4f} "
            f"{mean_of(block_results, 'str_post_mask'):<8.4f} "
            f"{mean_of(block_results, 'str_anchor_mask'):<8.4f} "
            f"{mean_of(block_results, 'str_anchor_mask_hung'):<8.4f} "
            f"{mean_of(block_results, 'str_pre_post'):<8.4f}"
        )
        print(f"valid blocks: {len(block_results)}")
        print(f"part rows   : {len(part_results)}")
    else:
        print("No valid object blocks. Check min_proto_count and prototype counts.")

    if args.print_parts and part_results:
        print()
        print("=" * 230)
        print("Part-level anchor-mask distance and anchor-hit table")
        print("identity/hung/greedy are distances: 1 - cos(anchor part prototype, target maskavg prototype).")
        print("hit_* are anchor-patch hit rates against the target part mask. Higher is better.")
        print("iou_* are pseudo-label patch IoUs against the target part GT mask. Higher is better.")
        print("=" * 230)
        print(
            f"{'class':<16} {'cat':<5} {'src_pid':<7} {'anchor_part':<30} "
            f"{'identity':<10} {'hung':<10} {'greedy':<10} "
            f"{'hit_id(name)':<42} {'hit_h(name)':<42} {'hit_g(name)':<42} "
            f"{'iou_id':<8} {'iou_h':<8} {'iou_g':<8} {'n':<6}"
        )
        for r in part_results:
            id_name = f"{fmt_rate(r.get('anchor_hit_identity'))} -> {r['identity_maskavg_part_name']}"
            h_name = f"{fmt_rate(r.get('anchor_hit_hungarian'))} -> {r['hungarian_maskavg_part_name']}"
            g_name = f"{fmt_rate(r.get('anchor_hit_greedy'))} -> {r['greedy_maskavg_part_name']}"
            print(
                f"{r['class_name']:<16} {int(r['category_id']):<5d} "
                f"{int(r['anchor_part_id']):<7d} {str(r['anchor_part_name'])[:29]:<30} "
                f"{r['identity_dist']:<10.6f} {r['hungarian_dist']:<10.6f} {r['greedy_dist']:<10.6f} "
                f"{id_name[:41]:<42} {h_name[:41]:<42} {g_name[:41]:<42} "
                f"{fmt_rate(r.get('pseudo_iou_identity')):<8} "
                f"{fmt_rate(r.get('pseudo_iou_hungarian')):<8} "
                f"{fmt_rate(r.get('pseudo_iou_greedy')):<8} "
                f"{int(r.get('anchor_total') or 0):<6d}"
            )
        print("-" * 230)
        print(
            "MICRO anchor hit: "
            f"identity={fmt_rate(global_hit['anchor_hit_identity'])}, "
            f"hungarian={fmt_rate(global_hit['anchor_hit_hungarian'])}, "
            f"greedy={fmt_rate(global_hit['anchor_hit_greedy'])}, "
            f"total={int(global_hit['anchor_total'])}"
        )
        print(
            "MACRO anchor hit over object blocks: "
            f"identity={fmt_rate(mean_of(block_results, 'anchor_hit_identity'))}, "
            f"hungarian={fmt_rate(mean_of(block_results, 'anchor_hit_hungarian'))}, "
            f"greedy={fmt_rate(mean_of(block_results, 'anchor_hit_greedy'))}"
        )
        print(
            "MACRO pseudo-label part mIoU over parts: "
            f"identity={fmt_rate(mean_of(part_results, 'pseudo_iou_identity'))}, "
            f"hungarian={fmt_rate(mean_of(part_results, 'pseudo_iou_hungarian'))}, "
            f"greedy={fmt_rate(mean_of(part_results, 'pseudo_iou_greedy'))}"
        )
        print(
            "MACRO pseudo-label mIoU over object blocks: "
            f"identity={fmt_rate(mean_of(block_results, 'pseudo_miou_identity'))}, "
            f"hungarian={fmt_rate(mean_of(block_results, 'pseudo_miou_hungarian'))}, "
            f"greedy={fmt_rate(mean_of(block_results, 'pseudo_miou_greedy'))}"
        )
        print(
            "MICRO pseudo-label IoU over all part masks: "
            f"identity={fmt_rate(global_hit.get('pseudo_iou_identity'))}, "
            f"hungarian={fmt_rate(global_hit.get('pseudo_iou_hungarian'))}, "
            f"greedy={fmt_rate(global_hit.get('pseudo_iou_greedy'))}"
        )

    if args.save_json:
        global_hit = finalize_hit_stats(global_hit_stats)
        summary = {
            "valid_blocks": len(block_results),
            "num_part_rows": len(part_results),
            "mean_identity_dist": mean_of(block_results, "identity_mean_dist"),
            "mean_hungarian_dist": mean_of(block_results, "hungarian_mean_dist"),
            "mean_greedy_dist": mean_of(block_results, "greedy_mean_dist"),

            "mean_anchor_hit_identity": mean_of(block_results, "anchor_hit_identity"),
            "mean_anchor_hit_hungarian": mean_of(block_results, "anchor_hit_hungarian"),
            "mean_anchor_hit_greedy": mean_of(block_results, "anchor_hit_greedy"),
            "mean_anchor_hit_delta_hungarian": mean_of(block_results, "anchor_hit_delta_hungarian"),
            "mean_anchor_hit_delta_greedy": mean_of(block_results, "anchor_hit_delta_greedy"),

            "micro_anchor_hit_identity": global_hit["anchor_hit_identity"],
            "micro_anchor_hit_hungarian": global_hit["anchor_hit_hungarian"],
            "micro_anchor_hit_greedy": global_hit["anchor_hit_greedy"],
            "micro_anchor_total": global_hit["anchor_total"],
            "micro_anchor_identity_hits": global_hit["anchor_identity_hits"],
            "micro_anchor_hungarian_hits": global_hit["anchor_hungarian_hits"],
            "micro_anchor_greedy_hits": global_hit["anchor_greedy_hits"],

            "mean_pseudo_iou_identity_over_parts": mean_of(part_results, "pseudo_iou_identity"),
            "mean_pseudo_iou_hungarian_over_parts": mean_of(part_results, "pseudo_iou_hungarian"),
            "mean_pseudo_iou_greedy_over_parts": mean_of(part_results, "pseudo_iou_greedy"),
            "mean_pseudo_iou_delta_hungarian_over_parts": mean_of(part_results, "pseudo_iou_delta_hungarian"),
            "mean_pseudo_iou_delta_greedy_over_parts": mean_of(part_results, "pseudo_iou_delta_greedy"),

            "mean_pseudo_miou_identity_over_blocks": mean_of(block_results, "pseudo_miou_identity"),
            "mean_pseudo_miou_hungarian_over_blocks": mean_of(block_results, "pseudo_miou_hungarian"),
            "mean_pseudo_miou_greedy_over_blocks": mean_of(block_results, "pseudo_miou_greedy"),
            "mean_pseudo_miou_delta_hungarian_over_blocks": mean_of(block_results, "pseudo_miou_delta_hungarian"),
            "mean_pseudo_miou_delta_greedy_over_blocks": mean_of(block_results, "pseudo_miou_delta_greedy"),

            "micro_pseudo_iou_identity": global_hit["pseudo_iou_identity"],
            "micro_pseudo_iou_hungarian": global_hit["pseudo_iou_hungarian"],
            "micro_pseudo_iou_greedy": global_hit["pseudo_iou_greedy"],
            "micro_pseudo_identity_inter": global_hit["pseudo_identity_inter"],
            "micro_pseudo_identity_union": global_hit["pseudo_identity_union"],
            "micro_pseudo_hungarian_inter": global_hit["pseudo_hungarian_inter"],
            "micro_pseudo_hungarian_union": global_hit["pseudo_hungarian_union"],
            "micro_pseudo_greedy_inter": global_hit["pseudo_greedy_inter"],
            "micro_pseudo_greedy_union": global_hit["pseudo_greedy_union"],

            "mean_str_pre_anchor": mean_of(block_results, "str_pre_anchor"),
            "mean_str_post_anchor": mean_of(block_results, "str_post_anchor"),
            "mean_str_pre_mask": mean_of(block_results, "str_pre_mask"),
            "mean_str_post_mask": mean_of(block_results, "str_post_mask"),
            "mean_str_anchor_mask": mean_of(block_results, "str_anchor_mask"),
            "mean_str_anchor_mask_hung": mean_of(block_results, "str_anchor_mask_hung"),
            "mean_str_pre_post": mean_of(block_results, "str_pre_post"),
        }

        payload = {
            "dataset": args.dataset,
            "init_weights": args.init_weights,
            "num_parts": num_parts,
            "min_proto_count": min_proto_count,

            # New schema:
            "block_results": block_results,
            "part_results": part_results,
            "summary": summary,

            # Backward compatibility with older script:
            "results": block_results,
        }

        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
