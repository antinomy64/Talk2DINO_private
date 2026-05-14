#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())

from src.dataset_joint_with_part_anchoraudit import DinoClipJointDataset, joint_collate_fn
from src.loss_stage3_gw import (
    build_stage2_visual_prototypes,
    build_class_part_blocks_from_dataset,
    safe_normalize,
)


def load_model(model_config, ckpt_path, device):
    with open(model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_class = cfg["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class)

    model = ModelClass.from_config(cfg["model"])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model.to(device).eval()
    return model, cfg


def build_dataset(args, pth_path):
    return DinoClipJointDataset(
        pth_path,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        min_obj_area_ratio=args.min_obj_area_ratio,
    )


def structure_retrieval_metric(feat_1, feat_2):
    """
    Same core logic as metric.py::structure_retrieval:
      1) cosine self-sim matrix for feat_1 / feat_2
      2) remove diagonal, row-center each structural vector
      3) normalize each structural vector
      4) retrieve by argmax structural-vector similarity
      5) ratio of identity retrieval
    """
    assert feat_1.shape[0] == feat_2.shape[0]
    N = feat_1.shape[0]
    if N <= 2:
        return float("nan")

    feat_1 = safe_normalize(feat_1.float(), dim=-1)
    feat_2 = safe_normalize(feat_2.float(), dim=-1)

    sim_1 = feat_1 @ feat_1.t()
    sim_2 = feat_2 @ feat_2.t()

    offdiag = ~torch.eye(N, dtype=torch.bool, device=feat_1.device)
    sim_1 = sim_1[offdiag].view(N, -1)
    sim_2 = sim_2[offdiag].view(N, -1)

    sim_1 = sim_1 - sim_1.mean(dim=-1, keepdim=True)
    sim_2 = sim_2 - sim_2.mean(dim=-1, keepdim=True)

    sim_1 = safe_normalize(sim_1, dim=-1)
    sim_2 = safe_normalize(sim_2, dim=-1)

    sim_12 = sim_1 @ sim_2.t()
    idx = sim_12.argmax(dim=0)
    target = torch.arange(N, device=feat_1.device)
    return float((idx == target).float().mean().detach().cpu().item())


@torch.no_grad()
def build_anchor_visual_proto(args, model, cfg):
    train_cfg = cfg.get("train", {})
    patch_temperature = float(train_cfg.get("patch_temperature", 0.07))
    em_iters = int(train_cfg.get("em_iters", 1))

    dataset = build_dataset(args, args.v_pth)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )

    proto = build_stage2_visual_prototypes(
        model=model,
        dataloader=loader,
        num_parts=args.num_parts,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        visual_source="anchor",
    )

    return (
        proto["visual_proto"].detach().to(args.device),
        proto["proto_count"].detach().to(args.device),
    )


@torch.no_grad()
def build_maskavg_visual_proto(args):
    """
    Build V by averaging patch tokens inside GT part masks, then averaging globally
    by 116 part ids.
    """
    dataset = build_dataset(args, args.v_pth)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )

    proto_sum = None
    proto_count = torch.zeros(args.num_parts, dtype=torch.long, device=args.device)

    for batch in loader:
        patch_tokens = batch["patch_tokens"].to(args.device).float()              # [B, N, D]
        part_ids = batch["part_category_id"].to(args.device).long()              # [B, K]
        part_valid = batch["part_valid_mask"].to(args.device).bool()             # [B, K]
        part_masks = batch["part_gt_mask_patch"].to(args.device).bool()          # [B, K, N]

        B, K = part_ids.shape
        D = patch_tokens.shape[-1]
        if proto_sum is None:
            proto_sum = torch.zeros(args.num_parts, D, dtype=torch.float32, device=args.device)

        for b in range(B):
            for k in range(K):
                if not part_valid[b, k]:
                    continue
                pid = int(part_ids[b, k].item())
                mask = part_masks[b, k]
                if mask.sum().item() == 0:
                    continue

                feat = patch_tokens[b, mask].mean(dim=0)
                proto_sum[pid] += feat
                proto_count[pid] += 1

    visual_proto = proto_sum / proto_count.clamp_min(1).float().unsqueeze(-1)
    visual_proto = safe_normalize(visual_proto, dim=-1)
    return visual_proto, proto_count


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--t_pth", required=True, help="Dataset used to collect preT/postT class blocks.")
    parser.add_argument("--v_pth", required=True, help="Dataset used to build V prototypes.")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--projector_ckpt", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--min_obj_area_ratio", type=float, default=0.0)

    parser.add_argument("--num_parts", type=int, default=116)
    parser.add_argument("--min_proto_count", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    model, cfg = load_model(args.model_config, args.projector_ckpt, args.device)

    V_anchor, cnt_anchor = build_anchor_visual_proto(args, model, cfg)
    V_maskavg, cnt_maskavg = build_maskavg_visual_proto(args)

    t_dataset = build_dataset(args, args.t_pth)
    blocks = build_class_part_blocks_from_dataset(t_dataset, device=torch.device(args.device))

    print("=" * 170)
    print("Structure retrieval: T_pre / T_post vs V_anchor and V_maskavg")
    print(f"T dataset : {args.t_pth}")
    print(f"V dataset : {args.v_pth}")
    print(f"ckpt      : {args.projector_ckpt}")
    print("=" * 170)
    print(
        f"{'class':<22} {'cat':<5} {'K':<4} "
        f"{'anc_min':<8} {'mask_min':<8} "
        f"{'pre-anc':<10} {'post-anc':<10} "
        f"{'pre-mask':<10} {'post-mask':<10} "
        f"{'prepost':<10}"
    )

    rows = []
    for b in blocks:
        part_ids = b["part_ids"].long().to(args.device)
        if part_ids.numel() < 3:
            continue

        if (cnt_anchor[part_ids] < args.min_proto_count).any():
            continue
        if (cnt_maskavg[part_ids] < args.min_proto_count).any():
            continue

        preT = b["part_text"].float().to(args.device)
        postT = model.project_clip_txt(preT).float()

        Va = V_anchor[part_ids].float()
        Vm = V_maskavg[part_ids].float()

        pre_anchor = structure_retrieval_metric(preT, Va)
        post_anchor = structure_retrieval_metric(postT, Va)
        pre_mask = structure_retrieval_metric(preT, Vm)
        post_mask = structure_retrieval_metric(postT, Vm)
        prepost = structure_retrieval_metric(preT, postT)

        rows.append((pre_anchor, post_anchor, pre_mask, post_mask, prepost))

        print(
            f"{b.get('class_name', ''):<22} {int(b['category_id']):<5d} {int(part_ids.numel()):<4d} "
            f"{int(cnt_anchor[part_ids].min().item()):<8d} {int(cnt_maskavg[part_ids].min().item()):<8d} "
            f"{pre_anchor:<10.6f} {post_anchor:<10.6f} "
            f"{pre_mask:<10.6f} {post_mask:<10.6f} "
            f"{prepost:<10.6f}"
        )

    if rows:
        valid = []
        for r in rows:
            t = torch.tensor(r)
            if torch.isfinite(t).all():
                valid.append(r)

        if valid:
            means = [sum(r[i] for r in valid) / len(valid) for i in range(5)]
            print("-" * 170)
            print(
                f"{'MEAN':<42} "
                f"{means[0]:<10.6f} {means[1]:<10.6f} "
                f"{means[2]:<10.6f} {means[3]:<10.6f} "
                f"{means[4]:<10.6f}"
            )
            print(f"valid classes for mean: {len(valid)}")

    print("\nMetric definition follows metric.py::structure_retrieval:")
    print("for each feature set, compute pairwise cosine self-similarity, remove diagonal,")
    print("row-center and normalize each structural vector, then retrieve by argmax.")
    print("A score of 1.0 means every node's structural-neighborhood vector retrieves its counterpart.")


if __name__ == "__main__":
    main()
