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
from src.loss_gw import (
    build_stage2_visual_prototypes,
    build_class_part_blocks_from_dataset,
    structure_spearman,
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--t_pth", required=True, help="Dataset used to collect preT/postT class blocks.")
    parser.add_argument("--v_pth", required=True, help="Dataset used to build Stage2 anchor V prototypes.")
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
    train_cfg = cfg.get("train", {})
    patch_temperature = float(train_cfg.get("patch_temperature", 0.07))
    em_iters = int(train_cfg.get("em_iters", 1))

    # 1) collect V: use current repository Stage2 anchor prototype pipeline
    v_dataset = build_dataset(args, args.v_pth)
    v_loader = DataLoader(
        v_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )

    proto = build_stage2_visual_prototypes(
        model=model,
        dataloader=v_loader,
        num_parts=args.num_parts,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        visual_source="anchor",
    )
    V_all = proto["visual_proto"].to(args.device)
    proto_count = proto["proto_count"].to(args.device)

    # 1) collect preT/postT: use current repository class-block pipeline
    t_dataset = build_dataset(args, args.t_pth)
    blocks = build_class_part_blocks_from_dataset(t_dataset, device=torch.device(args.device))

    print("=" * 120)
    print("Spearman: T_pre / T_post vs Stage2 anchor V")
    print(f"T dataset : {args.t_pth}")
    print(f"V dataset : {args.v_pth}")
    print(f"ckpt      : {args.projector_ckpt}")
    print("=" * 120)
    print(f"{'class':<24} {'cat':<5} {'K':<4} {'min_cnt':<8} {'preV':<12} {'postV':<12} {'prepost':<12}")

    rows = []
    for b in blocks:
        part_ids = b["part_ids"].long().to(args.device)
        # Only evaluate object classes with at least 3 parts.
        # K=1/2 cannot provide a meaningful upper-triangle structure Spearman.
        if part_ids.numel() < 3:
            continue
        if (proto_count[part_ids] < args.min_proto_count).any():
            continue

        # Feature collection
        preT = b["part_text"].float().to(args.device)
        postT = model.project_clip_txt(preT).float()
        V = V_all[part_ids].float()

        # 2) call repository function to compute Spearman
        preV = float(structure_spearman(preT, V).cpu())
        postV = float(structure_spearman(postT, V).cpu())
        prepost = float(structure_spearman(preT, postT).cpu())

        rows.append((preV, postV, prepost))

        print(
            f"{b.get('class_name', ''):<24} {int(b['category_id']):<5d} "
            f"{int(part_ids.numel()):<4d} {int(proto_count[part_ids].min().item()):<8d} "
            f"{preV:<12.6f} {postV:<12.6f} {prepost:<12.6f}"
        )

    if rows:
        valid_rows = [
            r for r in rows
            if torch.isfinite(torch.tensor(r[0]))
            and torch.isfinite(torch.tensor(r[1]))
            and torch.isfinite(torch.tensor(r[2]))
        ]
        if valid_rows:
            m_preV = sum(r[0] for r in valid_rows) / len(valid_rows)
            m_postV = sum(r[1] for r in valid_rows) / len(valid_rows)
            m_prepost = sum(r[2] for r in valid_rows) / len(valid_rows)
            print("-" * 120)
            print(f"{'MEAN':<35} {'':<6} {m_preV:<12.6f} {m_postV:<12.6f} {m_prepost:<12.6f}")
            print(f"valid classes for mean: {len(valid_rows)}")


if __name__ == "__main__":
    main()
