from __future__ import annotations

import argparse
import importlib
import json
import os

import torch
import yaml

from src.dataset_joint_with_part_anchoraudit import DinoClipJointDataset
from src.train_util_gw import do_train_gw


device = "cuda" if torch.cuda.is_available() else "cpu"


def train_and_eval_gw(
    config_file,
    train_dataset,
    val_dataset,
    optimizer="AdamW",
    weight_decay=0.05,
    scheduler="linear",
    warmup=0,
    name_pedix="",
    init_weights="",
    miou_eval_script=None,
    miou_eval_cfg=None,
    miou_eval_base_cfg=None,
    miou_result_dir="segmentation_results",
    miou_result_json_name=None,
    miou_bench_key=None,
    miou_extra_opts=None,
):
    out_dir = "weights"
    os.makedirs(out_dir, exist_ok=True)

    model_name = os.path.basename(config_file).split(".")[0]
    if name_pedix:
        model_name += f"_{name_pedix}"
    out_path = os.path.join(out_dir, model_name)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])
    model.to(device)

    if init_weights:
        print(f"Loading init weights from {init_weights}")
        ckpt = torch.load(init_weights, map_location="cpu")
        ret = model.load_state_dict(ckpt, strict=False)
        if ret is not None:
            print("Missing keys:", getattr(ret, "missing_keys", []))
            print("Unexpected keys:", getattr(ret, "unexpected_keys", []))

    print(model)

    model, train_history, val_history = do_train_gw(
        model,
        train_dataset,
        val_dataset,
        config["train"],
        optimizer_name=optimizer,
        weight_decay=weight_decay,
        scheduler_name=scheduler,
        warmup=warmup,
        eval_proj_name=model_name,
        miou_eval_script=miou_eval_script,
        miou_eval_cfg=miou_eval_cfg,
        miou_eval_base_cfg=miou_eval_base_cfg,
    )

    torch.save(model.state_dict(), f"{out_path}.pth")
    print(f"Saved model at {out_path}.pth")

    with open(f"{out_path}_history.json", "w") as f:
        json.dump({"train": train_history, "val": val_history}, f, indent=2)
    print(f"Saved training history at {out_path}_history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)

    parser.add_argument("--obj_feature_name", type=str, default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", type=str, default="patch_tokens")
    parser.add_argument("--obj_text_name", type=str, default="ann_feats")
    parser.add_argument("--part_text_name", type=str, default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", type=str, default=None)

    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--warmup", type=int, default=0)

    parser.add_argument("--name_pedix", type=str, default="")
    parser.add_argument("--init_weights", type=str, default="")

    parser.add_argument("--miou_eval_script", type=str, default="src/open_vocabulary_segmentation/main.py")
    parser.add_argument("--miou_eval_cfg", type=str, required=True)
    parser.add_argument("--miou_eval_base_cfg", type=str, required=True)
    parser.add_argument("--miou_result_dir", type=str, default="segmentation_results")
    parser.add_argument("--miou_result_json_name", type=str, default=None)
    parser.add_argument("--miou_bench_key", type=str, default=None)
    parser.add_argument("--miou_extra_opts", nargs="*", default=None)

    args = parser.parse_args()

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get("dataset", {})
    min_obj_area_ratio = float(dataset_cfg.get("min_obj_area_ratio", 0.0))

    is_train_wds = ".tar" in args.train_dataset
    is_val_wds = ".tar" in args.val_dataset

    train_dataset = DinoClipJointDataset(
        args.train_dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=is_train_wds,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
    )

    val_dataset = DinoClipJointDataset(
        args.val_dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=is_val_wds,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=0.0,
    )

    train_and_eval_gw(
        args.model_config,
        train_dataset,
        val_dataset,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup=args.warmup,
        name_pedix=args.name_pedix,
        init_weights=args.init_weights,
        miou_eval_script=args.miou_eval_script,
        miou_eval_cfg=args.miou_eval_cfg,
        miou_eval_base_cfg=args.miou_eval_base_cfg,
        miou_result_dir=args.miou_result_dir,
        miou_result_json_name=args.miou_result_json_name,
        miou_bench_key=args.miou_bench_key,
        miou_extra_opts=args.miou_extra_opts,
    )
