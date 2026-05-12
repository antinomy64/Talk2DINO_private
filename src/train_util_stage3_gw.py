"""Training utilities for Stage 3 GW refinement."""

from __future__ import annotations

from copy import deepcopy
import json
import os
import random
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_joint_with_part_anchoraudit import joint_collate_fn
from src.loss_stage3_gw import (
    Stage3GWLoss,
    build_class_part_blocks_from_dataset,
    build_stage2_visual_prototypes,
)


def set_seed(seed: int):
    print(f"Setting seed {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def _move_joint_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _to_float_or_nan(v) -> float:
    if torch.is_tensor(v):
        if v.numel() == 0:
            return float("nan")
        return float(v.detach().float().cpu().reshape(-1)[0].item())
    try:
        return float(v)
    except Exception:
        return float("nan")


def _mean_dict(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}

    keys = sorted({k for d in list_of_dicts for k in d.keys()})
    out = {}

    if "anchor_total_valid_parts" in keys and "anchor_total_hits" in keys:
        total_valid = 0.0
        total_hits = 0.0
        for d in list_of_dicts:
            total_valid += _to_float_or_nan(d.get("anchor_total_valid_parts", 0.0))
            total_hits += _to_float_or_nan(d.get("anchor_total_hits", 0.0))
        out["anchor_total_valid_parts"] = total_valid
        out["anchor_total_hits"] = total_hits
        out["anchor_hit_rate"] = 0.0 if total_valid <= 0 else total_hits / total_valid

    for k in keys:
        if k in {"anchor_hit_rate", "anchor_total_valid_parts", "anchor_total_hits"}:
            continue
        vals = []
        for d in list_of_dicts:
            if k not in d:
                continue
            val = _to_float_or_nan(d[k])
            if np.isfinite(val):
                vals.append(val)
        out[k] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

    return out


def train_stage3_gw(
    model,
    train_dataloader,
    criterion,
    optimizer,
    scheduler=None,
    epoch=0,
    audit_anchor_every: int = 0,
    audit_structure_every: int = 1,
):
    model.train()
    device = next(model.parameters()).device
    prev_iter = epoch * len(train_dataloader)
    running = []

    pbar = tqdm(train_dataloader)
    for n_batch, batch in enumerate(pbar):
        batch = _move_joint_batch_to_device(batch, device)

        if scheduler is not None:
            scheduler(n_batch + prev_iter)

        do_anchor_audit = audit_anchor_every > 0 and (n_batch % audit_anchor_every == 0)
        do_structure_audit = audit_structure_every > 0 and (n_batch % audit_structure_every == 0)

        losses = criterion(
            batch,
            do_anchor_audit=do_anchor_audit,
            do_structure_audit=do_structure_audit,
        )
        total_loss = losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running.append(losses)
        desc = (
            f"train total={losses['total'].item():.4f} "
            f"obj={losses['obj'].item():.6f} "
            f"gw={losses['gw'].item():.4f} "
            f"struct={losses['struct'].item():.4f}"
        )
        if do_anchor_audit:
            desc += f" anchor_post={losses['anchor_hit_rate_post'].item():.4f}"
        if do_structure_audit and "audit_spear_post_text_vs_visual" in losses:
            desc += (
                f" spear_preV={losses['audit_spear_pre_text_vs_visual'].item():.3f}"
                f" spear_postV={losses['audit_spear_post_text_vs_visual'].item():.3f}"
                f" str_preV={losses['audit_strret_pre_text_vs_visual'].item():.3f}"
                f" str_postV={losses['audit_strret_post_text_vs_visual'].item():.3f}"
            )
        pbar.set_description(desc)

    return _mean_dict(running)


def train_stage3_gw_global(
    model,
    criterion,
    optimizer,
    scheduler=None,
    epoch: int = 0,
    steps_per_epoch: int = 1,
    audit_structure_every: int = 1,
):
    model.train()
    running = []
    steps_per_epoch = max(1, int(steps_per_epoch))
    prev_iter = epoch * steps_per_epoch

    pbar = tqdm(range(steps_per_epoch))
    for step in pbar:
        if scheduler is not None:
            scheduler(step + prev_iter)

        do_structure_audit = audit_structure_every > 0 and (step % audit_structure_every == 0)
        losses = criterion.global_forward(do_structure_audit=do_structure_audit)
        total_loss = losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running.append(losses)
        desc = f"gw-only total={losses['total'].item():.4f} gw={losses['gw'].item():.4f} struct={losses['struct'].item():.4f}"
        if do_structure_audit and "audit_spear_post_text_vs_visual" in losses:
            desc += (
                f" spear_preV={losses['audit_spear_pre_text_vs_visual'].item():.3f}"
                f" spear_postV={losses['audit_spear_post_text_vs_visual'].item():.3f}"
                f" str_preV={losses['audit_strret_pre_text_vs_visual'].item():.3f}"
                f" str_postV={losses['audit_strret_post_text_vs_visual'].item():.3f}"
            )
        pbar.set_description(desc)

    return _mean_dict(running)


@torch.no_grad()
def validate_stage3_gw(
    model,
    val_dataloader,
    criterion,
    audit_anchor_every: int = 0,
    audit_structure_every: int = 1,
):
    model.eval()
    device = next(model.parameters()).device
    running = []

    pbar = tqdm(val_dataloader)
    for n_batch, batch in enumerate(pbar):
        batch = _move_joint_batch_to_device(batch, device)
        do_anchor_audit = audit_anchor_every > 0 and (n_batch % audit_anchor_every == 0)
        do_structure_audit = audit_structure_every > 0 and (n_batch % audit_structure_every == 0)
        losses = criterion(
            batch,
            do_anchor_audit=do_anchor_audit,
            do_structure_audit=do_structure_audit,
        )
        running.append(losses)
        desc = (
            f"val total={losses['total'].item():.4f} "
            f"obj={losses['obj'].item():.6f} "
            f"gw={losses['gw'].item():.4f} "
            f"struct={losses['struct'].item():.4f}"
        )
        if do_anchor_audit:
            desc += f" anchor_post={losses['anchor_hit_rate_post'].item():.4f}"
        if do_structure_audit and "audit_spear_post_text_vs_visual" in losses:
            desc += (
                f" spear_preV={losses['audit_spear_pre_text_vs_visual'].item():.3f}"
                f" spear_postV={losses['audit_spear_post_text_vs_visual'].item():.3f}"
                f" str_preV={losses['audit_strret_pre_text_vs_visual'].item():.3f}"
                f" str_postV={losses['audit_strret_post_text_vs_visual'].item():.3f}"
            )
        pbar.set_description(desc)

    return _mean_dict(running)


def _extract_miou_from_result_json(result_json_path: str, bench_key: Optional[str] = None) -> float:
    if not os.path.exists(result_json_path):
        raise FileNotFoundError(f"mIoU result json not found: {result_json_path}")

    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if bench_key is not None:
        if bench_key not in data:
            raise KeyError(
                f"bench key '{bench_key}' not found in {result_json_path}; keys={list(data.keys())}"
            )
        return float(data[bench_key])

    if len(data) == 1:
        return float(next(iter(data.values())))
    if "avg_miou" in data:
        return float(data["avg_miou"])
    if "voc116_obj" in data:
        return float(data["voc116_obj"])
    if "voc116_part" in data:
        return float(data["voc116_part"])

    for value in data.values():
        try:
            return float(value)
        except Exception:
            pass

    raise RuntimeError(f"Could not extract numeric mIoU from {result_json_path}: {data}")


def evaluate_object_miou_subprocess(
    model,
    proj_name: str,
    eval_script: str,
    eval_cfg: str,
    eval_base_cfg: str,
    result_dir: str = "segmentation_results",
    result_json_name: Optional[str] = None,
    bench_key: Optional[str] = None,
    extra_opts: Optional[List[str]] = None,
):
    os.makedirs("weights", exist_ok=True)
    ckpt_path = os.path.join("weights", f"{proj_name}.pth")
    torch.save(model.state_dict(), ckpt_path)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=29517",
        eval_script,
        "--eval",
        "--eval_cfg",
        eval_cfg,
        "--eval_base_cfg",
        eval_base_cfg,
        "--opts",
        f"model.proj_name={proj_name}",
    ]
    if extra_opts:
        cmd.extend(extra_opts)

    print("[mIoU eval cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, check=True)

    json_name = result_json_name if result_json_name is not None else proj_name
    result_json_path = os.path.join(result_dir, f"{json_name}.json")
    miou = _extract_miou_from_result_json(result_json_path, bench_key=bench_key)

    return {
        "obj_eval_miou": float(miou),
        "obj_eval_ckpt_path": ckpt_path,
        "obj_eval_result_json": result_json_path,
        "obj_eval_subprocess_returncode": int(proc.returncode),
    }


def _iter_dataset_samples(dataset):
    """Yield sample dictionaries from DinoClipJointDataset.data.

    In the current pth-backed dataset, dataset.data may be either:
      - a list of sample dictionaries, or
      - a dict mapping integer ids to sample dictionaries.

    Iterating a dict directly yields keys, which caused:
      TypeError: 'int' object is not subscriptable
    """
    if not hasattr(dataset, "data"):
        raise AttributeError("Expected dataset to have .data")

    data = dataset.data
    if isinstance(data, dict):
        return data.values()
    return data


def _infer_num_parts(train_dataset) -> int:
    max_pid = -1
    for sample in _iter_dataset_samples(train_dataset):
        pids = sample["part_category_id"]
        if torch.is_tensor(pids) and pids.numel() > 0:
            max_pid = max(max_pid, int(pids.max().item()))
        elif not torch.is_tensor(pids) and len(pids) > 0:
            max_pid = max(max_pid, int(max(pids)))
    if max_pid < 0:
        raise RuntimeError("Could not infer num_parts from train_dataset.data")
    return max_pid + 1


def do_train_stage3_gw(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    seed: int = 123,
    optimizer_name: str = "AdamW",
    weight_decay: float = 0.05,
    scheduler_name: str = "linear",
    warmup: int = 0,
    eval_proj_name: str = "",
    miou_eval_script: Optional[str] = None,
    miou_eval_cfg: Optional[str] = None,
    miou_eval_base_cfg: Optional[str] = None,
):
    device = next(model.parameters()).device
    set_seed(seed)

    lr = train_cfg["lr"]
    num_epochs = train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    shuffle = train_cfg.get("shuffle", True)

    obj_ltype = train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce"))
    obj_margin = train_cfg.get("margin", 0.2)
    obj_max_violation = train_cfg.get("max_violation", True)

    lambda_obj = float(train_cfg.get("lambda_obj", 0.0))
    lambda_gw = float(train_cfg.get("lambda_gw", 0.0))
    lambda_struct = float(train_cfg.get("lambda_struct", 0.0))
    patch_temperature = float(train_cfg.get("patch_temperature", 0.07))
    em_iters = int(train_cfg.get("em_iters", 1))
    gw_epsilon = float(train_cfg.get("gw_epsilon", 0.05))
    gw_max_iter = int(train_cfg.get("gw_max_iter", 20))
    sinkhorn_iter = int(train_cfg.get("sinkhorn_iter", 50))
    min_proto_count = int(train_cfg.get("min_proto_count", 1))
    audit_anchor_every = int(train_cfg.get("audit_anchor_every", 0))
    audit_structure_every = int(train_cfg.get("audit_structure_every", 1))
    gw_only_steps_per_epoch = int(train_cfg.get("gw_only_steps_per_epoch", 1))
    stage2_visual_source = str(train_cfg.get("stage2_visual_source", "zpart")).lower()
    gw_only_mode = lambda_obj <= 0.0

    if not eval_proj_name:
        raise ValueError("eval_proj_name must be provided for mIoU evaluation.")
    if miou_eval_script is None or miou_eval_cfg is None or miou_eval_base_cfg is None:
        raise ValueError("miou_eval_script / miou_eval_cfg / miou_eval_base_cfg must all be provided.")

    print(
        "[stage3 gw config] "
        f"lambda_obj={lambda_obj}, "
        f"lambda_gw={lambda_gw}, "
        f"patch_temperature={patch_temperature}, "
        f"em_iters={em_iters}, "
        f"gw_epsilon={gw_epsilon}, "
        f"gw_max_iter={gw_max_iter}, "
        f"audit_anchor_every={audit_anchor_every}, "
        f"audit_structure_every={audit_structure_every}, "
        f"gw_only_mode={gw_only_mode}, "
        f"gw_only_steps_per_epoch={gw_only_steps_per_epoch}, "
        f"stage2_visual_source={stage2_visual_source}, "
        f"min_obj_area_ratio={getattr(train_dataset, 'min_obj_area_ratio', 0.0)}"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        collate_fn=joint_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=joint_collate_fn,
    )

    # Stage 2: build global visual prototypes in memory.
    # Do not use train_cfg.get("num_parts", _infer_num_parts(...)) here:
    # Python evaluates the default argument eagerly, so _infer_num_parts would
    # still run even when num_parts is present in the config.
    num_parts_cfg = train_cfg.get("num_parts", None)
    if num_parts_cfg is None:
        num_parts = _infer_num_parts(train_dataset)
    else:
        num_parts = int(num_parts_cfg)
    print(f"[Stage2] Building visual prototypes in memory, num_parts={num_parts}, visual_source={stage2_visual_source}")
    proto_pack = build_stage2_visual_prototypes(
        model=model,
        dataloader=train_dataloader,
        num_parts=num_parts,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        visual_source=stage2_visual_source,
    )

    visual_proto = proto_pack["visual_proto"]
    proto_count = proto_pack["proto_count"]

    class_blocks = build_class_part_blocks_from_dataset(train_dataset, device=device)

    criterion = Stage3GWLoss(
        sim_model=model,
        visual_proto=visual_proto,
        class_blocks=class_blocks,
        obj_ltype=obj_ltype,
        obj_margin=obj_margin,
        obj_max_violation=obj_max_violation,
        lambda_obj=lambda_obj,
        lambda_gw=lambda_gw,
        lambda_struct=lambda_struct,
        gw_epsilon=gw_epsilon,
        gw_max_iter=gw_max_iter,
        sinkhorn_iter=sinkhorn_iter,
        min_proto_count=min_proto_count,
        proto_count=proto_count,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    )

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented")

    if gw_only_mode:
        total_steps = max(1, gw_only_steps_per_epoch) * num_epochs
    else:
        total_steps = len(train_dataloader) * num_epochs
    if scheduler_name == "linear" and warmup == 0:
        scheduler = None
    elif scheduler_name == "linear" and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == "cosine":
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    else:
        scheduler = None

    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")

        if gw_only_mode:
            train_metrics = train_stage3_gw_global(
                model,
                criterion,
                optimizer,
                scheduler=scheduler,
                epoch=epoch,
                steps_per_epoch=gw_only_steps_per_epoch,
                audit_structure_every=audit_structure_every,
            )
        else:
            train_metrics = train_stage3_gw(
                model,
                train_dataloader,
                criterion,
                optimizer,
                scheduler=scheduler,
                epoch=epoch,
                audit_anchor_every=audit_anchor_every,
                audit_structure_every=audit_structure_every,
            )
        val_metrics = validate_stage3_gw(
            model,
            val_dataloader,
            criterion,
            audit_anchor_every=audit_anchor_every,
            audit_structure_every=audit_structure_every,
        )

        obj_eval_metrics = {
            "obj_eval_miou": float("nan"),
            "obj_eval_ckpt_path": "",
            "obj_eval_result_json": "",
            "obj_eval_subprocess_returncode": -1,
            "obj_eval_miou_delta_vs_baseline": float("nan"),
        }

        val_metrics = {**val_metrics, **obj_eval_metrics}
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"train_total={train_metrics['total']:.4f}, "
            f"train_gw={train_metrics.get('gw', 0.0):.4f}, "
            f"train_struct={train_metrics.get('struct', 0.0):.4f}, "
            f"val_total={val_metrics['total']:.4f}, "
            f"val_gw={val_metrics.get('gw', 0.0):.4f}, "
            f"val_struct={val_metrics.get('struct', 0.0):.4f}, "
            f"obj_eval_miou=skipped"
        )

    return model, train_history, val_history
