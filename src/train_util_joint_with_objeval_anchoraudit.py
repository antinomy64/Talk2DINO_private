
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
from src.loss_joint import JointObjPartLoss


def set_seed(seed: int):
    print(f'Setting seed {seed}...')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


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
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _mean_dict(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}
    keys = list(list_of_dicts[0].keys())
    out = {}

    # Anchor metrics should be aggregated by counts, not by simple batch mean.
    if "anchor_total_valid_parts" in keys and "anchor_total_hits" in keys:
        total_valid = 0.0
        total_hits = 0.0
        for d in list_of_dicts:
            v_valid = d["anchor_total_valid_parts"]
            v_hits = d["anchor_total_hits"]
            if torch.is_tensor(v_valid):
                total_valid += float(v_valid.detach().float().cpu().item())
            else:
                total_valid += float(v_valid)
            if torch.is_tensor(v_hits):
                total_hits += float(v_hits.detach().float().cpu().item())
            else:
                total_hits += float(v_hits)

        out["anchor_total_valid_parts"] = total_valid
        out["anchor_total_hits"] = total_hits
        out["anchor_hit_rate"] = 0.0 if total_valid <= 0 else total_hits / total_valid

    for k in keys:
        if k in {"anchor_hit_rate", "anchor_total_valid_parts", "anchor_total_hits"}:
            continue
        vals = []
        for d in list_of_dicts:
            v = d[k]
            if torch.is_tensor(v):
                vals.append(v.detach().float().cpu())
            else:
                vals.append(torch.tensor(float(v)))
        out[k] = torch.stack(vals).mean().item()
    return out


def train_joint(model, train_dataloader, criterion, optimizer, scheduler=None, epoch=0):
    model.train()
    device = next(model.parameters()).device
    prev_iter = epoch * len(train_dataloader)

    running = []
    pbar = tqdm(train_dataloader)
    for n_batch, batch in enumerate(pbar):
        batch = _move_joint_batch_to_device(batch, device)

        if scheduler is not None:
            scheduler(n_batch + prev_iter)

        losses = criterion(batch)
        total_loss = losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running.append(losses)
        pbar.set_description(
            f"train total={losses['total'].item():.4f} obj={losses['obj'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f} anchor={losses['anchor_hit_rate'].item():.4f}"
        )

    return _mean_dict(running)


@torch.no_grad()
def validate_joint(model, val_dataloader, criterion):
    model.eval()
    device = next(model.parameters()).device

    running = []
    pbar = tqdm(val_dataloader)
    for batch in pbar:
        batch = _move_joint_batch_to_device(batch, device)
        losses = criterion(batch)
        running.append(losses)
        pbar.set_description(
            f"val total={losses['total'].item():.4f} obj={losses['obj'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f} anchor={losses['anchor_hit_rate'].item():.4f}"
        )

    return _mean_dict(running)


def _extract_miou_from_result_json(result_json_path: str, bench_key: Optional[str] = None) -> float:
    if not os.path.exists(result_json_path):
        raise FileNotFoundError(f"mIoU result json not found: {result_json_path}")

    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if bench_key is not None:
        if bench_key not in data:
            raise KeyError(f"bench key '{bench_key}' not found in {result_json_path}; keys={list(data.keys())}")
        return float(data[bench_key])

    if len(data) == 1:
        return float(next(iter(data.values())))

    if "avg_miou" in data:
        return float(data["avg_miou"])

    if "voc116_obj" in data:
        return float(data["voc116_obj"])
    if "voc116_part" in data:
        return float(data["voc116_part"])
    for v in data.values():
        try:
            return float(v)
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
    miou_eval_port: int = 29517,
):
    os.makedirs("weights", exist_ok=True)
    ckpt_path = os.path.join("weights", f"{proj_name}.pth")
    torch.save(model.state_dict(), ckpt_path)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        f"--master_port={miou_eval_port}",
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


def do_train_joint(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    seed: int = 123,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.05,
    scheduler_name: str = 'linear',
    warmup: int = 0,
    eval_proj_name: str = "",
    miou_eval_script: Optional[str] = None,
    miou_eval_cfg: Optional[str] = None,
    miou_eval_base_cfg: Optional[str] = None,
    miou_result_dir: str = "segmentation_results",
    miou_result_json_name: Optional[str] = None,
    miou_bench_key: Optional[str] = None,
    miou_extra_opts: Optional[List[str]] = None,
    miou_eval_port: int = 29517,
):
    device = next(model.parameters()).device
    set_seed(seed)

    lr = train_cfg['lr']
    num_epochs = train_cfg['num_epochs']
    batch_size = train_cfg['batch_size']
    shuffle = train_cfg.get('shuffle', True)
    save_best_model = train_cfg.get('save_best_model', True)

    object_miou_max_drop = float(train_cfg.get('object_miou_max_drop', 0.5))
    select_best_by_miou = bool(train_cfg.get('select_best_by_miou', True))

    obj_ltype = train_cfg.get('obj_ltype', train_cfg.get('ltype', 'infonce'))
    obj_margin = train_cfg.get('margin', 0.2)
    obj_max_violation = train_cfg.get('max_violation', True)

    lambda_obj = train_cfg.get('lambda_obj', 1.0)
    lambda_inst = train_cfg.get('lambda_inst', 0.2)
    lambda_overlap = train_cfg.get('lambda_overlap', 0.05)
    lambda_spear = train_cfg.get('lambda_spear', 0.0)
    topk_ratio = train_cfg.get('topk_ratio', 0.1)
    patch_temperature = train_cfg.get('patch_temperature', 0.07)
    em_iters = int(train_cfg.get('em_iters', 3))

    if not eval_proj_name:
        raise ValueError("eval_proj_name must be provided for mIoU evaluation.")
    if miou_eval_script is None or miou_eval_cfg is None or miou_eval_base_cfg is None:
        raise ValueError("miou_eval_script / miou_eval_cfg / miou_eval_base_cfg must all be provided.")

    print(
        "[joint config] "
        f"lambda_obj={lambda_obj}, "
        f"lambda_inst={lambda_inst}, "
        f"lambda_overlap={lambda_overlap}, "
        f"lambda_spear={lambda_spear}, "
        f"em_iters={em_iters}, "
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

    criterion = JointObjPartLoss(
        model,
        obj_ltype=obj_ltype,
        obj_margin=obj_margin,
        obj_max_violation=obj_max_violation,
        lambda_obj=lambda_obj,
        lambda_inst=lambda_inst,
        lambda_overlap=lambda_overlap,
        lambda_spear=lambda_spear,
        topk_ratio=topk_ratio,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    )

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented")

    total_steps = len(train_dataloader) * num_epochs
    if scheduler_name == 'linear' and warmup == 0:
        scheduler = None
    elif scheduler_name == 'linear' and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == 'cosine':
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    else:
        scheduler = None

    train_history = []
    val_history = []
    best_model = deepcopy(model)
    best_val = None
    best_obj_miou = None

    baseline_obj_eval = evaluate_object_miou_subprocess(
        model=model,
        proj_name=eval_proj_name,
        eval_script=miou_eval_script,
        eval_cfg=miou_eval_cfg,
        eval_base_cfg=miou_eval_base_cfg,
        result_dir=miou_result_dir,
        result_json_name=miou_result_json_name,
        bench_key=miou_bench_key,
        extra_opts=miou_extra_opts,
        miou_eval_port=miou_eval_port,
    )
    baseline_obj_miou = baseline_obj_eval["obj_eval_miou"]
    print(f"[baseline object mIoU] miou={baseline_obj_miou:.4f}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")
        train_metrics = train_joint(model, train_dataloader, criterion, optimizer, scheduler=scheduler, epoch=epoch)
        val_metrics = validate_joint(model, val_dataloader, criterion)

        obj_eval_metrics = evaluate_object_miou_subprocess(
            model=model,
            proj_name=eval_proj_name,
            eval_script=miou_eval_script,
            eval_cfg=miou_eval_cfg,
            eval_base_cfg=miou_eval_base_cfg,
            result_dir=miou_result_dir,
            result_json_name=miou_result_json_name,
            bench_key=miou_bench_key,
            extra_opts=miou_extra_opts,
            miou_eval_port=miou_eval_port,
        )
        obj_eval_metrics["obj_eval_miou_delta_vs_baseline"] = float(obj_eval_metrics["obj_eval_miou"] - baseline_obj_miou)

        # No extra audit pass: anchor metrics already come from criterion.forward
        val_metrics = {
            **val_metrics,
            **obj_eval_metrics,
        }

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"train_total={train_metrics['total']:.4f}, val_total={val_metrics['total']:.4f}, "
            f"obj_eval_miou={val_metrics['obj_eval_miou']:.4f}, "
            f"miou_delta_vs_baseline={val_metrics['obj_eval_miou_delta_vs_baseline']:.4f}, "
            f"anchor_hit_rate={val_metrics.get('anchor_hit_rate', 0.0):.4f}"
        )

        current_obj_miou = val_metrics["obj_eval_miou"]
        obj_ok = current_obj_miou >= (baseline_obj_miou - object_miou_max_drop)

        if save_best_model:
            if select_best_by_miou:
                if obj_ok and (best_obj_miou is None or current_obj_miou > best_obj_miou):
                    best_obj_miou = current_obj_miou
                    best_val = val_metrics['total']
                    best_model = deepcopy(model)
                    print("Best model updated by object mIoU under guardrail.")
                elif not obj_ok:
                    print(
                        f"Skip best update because object mIoU dropped too much: "
                        f"{current_obj_miou:.4f} < {baseline_obj_miou - object_miou_max_drop:.4f}"
                    )
            else:
                if obj_ok and (best_val is None or val_metrics['total'] < best_val):
                    best_val = val_metrics['total']
                    best_model = deepcopy(model)
                    print("Best validation total loss under object mIoU guardrail, saving current best model in memory.")
                elif not obj_ok:
                    print(
                        f"Skip best update because object mIoU dropped too much: "
                        f"{current_obj_miou:.4f} < {baseline_obj_miou - object_miou_max_drop:.4f}"
                    )

    model = best_model if save_best_model else model
    return model, train_history, val_history
