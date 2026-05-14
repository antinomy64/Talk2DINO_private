"""
Stage 3 hard-bijective GW-matched point alignment for Talk2DINO.

Final Stage3 objective implemented here:

    Z = projector(T)

    L_gw     = use hard-bijective GW to find a permutation pi between
               D(Z) and D(V), then directly pull each projected text point
               to its matched visual prototype:
                   mean_i [1 - cos(Z_i, V_{pi(i)})]

    L_struct = structure preservation loss between projector input and output:
                   MSE(D(Z), D(T))

    L_total  = lambda_obj * L_obj + lambda_gw * L_gw + lambda_struct * L_struct
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from src.loss import ContrastiveLoss
from src.loss_joint import JointObjPartLoss


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def compute_relative_scores(local_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute the same relative part-patch scores used by loss_joint.py.

    local_scores: [K, M], scores for K parts over M object-internal patches.
    relative_score(k, m) = score(k, m) - best_score(other_part, m)
    """
    k, _ = local_scores.shape
    if k <= 1:
        return local_scores

    top2_vals, top2_idx = torch.topk(local_scores, k=min(2, k), dim=0)
    best_vals = top2_vals[0]
    best_idx = top2_idx[0]
    second_vals = top2_vals[1]

    row_ids = torch.arange(k, device=local_scores.device)[:, None]
    is_top1 = row_ids == best_idx[None, :]
    best_other = torch.where(is_top1, second_vals[None, :], best_vals[None, :])
    return local_scores - best_other


@torch.no_grad()
def extract_z_part_from_batch(
    model: nn.Module,
    batch: Dict,
    patch_temperature: float = 0.07,
    em_iters: int = 1,
    anchor_helper: JointObjPartLoss | None = None,
    return_anchor_tokens: bool = False,
):
    """
    Extract pseudo part visual features from one batch by reusing the exact
    Stage-1 anchor/prototype routine.

    When return_anchor_tokens=True, also return the selected single anchor patch
    tokens and their validity mask.
    """
    device = next(model.parameters()).device

    part_text_feat = batch["part_text_feat"].to(device).float()
    patch_tokens = batch["patch_tokens"].to(device).float()
    obj_mask_patch = batch["obj_mask_patch"].to(device).bool()
    part_valid_mask = batch["part_valid_mask"].to(device).bool()

    if anchor_helper is None:
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

    part_proj = model.project_clip_txt(part_text_feat)  # [B, K, D]
    part_proj = anchor_helper._safe_normalize(part_proj, dim=-1)
    patch_tokens = anchor_helper._safe_normalize(patch_tokens, dim=-1)

    # Match Stage-1 JointObjPartLoss.forward exactly.
    abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / float(patch_temperature)
    abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    dummy_part_gt_mask_patch = torch.zeros(
        part_valid_mask.shape[0],
        part_valid_mask.shape[1],
        patch_tokens.shape[1],
        dtype=torch.bool,
        device=device,
    )

    if return_anchor_tokens:
        z_part, _, _, anchor_tokens, anchor_valid = anchor_helper._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=dummy_part_gt_mask_patch,
            num_iters=em_iters,
            return_anchor_tokens=True,
        )
        return z_part, anchor_tokens, anchor_valid

    z_part, _, _ = anchor_helper._anchor_proto_em_pool(
        patch_tokens=patch_tokens,
        abs_logits=abs_logits,
        obj_mask_patch=obj_mask_patch,
        part_valid_mask=part_valid_mask,
        part_gt_mask_patch=dummy_part_gt_mask_patch,
        num_iters=em_iters,
    )
    return z_part


@torch.no_grad()
def build_stage2_visual_prototypes(
    model: nn.Module,
    dataloader,
    num_parts: int,
    patch_temperature: float = 0.07,
    em_iters: int = 1,
    visual_source: str = "zpart",
) -> Dict[str, torch.Tensor]:
    """
    Build global visual prototypes in memory.

    visual_source:
      - "zpart": mean Stage-1 z_part pseudo visual features by part_category_id
      - "anchor": mean selected single anchor patch tokens by part_category_id
    """
    device = next(model.parameters()).device
    model.eval()

    visual_source = str(visual_source).lower()
    if visual_source not in {"zpart", "anchor"}:
        raise ValueError(f"visual_source must be 'zpart' or 'anchor', got {visual_source}")

    proto_sum = None
    proto_count = torch.zeros(num_parts, device=device)

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

    for batch in tqdm(dataloader, total=len(dataloader), desc="Build global visual prototypes"):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        batch = moved

        if visual_source == "anchor":
            _, anchor_tokens, anchor_valid = extract_z_part_from_batch(
                model=model,
                batch=batch,
                patch_temperature=patch_temperature,
                em_iters=em_iters,
                anchor_helper=anchor_helper,
                return_anchor_tokens=True,
            )
            feat_to_accumulate = anchor_tokens
        else:
            z_part = extract_z_part_from_batch(
                model=model,
                batch=batch,
                patch_temperature=patch_temperature,
                em_iters=em_iters,
                anchor_helper=anchor_helper,
            )
            anchor_valid = None
            feat_to_accumulate = z_part

        part_ids = batch["part_category_id"].long()       # [B, K]
        part_valid = batch["part_valid_mask"].bool()      # [B, K]
        if anchor_valid is not None:
            part_valid = part_valid & anchor_valid.bool()

        if proto_sum is None:
            dim = feat_to_accumulate.shape[-1]
            proto_sum = torch.zeros(num_parts, dim, device=device)

        bsz, max_k = part_ids.shape
        for b in range(bsz):
            for k in range(max_k):
                if not bool(part_valid[b, k]):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0 or pid >= num_parts:
                    continue
                proto_sum[pid] += feat_to_accumulate[b, k]
                proto_count[pid] += 1.0

    if proto_sum is None:
        raise RuntimeError("No prototypes were accumulated. Check dataloader and dataset fields.")

    visual_proto = proto_sum / proto_count.clamp_min(1.0)[:, None]
    visual_proto = safe_normalize(visual_proto, dim=-1)

    return {
        "visual_proto": visual_proto.detach(),
        "proto_count": proto_count.detach(),
        "visual_source": visual_source,
    }


def build_class_part_blocks_from_dataset(dataset, device: torch.device) -> List[Dict]:
    blocks_by_cat = {}

    if not hasattr(dataset, "data"):
        raise AttributeError("Expected dataset to have .data. This helper is for pth-backed DinoClipJointDataset.")

    data_iter = dataset.data.values() if isinstance(dataset.data, dict) else dataset.data

    for sample in tqdm(data_iter, total=len(dataset.data), desc="Building class-part blocks"):
        category_id = int(sample["category_id"])
        if category_id in blocks_by_cat:
            continue

        part_ids = sample["part_category_id"]
        part_text = sample["part_text_feat"]

        if not torch.is_tensor(part_ids):
            part_ids = torch.tensor(part_ids, dtype=torch.long)
        if not torch.is_tensor(part_text):
            part_text = torch.tensor(part_text)

        if part_ids.numel() == 0:
            continue

        blocks_by_cat[category_id] = {
            "category_id": category_id,
            "class_name": sample.get("class_name", ""),
            "part_ids": part_ids.long().to(device),
            "part_text": part_text.float().to(device),
            "part_names": sample.get("part_class_name", []),
        }

    blocks = list(blocks_by_cat.values())
    blocks.sort(key=lambda x: int(x["category_id"]))
    return blocks


def pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    x = safe_normalize(x, dim=-1)
    sim = x @ x.T
    return (1.0 - sim).clamp_min(0.0)


def sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Legacy entropic OT helper retained for old imports/debug scripts. Not used by Stage3 training."""
    kernel = torch.exp(-cost / epsilon).clamp_min(eps)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(max_iter):
        u = a / (kernel @ v).clamp_min(eps)
        v = b / (kernel.T @ u).clamp_min(eps)
    return u[:, None] * kernel * v[None, :]


def gw_cost_matrix(C1: torch.Tensor, C2: torch.Tensor, T: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Squared-loss GW linearized cost matrix.

    M_ij = sum_k C1_ik^2 p_k + sum_l C2_jl^2 q_l - 2 * (C1 T C2^T)_ij
    """
    const1 = (C1 ** 2) @ p
    const2 = (C2 ** 2) @ q
    return const1[:, None] + const2[None, :] - 2.0 * C1 @ T @ C2.T


@torch.no_grad()
def entropic_gw(
    C1: torch.Tensor,
    C2: torch.Tensor,
    epsilon: float = 0.01,
    max_iter: int = 20,
    sinkhorn_iter: int = 50,
    init: str = "identity",
    hard: bool = False,
) -> torch.Tensor:
    """
    Legacy entropic GW solver retained for old debug scripts. Not used by Stage3 training.
    """
    if C1.shape != C2.shape:
        raise ValueError(f"GW expects same-size blocks, got {C1.shape} and {C2.shape}")
    k = C1.shape[0]
    device = C1.device
    p = torch.full((k,), 1.0 / k, device=device)
    q = torch.full((k,), 1.0 / k, device=device)
    if init == "identity":
        T = torch.eye(k, device=device) / float(k)
    elif init == "uniform":
        T = p[:, None] * q[None, :]
    else:
        raise ValueError(f"Unknown GW init: {init}")
    for _ in range(max_iter):
        cost = gw_cost_matrix(C1, C2, T, p, q)
        cost = cost - cost.min()
        T = sinkhorn(p, q, cost, epsilon=epsilon, max_iter=sinkhorn_iter)
    if hard:
        idx = T.argmax(dim=1)
        T_hard = torch.zeros_like(T)
        T_hard[torch.arange(k, device=device), idx] = 1.0 / float(k)
        T = T_hard
    return T.detach()


@torch.no_grad()
def make_perm_transport(row_to_col: torch.Tensor, k: int) -> torch.Tensor:
    """Strict bijective transport for a row->column permutation."""
    T = torch.zeros(k, k, device=row_to_col.device, dtype=torch.float32)
    T[torch.arange(k, device=row_to_col.device), row_to_col] = 1.0 / float(k)
    return T


def hard_gw_struct_objective(C1: torch.Tensor, C2: torch.Tensor, row_to_col: torch.Tensor) -> torch.Tensor:
    """
    Hard-bijective GW structural objective for a fixed permutation.

    row_to_col[i] = j means source node i is matched to target node j.
    This returns mean_{i,k} (C1[i,k] - C2[pi(i), pi(k)])^2.
    """
    C2_perm = C2[row_to_col][:, row_to_col]
    return F.mse_loss(C1, C2_perm)




@torch.no_grad()
def improve_perm_by_pair_swaps(
    C1: torch.Tensor,
    C2: torch.Tensor,
    perm: torch.Tensor,
    max_passes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Greedy 2-swap local improvement for a hard GW permutation.

    This is not brute-force enumeration over all permutations. It only checks
    pair swaps around the current hard bijection and accepts swaps that reduce
    the GW structural objective.
    """
    best_perm = perm.detach().clone()
    best_obj = hard_gw_struct_objective(C1, C2, best_perm).detach()
    k = int(best_perm.numel())

    for _ in range(max(1, int(max_passes))):
        improved = False
        for a in range(k):
            for b in range(a + 1, k):
                cand = best_perm.clone()
                tmp = cand[a].clone()
                cand[a] = cand[b]
                cand[b] = tmp
                obj = hard_gw_struct_objective(C1, C2, cand).detach()
                if obj.item() + 1e-12 < best_obj.item():
                    best_perm = cand
                    best_obj = obj
                    improved = True
        if not improved:
            break

    return best_perm, best_obj


@torch.no_grad()
def hard_bijective_gw_match(
    C1: torch.Tensor,
    C2: torch.Tensor,
    num_iters: int = 30,
    num_restarts: int = 50,
    include_identity: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find a hard one-to-one GW permutation with iterative linearization + Hungarian.

    This is not used as a feature-level alignment target. It is used only to
    choose the permutation pi for the structural objective:
        mean (C1 - C2[pi][:, pi])^2

    Returns:
        best_perm: [K], row i -> column best_perm[i]
        best_obj:  scalar objective value under detached C1/C2
    """
    if C1.shape != C2.shape:
        raise ValueError(f"Hard bijective GW expects same-size blocks, got {C1.shape} and {C2.shape}")

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:
        raise ImportError(
            "hard_bijective_gw_match requires scipy.optimize.linear_sum_assignment. "
            "Install scipy or run in the environment where scipy is available."
        ) from exc

    C1d = C1.detach()
    C2d = C2.detach()
    k = C1d.shape[0]
    device = C1d.device
    p = torch.full((k,), 1.0 / float(k), device=device)
    q = torch.full((k,), 1.0 / float(k), device=device)

    best_perm = None
    best_obj = None

    total_restarts = max(1, int(num_restarts))
    for restart_id in range(total_restarts):
        if restart_id == 0 and include_identity:
            perm = torch.arange(k, device=device)
        else:
            perm = torch.randperm(k, device=device)

        T = make_perm_transport(perm, k).to(device=device, dtype=C1d.dtype)

        for _ in range(max(1, int(num_iters))):
            cost = gw_cost_matrix(C1d, C2d, T, p, q)
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

            new_perm = torch.empty(k, dtype=torch.long, device=device)
            row_tensor = torch.tensor(row_ind, dtype=torch.long, device=device)
            col_tensor = torch.tensor(col_ind, dtype=torch.long, device=device)
            new_perm[row_tensor] = col_tensor

            new_T = make_perm_transport(new_perm, k).to(device=device, dtype=C1d.dtype)
            if torch.equal(new_perm, perm):
                perm = new_perm
                T = new_T
                break
            perm = new_perm
            T = new_T

        perm, obj = improve_perm_by_pair_swaps(
            C1d,
            C2d,
            perm,
            max_passes=max(1, int(num_iters) // 2),
        )
        if best_obj is None or obj.item() < best_obj.item():
            best_obj = obj.detach()
            best_perm = perm.detach().clone()

    return best_perm, best_obj


def upper_tri_vector(mat: torch.Tensor) -> torch.Tensor:
    """Return upper-triangular entries excluding the diagonal."""
    k = mat.shape[0]
    if k < 2:
        return mat.new_empty((0,))
    idx = torch.triu_indices(k, k, offset=1, device=mat.device)
    return mat[idx[0], idx[1]]


def rankdata_torch(x: torch.Tensor) -> torch.Tensor:
    """Simple rankdata for audit-only Spearman; ties are not specially averaged."""
    if x.numel() == 0:
        return x.float()
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def spearman_corr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spearman correlation between two 1-D tensors, audit-only."""
    x = x.flatten().float()
    y = y.flatten().float()
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.numel() < 2 or y.numel() < 2:
        return x.new_tensor(float("nan"))
    rx = rankdata_torch(x)
    ry = rankdata_torch(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if denom <= eps:
        return x.new_tensor(float("nan"))
    return (rx * ry).sum() / denom


def pairwise_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    x = safe_normalize(x.float(), dim=-1)
    return x @ x.T


def structure_spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spearman between pairwise cosine structures of a and b."""
    sim_a = pairwise_cosine_similarity(a)
    sim_b = pairwise_cosine_similarity(b)
    return spearman_corr_torch(upper_tri_vector(sim_a), upper_tri_vector(sim_b))


def structure_retrieval_metric(feat_1: torch.Tensor, feat_2: torch.Tensor) -> torch.Tensor:
    assert feat_1.shape[0] == feat_2.shape[0]
    n = feat_1.shape[0]
    if n <= 1:
        return feat_1.new_tensor(float("nan"))

    feat_1_ = safe_normalize(feat_1.float(), dim=-1)
    feat_2_ = safe_normalize(feat_2.float(), dim=-1)

    sim_1 = feat_1_ @ feat_1_.T
    sim_2 = feat_2_ @ feat_2_.T

    eye = torch.eye(n, dtype=torch.bool, device=feat_1.device)
    sim_1 = sim_1[~eye].view(n, -1)
    sim_2 = sim_2[~eye].view(n, -1)

    sim_1 = sim_1 - sim_1.mean(dim=-1, keepdim=True)
    sim_2 = sim_2 - sim_2.mean(dim=-1, keepdim=True)

    sim_1_norm = safe_normalize(sim_1, dim=-1)
    sim_2_norm = safe_normalize(sim_2, dim=-1)

    sim_1_2 = sim_1_norm @ sim_2_norm.T
    if torch.isnan(sim_1_2).any():
        return feat_1.new_tensor(float("nan"))

    idx = sim_1_2.argmax(dim=0)
    target = torch.arange(n, device=feat_1.device)
    return (idx == target).float().mean()


class Stage3GWLoss(nn.Module):
    """
    Stage 3 loss with exactly two Stage3 terms:
      1) hard-bijective GW-matched point alignment between projected T and V
      2) T-structure preservation before/after projection

    Object InfoNCE is kept only for backward compatibility with existing configs;
    set lambda_obj=0.0 for GW-only Stage3.
    """

    def __init__(
        self,
        sim_model: nn.Module,
        visual_proto: torch.Tensor,
        class_blocks: List[Dict],
        obj_ltype: str = "infonce",
        obj_margin: float = 0.2,
        obj_max_violation: bool = True,
        lambda_obj: float = 600.0,
        lambda_gw: float = 0.25,
        lambda_struct: float = 0.0,
        gw_epsilon: float = 0.05,       # retained for config compatibility; not used by hard GW
        gw_max_iter: int = 20,
        sinkhorn_iter: int = 50,        # reused as number of hard-GW random restarts
        min_proto_count: int = 1,
        proto_count: torch.Tensor | None = None,
        patch_temperature: float = 0.07,
        em_iters: int = 1,
    ):
        super().__init__()
        self.sim_model = sim_model
        self.visual_proto = safe_normalize(visual_proto.float(), dim=-1)
        self.class_blocks = class_blocks
        self.lambda_obj = float(lambda_obj)
        self.lambda_gw = float(lambda_gw)
        self.lambda_struct = float(lambda_struct)
        self.gw_epsilon = float(gw_epsilon)
        self.gw_max_iter = int(gw_max_iter)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.min_proto_count = int(min_proto_count)
        self.proto_count = proto_count
        self.patch_temperature = float(patch_temperature)
        self.em_iters = int(em_iters)

        self.obj_criterion = ContrastiveLoss(
            sim_model,
            margin=obj_margin,
            max_violation=obj_max_violation,
            ltype=obj_ltype,
        )

        self.anchor_helper = JointObjPartLoss(
            sim_model=sim_model,
            obj_ltype=obj_ltype,
            lambda_obj=0.0,
            lambda_inst=0.0,
            lambda_overlap=0.0,
            lambda_spear=0.0,
            patch_temperature=self.patch_temperature,
            em_iters=self.em_iters,
        )

        self.gw_blocks = []
        self._prepare_gw_blocks()

    @torch.no_grad()
    def _prepare_gw_blocks(self) -> None:
        """Cache per-object text/visual blocks only. Do not precompute transport."""
        self.gw_blocks = []

        for block in self.class_blocks:
            part_ids = block["part_ids"]
            part_text = block["part_text"].float()

            if part_ids.numel() < 2:
                continue

            if self.proto_count is not None:
                counts = self.proto_count[part_ids]
                if bool((counts < self.min_proto_count).any()):
                    print(
                        f"[Stage3GWLoss] skip block {block.get('class_name', '')}: "
                        f"prototype count below {self.min_proto_count}"
                    )
                    continue

            visual = self.visual_proto[part_ids]
            if not torch.isfinite(visual).all():
                print(f"[Stage3GWLoss] skip block {block.get('class_name', '')}: non-finite visual proto")
                continue

            self.gw_blocks.append(
                {
                    "category_id": int(block["category_id"]),
                    "class_name": block.get("class_name", ""),
                    "part_ids": part_ids.detach(),
                    "part_text": part_text.detach(),
                    "visual": visual.detach(),
                }
            )

        print(f"[Stage3GWLoss] valid GW blocks: {len(self.gw_blocks)}")
        for block in self.gw_blocks:
            print(
                f"  - {block['class_name']} "
                f"category_id={block['category_id']} parts={block['part_ids'].numel()}"
            )

    def _gw_loss(self) -> torch.Tensor:
        """
        Hard-bijective GW-matched point alignment loss.

        For each block:
          Z = projector(T)
          C_z = D(Z)
          C_v = D(V)
          pi = hard_bijective_gw_match(C_z.detach(), C_v.detach())

        Then use pi as hard pseudo correspondence and directly pull each
        projected text point to its matched visual prototype:
          loss = mean_i [1 - cos(Z_i, V_{pi(i)})]

        No soft transport, no weighted target.
        """
        losses = []

        for block in self.gw_blocks:
            part_text = block["part_text"]
            visual = block["visual"]

            projected_text = self.sim_model.project_clip_txt(part_text)
            projected_text = safe_normalize(projected_text, dim=-1)
            visual = safe_normalize(visual, dim=-1)

            C_proj = pairwise_cosine_distance(projected_text)
            C_visual = pairwise_cosine_distance(visual).detach()

            perm, _ = hard_bijective_gw_match(
                C_proj.detach(),
                C_visual,
                num_iters=self.gw_max_iter,
                num_restarts=max(1, self.sinkhorn_iter),
                include_identity=True,
            )

            # perm[i] = j means projected text row i is matched to visual row j.
            # Use the GW correspondence as hard pseudo supervision:
            # pull Z_i directly toward V_{perm[i]}.
            matched_visual = visual[perm].detach()
            loss = 1.0 - (projected_text * matched_visual).sum(dim=-1)
            loss = loss.mean()
            losses.append(loss)

        if len(losses) == 0:
            return self.visual_proto.new_tensor(0.0)

        return torch.stack(losses).mean()

    def _struct_loss(self) -> torch.Tensor:
        """Preserve T structure after projection: MSE(D(projector(T)), D(T))."""
        losses = []

        for block in self.gw_blocks:
            part_text = block["part_text"]
            if part_text.shape[0] < 2:
                continue

            projected_text = self.sim_model.project_clip_txt(part_text)
            projected_text = safe_normalize(projected_text, dim=-1)

            C_pre = pairwise_cosine_distance(part_text).detach()
            C_post = pairwise_cosine_distance(projected_text)
            losses.append(F.mse_loss(C_post, C_pre))

        if len(losses) == 0:
            return self.visual_proto.new_tensor(0.0)

        return torch.stack(losses).mean()

    @torch.no_grad()
    def _structure_audit(self) -> Dict[str, torch.Tensor]:
        values: Dict[str, List[torch.Tensor]] = {
            "audit_spear_pre_text_vs_visual": [],
            "audit_spear_post_text_vs_visual": [],
            "audit_strret_pre_text_vs_visual": [],
            "audit_strret_post_text_vs_visual": [],
        }

        for block in self.gw_blocks:
            part_text = block["part_text"]
            visual = safe_normalize(block["visual"], dim=-1)
            if part_text.shape[0] < 2:
                continue

            pre_text = safe_normalize(part_text.float(), dim=-1)
            post_text = self.sim_model.project_clip_txt(part_text)
            post_text = safe_normalize(post_text.float(), dim=-1)

            sim_pre = pairwise_cosine_similarity(pre_text)
            sim_post = pairwise_cosine_similarity(post_text)
            sim_vis = pairwise_cosine_similarity(visual)

            values["audit_spear_pre_text_vs_visual"].append(
                spearman_corr_torch(upper_tri_vector(sim_pre), upper_tri_vector(sim_vis))
            )
            values["audit_spear_post_text_vs_visual"].append(
                spearman_corr_torch(upper_tri_vector(sim_post), upper_tri_vector(sim_vis))
            )
            values["audit_strret_pre_text_vs_visual"].append(
                structure_retrieval_metric(pre_text, visual)
            )
            values["audit_strret_post_text_vs_visual"].append(
                structure_retrieval_metric(post_text, visual)
            )

        out: Dict[str, torch.Tensor] = {}
        device = self.visual_proto.device
        for key, vals in values.items():
            if len(vals) == 0:
                out[key] = torch.tensor(float("nan"), device=device)
                continue
            stacked = torch.stack([v.to(device).float() for v in vals])
            finite = torch.isfinite(stacked)
            out[key] = stacked[finite].mean() if finite.any() else torch.tensor(float("nan"), device=device)
        return out

    @torch.no_grad()
    def _anchor_audit_with_model(self, batch: Dict, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Batch-level anchor hit audit using real part_gt_mask_patch if available."""
        required = [
            "part_text_feat",
            "patch_tokens",
            "obj_mask_patch",
            "part_valid_mask",
            "part_gt_mask_patch",
        ]
        if any(k not in batch for k in required):
            z = self.visual_proto.new_tensor(0.0)
            return {
                "anchor_hit_rate": z,
                "anchor_total_valid_parts": z,
                "anchor_total_hits": z,
            }

        device = self.visual_proto.device
        part_text_feat = batch["part_text_feat"].to(device).float()
        patch_tokens = batch["patch_tokens"].to(device).float()
        obj_mask_patch = batch["obj_mask_patch"].to(device).bool()
        part_valid_mask = batch["part_valid_mask"].to(device).bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].to(device).bool()

        part_proj = model.project_clip_txt(part_text_feat)
        part_proj = safe_normalize(part_proj, dim=-1)
        patch_tokens = safe_normalize(patch_tokens, dim=-1)

        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / float(self.patch_temperature)
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        _, _, anchor_metrics = self.anchor_helper._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=self.em_iters,
        )
        return anchor_metrics

    @torch.no_grad()
    def _anchor_audit(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Current/post-projection anchor hit audit."""
        post = self._anchor_audit_with_model(batch, self.sim_model)
        return {
            "anchor_hit_rate_post": post["anchor_hit_rate"],
            "anchor_total_valid_parts_post": post["anchor_total_valid_parts"],
            "anchor_total_hits_post": post["anchor_total_hits"],
            # Backward-compatible alias used by old logging scripts.
            "anchor_hit_rate": post["anchor_hit_rate"],
            "anchor_total_valid_parts": post["anchor_total_valid_parts"],
            "anchor_total_hits": post["anchor_total_hits"],
        }

    def forward(
        self,
        batch=None,
        do_anchor_audit: bool = False,
        do_structure_audit: bool = False,
    ):
        device = self.visual_proto.device
        zero = torch.tensor(0.0, device=device)

        if batch is not None and self.lambda_obj > 0:
            obj_feat = batch["obj_feat"]
            obj_text_feat = batch["obj_text_feat"]
            obj_loss = self.obj_criterion(
                obj_feat,
                obj_text_feat,
                return_similarity_mat=False,
                self_attn_maps=None,
                cls=None,
                text_input_mask=None,
                text_argmax=None,
            )
        else:
            obj_loss = zero

        gw_loss = self._gw_loss() if self.lambda_gw > 0 else zero
        struct_loss = self._struct_loss() if self.lambda_struct > 0 else zero

        total = (
            self.lambda_obj * obj_loss
            + self.lambda_gw * gw_loss
            + self.lambda_struct * struct_loss
        )

        out = {
            "total": total,
            "obj": obj_loss.detach(),
            "gw": gw_loss.detach(),
            "struct": struct_loss.detach(),
            "inst": zero.detach(),
            "overlap": zero.detach(),
            "spear": zero.detach(),
        }

        if batch is not None and do_anchor_audit:
            out.update(self._anchor_audit(batch))

        if do_structure_audit:
            out.update(self._structure_audit())

        return out
