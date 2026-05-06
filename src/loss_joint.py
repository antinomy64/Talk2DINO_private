import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import ContrastiveLoss


class JointObjPartLoss(nn.Module):
    """
    Joint loss for:
      1) object-level branch: keep original contrastive objective
      2) part-level branch: object-inside part supervision on patch tokens

    Absolute-space anchor + prototype EM version:
      - use ALL-PARTS bank from the dataset
      - DO NOT build text/vision residuals
      - DO NOT subtract object feature on either side
      - find unique anchors on relative scores computed from absolute part-patch logits
      - compute anchor hit metrics INSIDE forward
      - use anchors only to initialize prototypes
      - run prototype EM in the original projected patch-token space
      - compute final part vision features by mean pooling original patch tokens

    Spearman/structure regularization:
      A) preserve text-side PART-PART graph structure before/after projection
      B) preserve text-side PART-OBJ relation before/after projection

      A) part-part graph
         pre  graph: cosine(part_text_feat_i,  part_text_feat_j)
         post graph: cosine(project(part_text_feat_i), project(part_text_feat_j))

      B) part-obj relation
         pre  relation: cosine(part_text_feat_i,  obj_text_feat)
         post relation: cosine(project(part_text_feat_i), project(obj_text_feat))

      The final spear loss is the mean of the two valid terms above.
    """

    def __init__(
        self,
        sim_model,
        obj_ltype: str = "infonce",
        obj_margin: float = 0.2,
        obj_max_violation: bool = True,
        lambda_obj: float = 1.0,
        lambda_inst: float = 0.2,
        lambda_overlap: float = 0.05,
        lambda_spear: float = 0.0,
        topk_ratio: float = 0.1,
        patch_temperature: float = 0.07,
        eps: float = 1e-6,
        em_iters: int = 3,
    ):
        super().__init__()
        self.sim_model = sim_model
        self.obj_criterion = ContrastiveLoss(
            sim_model,
            margin=obj_margin,
            max_violation=obj_max_violation,
            ltype=obj_ltype,
        )
        self.lambda_obj = lambda_obj
        self.lambda_inst = lambda_inst
        self.lambda_overlap = lambda_overlap
        self.lambda_spear = lambda_spear
        self.topk_ratio = topk_ratio
        self.patch_temperature = patch_temperature
        self.eps = eps
        self.em_iters = int(em_iters)

    def _safe_normalize(self, x, dim=-1):
        return x / x.norm(dim=dim, keepdim=True).clamp_min(self.eps)

    def forward(self, batch):
        obj_feat = batch["obj_feat"]
        patch_tokens = batch["patch_tokens"]
        obj_text_feat = batch["obj_text_feat"]
        part_text_feat = batch["part_text_feat"]
        obj_mask_patch = batch["obj_mask_patch"]
        part_valid_mask = batch["part_valid_mask"]
        part_gt_mask_patch = batch["part_gt_mask_patch"]

        obj_loss = self.obj_criterion(
            obj_feat,
            obj_text_feat,
            return_similarity_mat=False,
            self_attn_maps=None,
            cls=None,
            text_input_mask=None,
            text_argmax=None,
        )

        zero = obj_loss.new_tensor(0.0)

        if part_text_feat.shape[1] == 0 or not part_valid_mask.any():
            total = self.lambda_obj * obj_loss
            return {
                "total": total,
                "obj": obj_loss.detach(),
                "inst": zero.detach(),
                "overlap": zero.detach(),
                "spear": zero.detach(),
                "anchor_hit_rate": zero.detach(),
                "anchor_total_valid_parts": zero.detach(),
                "anchor_total_hits": zero.detach(),
            }

        # Project text features into the same space as patch tokens.
        part_proj = self.sim_model.project_clip_txt(part_text_feat.float())   # [B, K, D]
        obj_proj = self.sim_model.project_clip_txt(obj_text_feat.float())     # [B, D]
        part_proj = self._safe_normalize(part_proj, dim=-1)
        obj_proj = self._safe_normalize(obj_proj, dim=-1)
        patch_tokens = self._safe_normalize(patch_tokens.float(), dim=-1)

        # Absolute part-patch score map inside the object.
        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / self.patch_temperature
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        z_part, proto_part, anchor_metrics = self._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=self.em_iters,
        )

        inst_loss = self._instance_consistency_loss(part_proj, z_part, part_valid_mask)

        overlap_loss = (
            self._soft_part_overlap_loss(
                abs_logits=abs_logits,
                obj_mask_patch=obj_mask_patch,
                part_valid_mask=part_valid_mask,
            )
            if self.lambda_overlap > 0
            else zero
        )

        # New structure-preserving "Spearman-style" loss:
        #   1) keep the part-part text graph stable before/after projection
        #   2) keep the part-obj relation stable before/after projection
        spear_loss = (
            self._combined_structure_spearman_surrogate_loss(
                obj_text_feat=obj_text_feat,
                part_text_feat=part_text_feat,
                obj_proj=obj_proj,
                part_proj=part_proj,
                part_valid_mask=part_valid_mask,
            )
            if self.lambda_spear > 0
            else zero
        )

        total = (
            self.lambda_obj * obj_loss
            + self.lambda_inst * inst_loss
            + self.lambda_overlap * overlap_loss
            + self.lambda_spear * spear_loss
        )

        return {
            "total": total,
            "obj": obj_loss.detach(),
            "inst": inst_loss.detach(),
            "overlap": overlap_loss.detach(),
            "spear": spear_loss.detach(),
            "anchor_hit_rate": anchor_metrics["anchor_hit_rate"].detach(),
            "anchor_total_valid_parts": anchor_metrics["anchor_total_valid_parts"].detach(),
            "anchor_total_hits": anchor_metrics["anchor_total_hits"].detach(),
        }

    def _compute_relative_scores(self, local_scores: torch.Tensor) -> torch.Tensor:
        Kb, Mb = local_scores.shape
        if Kb <= 1:
            return local_scores

        top2_vals, top2_idx = torch.topk(local_scores, k=min(2, Kb), dim=0)
        best_vals = top2_vals[0]
        best_idx = top2_idx[0]
        second_vals = top2_vals[1]

        row_ids = torch.arange(Kb, device=local_scores.device)[:, None]
        is_top1 = row_ids == best_idx[None, :]
        best_other = torch.where(is_top1, second_vals[None, :], best_vals[None, :])

        rel_scores = local_scores - best_other
        return rel_scores

    def _anchor_proto_em_pool(
        self,
        patch_tokens,
        abs_logits,
        obj_mask_patch,
        part_valid_mask,
        part_gt_mask_patch,
        num_iters=3,
    ):
        B, K, N = abs_logits.shape
        D = patch_tokens.shape[-1]
        z = patch_tokens.new_zeros((B, K, D))
        proto_part = patch_tokens.new_zeros((B, K, D))

        total_valid_parts = patch_tokens.new_tensor(0.0)
        total_anchor_hits = patch_tokens.new_tensor(0.0)

        for b in range(B):
            valid_patch_mask = obj_mask_patch[b]
            valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)

            if valid_part_idx.numel() == 0 or valid_patch_mask.sum() == 0:
                continue

            valid_patch_tokens = patch_tokens[b][valid_patch_mask]
            local_scores = abs_logits[b][valid_part_idx][:, valid_patch_mask]

            Kb, Mb = local_scores.shape
            if Mb == 0:
                continue

            rel_scores = self._compute_relative_scores(local_scores)
            flat_scores = rel_scores.reshape(-1)
            sorted_idx = torch.argsort(flat_scores, descending=True)

            anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=local_scores.device)
            patch_taken = torch.zeros((Mb,), dtype=torch.bool, device=local_scores.device)

            assigned_parts = 0
            for flat_id in sorted_idx:
                p_local = torch.div(flat_id, Mb, rounding_mode='floor')
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

            valid_patch_idx_global = torch.nonzero(valid_patch_mask, as_tuple=False).squeeze(1)
            anchor_idx_global = valid_patch_idx_global[anchor_idx_local]

            gt_masks = part_gt_mask_patch[b, valid_part_idx]
            hit_vec = gt_masks[torch.arange(Kb, device=gt_masks.device), anchor_idx_global]

            total_valid_parts += float(Kb)
            total_anchor_hits += float(hit_vec.long().sum().item())

            C = valid_patch_tokens[anchor_idx_local]

            assign = None
            for _ in range(max(int(num_iters), 1)):
                assign_scores = valid_patch_tokens @ C.T
                assign = assign_scores.argmax(dim=1)
                assign[anchor_idx_local] = torch.arange(Kb, device=assign.device)

                onehot = F.one_hot(assign, num_classes=Kb).float()
                count = onehot.sum(dim=0).clamp_min(1.0)
                proto_sum = onehot.T @ valid_patch_tokens
                C = proto_sum / count[:, None]
                C = self._safe_normalize(C, dim=-1)

            region_onehot = F.one_hot(assign, num_classes=Kb).float()
            region_count = region_onehot.sum(dim=0).clamp_min(1.0)
            region_sum = region_onehot.T @ valid_patch_tokens
            z_local = region_sum / region_count[:, None]
            z_local = self._safe_normalize(z_local, dim=-1)

            z[b, valid_part_idx] = z_local
            proto_part[b, valid_part_idx] = C

        hit_rate = total_anchor_hits / total_valid_parts.clamp_min(1.0)
        anchor_metrics = {
            "anchor_hit_rate": hit_rate,
            "anchor_total_valid_parts": total_valid_parts,
            "anchor_total_hits": total_anchor_hits,
        }
        return z, proto_part, anchor_metrics

    def _instance_consistency_loss(self, part_proj, z_part, part_valid_mask):
        cos = F.cosine_similarity(part_proj, z_part.detach(), dim=-1)
        loss = 1.0 - cos
        return self._masked_mean(loss, part_valid_mask)

    def _corr_loss(self, x, y):
        x = x - x.mean()
        y = y - y.mean()
        denom = (
            torch.sqrt((x ** 2).sum() + self.eps)
            * torch.sqrt((y ** 2).sum() + self.eps)
        )
        corr = (x * y).sum() / (denom + self.eps)
        return 1.0 - corr

    def _part_graph_spearman_surrogate_loss(
        self,
        part_text_feat,
        part_proj,
        part_valid_mask,
    ):
        """
        Preserve text-side PART-PART graph structure before/after projection.

        For each sample b:
          pre  graph = cosine(part_text_feat_i, part_text_feat_j)
          post graph = cosine(part_proj_i,      part_proj_j)

        Then take the upper triangle (excluding diagonal) and maximize the
        correlation between the two vectors.
        """
        pre_part = self._safe_normalize(part_text_feat.float(), dim=-1)
        post_part = self._safe_normalize(part_proj.float(), dim=-1)

        losses = []
        B, _, _ = pre_part.shape
        for b in range(B):
            valid_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)
            Kb = int(valid_idx.numel())
            if Kb < 2:
                continue

            pre_b = pre_part[b, valid_idx]
            post_b = post_part[b, valid_idx]

            pre_sim = pre_b @ pre_b.T
            post_sim = post_b @ post_b.T

            tri = torch.triu_indices(Kb, Kb, offset=1, device=pre_sim.device)
            pre_vec = pre_sim[tri[0], tri[1]]
            post_vec = post_sim[tri[0], tri[1]]

            if pre_vec.numel() < 2:
                continue

            losses.append(self._corr_loss(pre_vec, post_vec))

        if len(losses) == 0:
            return part_proj.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _part_obj_relation_spearman_surrogate_loss(
        self,
        obj_text_feat,
        part_text_feat,
        obj_proj,
        part_proj,
        part_valid_mask,
    ):
        """
        Preserve text-side PART-OBJ relation before/after projection.

        For each sample b:
          pre_scores[k]  = cosine(part_text_feat_k, obj_text_feat)
          post_scores[k] = cosine(part_proj_k,      obj_proj)

        Then maximize the correlation between these two score vectors.
        """
        pre_obj = self._safe_normalize(obj_text_feat.float(), dim=-1)     # [B, D_t]
        pre_part = self._safe_normalize(part_text_feat.float(), dim=-1)   # [B, K, D_t]
        post_obj = self._safe_normalize(obj_proj.float(), dim=-1)         # [B, D_v]
        post_part = self._safe_normalize(part_proj.float(), dim=-1)       # [B, K, D_v]

        pre_scores = torch.einsum("bkd,bd->bk", pre_part, pre_obj)
        post_scores = torch.einsum("bkd,bd->bk", post_part, post_obj)

        losses = []
        B, _ = pre_scores.shape
        for b in range(B):
            valid_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)
            Kb = int(valid_idx.numel())
            if Kb < 2:
                continue

            pre_vec = pre_scores[b, valid_idx]
            post_vec = post_scores[b, valid_idx]

            if pre_vec.numel() < 2:
                continue

            losses.append(self._corr_loss(pre_vec, post_vec))

        if len(losses) == 0:
            return part_proj.new_tensor(0.0)
        return torch.stack(losses).mean()
    
    def _soft_part_overlap_loss(self, abs_logits, obj_mask_patch, part_valid_mask):
        """
        Penalize different parts attending to the same object patches.

        abs_logits:       [B, K, N]
        obj_mask_patch:   [B, N]
        part_valid_mask:  [B, K]
        """
        if not part_valid_mask.any():
            return abs_logits.new_tensor(0.0)

        logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        # Per-part soft patch distribution inside object mask.
        attn = F.softmax(logits, dim=-1)  # [B, K, N]
        attn = attn * part_valid_mask[:, :, None].float()

        # Pairwise overlap between part attention maps.
        # overlap[b, k, l] is high if part k and part l attend to same patches.
        overlap = torch.einsum("bkn,bln->bkl", attn, attn)  # [B, K, K]

        B, K, _ = overlap.shape
        valid_pair = part_valid_mask[:, :, None] & part_valid_mask[:, None, :]

        eye = torch.eye(K, device=overlap.device, dtype=torch.bool)[None, :, :]
        valid_pair = valid_pair & ~eye

        if not valid_pair.any():
            return abs_logits.new_tensor(0.0)

        return overlap[valid_pair].mean()

    def _combined_structure_spearman_surrogate_loss(
        self,
        obj_text_feat,
        part_text_feat,
        obj_proj,
        part_proj,
        part_valid_mask,
    ):
        graph_loss = self._part_graph_spearman_surrogate_loss(
            part_text_feat=part_text_feat,
            part_proj=part_proj,
            part_valid_mask=part_valid_mask,
        )
        objrel_loss = self._part_obj_relation_spearman_surrogate_loss(
            obj_text_feat=obj_text_feat,
            part_text_feat=part_text_feat,
            obj_proj=obj_proj,
            part_proj=part_proj,
            part_valid_mask=part_valid_mask,
        )

        # Equal-weight combination to keep overall spear scale stable.
        return 0.5 * (graph_loss + objrel_loss)

    def _masked_mean(self, x, mask):
        if not mask.any():
            return x.new_tensor(0.0)
        x = x * mask.float()
        return x.sum() / (mask.float().sum() + self.eps)
