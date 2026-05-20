import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.loss_gw as lsgw
from src.loss_gw import (
    Stage3GWLoss,
    safe_normalize,
    pairwise_cosine_distance,
    hard_bijective_gw_match,
    hard_gw_struct_objective,
)


class ToyProjector(nn.Module):
    def __init__(self, dim: int, init: str = "identity"):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        if init == "identity":
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(dim))
        elif init == "random":
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init: {init}")

    def project_clip_txt(self, x):
        return self.proj(x)


def make_rotation(dim, device):
    A = torch.randn(dim, dim, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


@torch.no_grad()
def evaluate(model, V, T_fake, perm, args):
    z = safe_normalize(model.project_clip_txt(T_fake), dim=-1)
    C_z = pairwise_cosine_distance(z)
    C_t = pairwise_cosine_distance(T_fake)
    C_v = pairwise_cosine_distance(V)

    pred_perm, pred_obj = hard_bijective_gw_match(
        C_z,
        C_v,
        num_iters=args.gw_max_iter,
        num_restarts=args.num_restarts,
        include_identity=True,
    )

    gw_struct = hard_gw_struct_objective(C_z, C_v, pred_perm).item()
    struct_mse = F.mse_loss(C_z, C_t).item()
    perm_acc = (pred_perm == perm).float().mean().item()

    # These are NOT optimized by the final two-loss objective. They are printed
    # only to remind you whether feature-level retrieval changed accidentally.
    sim = z @ V.T
    retr = sim.argmax(dim=1)
    retr_acc = (retr == perm).float().mean().item()

    return {
        "gw_struct": gw_struct,
        "struct_mse": struct_mse,
        "perm_acc": perm_acc,
        "retr_acc_not_optimized": retr_acc,
        "pred_perm": pred_perm.detach().cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda_gw", type=float, default=1.0)
    parser.add_argument("--lambda_struct", type=float, default=0.1)
    parser.add_argument("--gw_max_iter", type=int, default=50)
    parser.add_argument("--num_restarts", type=int, default=500)
    parser.add_argument("--projector_init", choices=["identity", "random"], default="identity")
    parser.add_argument("--rotation", action="store_true", default=True)
    parser.add_argument("--no_rotation", action="store_false", dest="rotation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("using loss file:", lsgw.__file__)
    print("has hard_bijective_gw_match:", hasattr(lsgw, "hard_bijective_gw_match"))

    K, D = args.k, args.dim
    V = safe_normalize(torch.randn(K, D, device=device), dim=-1)
    perm = torch.randperm(K, device=device)

    if args.rotation:
        R = make_rotation(D, device)
        T_fake = safe_normalize(V[perm] @ R, dim=-1)
    else:
        T_fake = V[perm].clone()

    model = ToyProjector(D, init=args.projector_init).to(device)

    class_blocks = [
        {
            "category_id": 0,
            "class_name": "toy",
            "part_ids": torch.arange(K, device=device),
            "part_text": T_fake.detach(),
            "part_names": [f"p{i}" for i in range(K)],
        }
    ]

    criterion = Stage3GWLoss(
        sim_model=model,
        visual_proto=V.detach(),
        class_blocks=class_blocks,
        lambda_obj=0.0,
        lambda_gw=args.lambda_gw,
        lambda_struct=args.lambda_struct,
        gw_max_iter=args.gw_max_iter,
        sinkhorn_iter=args.num_restarts,  # in the new file this means hard-GW restarts
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("perm:", perm.detach().cpu().tolist())
    print(
        f"settings: k={K} dim={D} steps={args.steps} lr={args.lr} "
        f"lambda_gw={args.lambda_gw} lambda_struct={args.lambda_struct} "
        f"gw_max_iter={args.gw_max_iter} num_restarts={args.num_restarts} "
        f"projector_init={args.projector_init} rotation={args.rotation}"
    )

    for step in range(args.steps + 1):
        losses = criterion(batch=None, do_structure_audit=False)
        loss = losses["total"]

        if step % 50 == 0 or step == args.steps:
            m = evaluate(model, V, T_fake, perm, args)
            print(
                f"step={step:04d} "
                f"total={loss.item():.8f} "
                f"gw={losses['gw'].item():.8f} "
                f"struct={losses['struct'].item():.8f} "
                f"hard_perm_acc={m['perm_acc']:.3f} "
                f"gw_struct={m['gw_struct']:.8e} "
                f"struct_mse={m['struct_mse']:.8e} "
                f"retr_acc_not_optimized={m['retr_acc_not_optimized']:.3f} "
                f"pred={m['pred_perm']}"
            )

        if step == args.steps:
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


if __name__ == "__main__":
    main()
