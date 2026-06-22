#!/usr/bin/env python3
import argparse, csv, itertools
from pathlib import Path

import torch

from tools.train_gensq_pointnet_local_scale_exp import set_seed, fibonacci_sphere, chamfer
from tools.train_gensq_weighted_pca_fps_clean_lr import (
    WeightedPCANet,
    random_rotations,
    batched_fps,
    gather_points,
)

PERMS = torch.tensor(list(itertools.permutations([0, 1, 2])), dtype=torch.long)


SHAPES = [
    ("sphere",                    [0.25,0.25,0.25], [2.0,2.0,2.0]),
    ("ellipsoid",                 [0.12,0.22,0.34], [2.0,2.0,2.0]),
    ("mixed_smooth_boxy",         [0.10,0.20,0.34], [1.2,4.0,8.0]),
    ("boxy_elongated",            [0.10,0.18,0.34], [8.0,8.0,8.0]),
    ("diamond_anisotropic",       [0.09,0.25,0.34], [1.0,1.0,1.0]),
    ("one_sharp_two_boxy",        [0.09,0.25,0.34], [1.0,8.0,8.0]),
    ("flat_pancake_boxy",         [0.30,0.32,0.09], [2.0,6.0,6.0]),
    ("very_anisotropic_cigar",    [0.08,0.12,0.35], [4.0,4.0,8.0]),
    ("near_equal_scales",         [0.21,0.22,0.23], [1.2,5.0,8.0]),
    ("two_equal_scales",          [0.20,0.20,0.34], [2.0,6.0,6.0]),
    ("boxy_rounded_middle",       [0.14,0.28,0.34], [8.0,1.2,8.0]),
    ("thin_equal_large_axes",     [0.08,0.34,0.34], [8.0,2.0,2.0]),
    ("monotonic_scale_exp",       [0.16,0.24,0.32], [1.0,4.5,8.0]),
    ("small_boxy_big_round",      [0.09,0.21,0.34], [8.0,2.0,1.2]),
    ("middle_boxy",               [0.11,0.26,0.33], [2.0,8.0,2.0]),
    ("moderate_all_boxy",         [0.13,0.23,0.31], [5.0,6.0,7.0]),
]


def logit_from_range(x, lo, hi):
    y = (x - lo) / (hi - lo)
    y = y.clamp(1e-4, 1 - 1e-4)
    return torch.log(y / (1 - y))


def value_from_logit(z, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(z)


def sample_gensq_newton(scale, exp, dirs, iters=32):
    if dirs.dim() == 2:
        dirs = dirs[None].expand(scale.shape[0], dirs.shape[0], 3)

    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dabs = dirs.abs().clamp_min(1e-12)

    log_s = scale[:, None, :].clamp_min(1e-12).log()
    e = exp[:, None, :]

    # robust log-radius initialization
    u = scale.mean(dim=1).clamp_min(1e-6).log()[:, None].expand(scale.shape[0], dirs.shape[1]).clone()

    lo = scale.min(dim=1).values.clamp_min(1e-6).log()[:, None] - 4.0
    hi = scale.max(dim=1).values.clamp_min(1e-6).log()[:, None] + 4.0

    log_d = dabs.log()

    for _ in range(iters):
        terms = torch.exp(e * (u[..., None] + log_d - log_s)).clamp_max(1e12)
        f = terms.sum(dim=-1) - 1.0
        df = (e * terms).sum(dim=-1).clamp_min(1e-8)
        u = (u - f / df).clamp(lo, hi)

    return torch.exp(u)[..., None] * dirs


def best_perm(frame, gt_R):
    device = frame.device
    perms = PERMS.to(device)
    dots = torch.einsum("bik,bil->bkl", frame, gt_R).abs().clamp(0, 1)
    scores = []
    for p in perms:
        scores.append(dots[:, torch.arange(3, device=device), p].pow(2).mean(dim=1))
    best = torch.stack(scores, dim=1).argmax(dim=1)
    return perms[best]


@torch.no_grad()
def make_inputs(scales, exps, args, device):
    m = scales.shape[0]
    base_dirs = fibonacci_sphere(args.candidate_n, device)

    dir_R = random_rotations(m, device)
    dirs = base_dirs[None].expand(m, args.candidate_n, 3) @ dir_R.transpose(1, 2)
    candidates = sample_gensq_newton(scales, exps, dirs)

    idx = batched_fps(candidates, args.points)
    local = gather_points(candidates, idx)

    gt_R = random_rotations(m, device)
    world = local @ gt_R.transpose(1, 2)
    return world, gt_R


def observed_scale_init(canon, scale_min, scale_max):
    q_hi = torch.quantile(canon, 0.99, dim=1)
    q_lo = torch.quantile(canon, 0.01, dim=1)
    s = 0.5 * (q_hi - q_lo).abs()
    return s.clamp(scale_min, scale_max)


def make_starts(gt_scale, gt_exp, obs_scale, args):
    starts = []

    starts.append(("gt_exact", gt_scale, gt_exp))
    starts.append(("gt_jit_low",  (gt_scale * 0.97).clamp(args.scale_min, args.scale_max), (gt_exp - 0.5).clamp(args.exp_min, args.exp_max)))
    starts.append(("gt_jit_high", (gt_scale * 1.03).clamp(args.scale_min, args.scale_max), (gt_exp + 0.5).clamp(args.exp_min, args.exp_max)))
    starts.append(("gt_jit_mix",  (gt_scale * torch.tensor([1.08,0.92,1.02], device=gt_scale.device)).clamp(args.scale_min, args.scale_max),
                   (gt_exp + torch.tensor([0.7,-0.4,0.2], device=gt_exp.device)).clamp(args.exp_min, args.exp_max)))

    starts.append(("obs_mid2",  obs_scale, torch.full_like(gt_exp, 2.0)))
    starts.append(("obs_boxy7", obs_scale, torch.full_like(gt_exp, 7.0)))
    starts.append(("obs_round12", obs_scale, torch.full_like(gt_exp, 1.2)))

    rand_scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand_like(gt_scale)
    rand_exp = args.exp_min + (args.exp_max - args.exp_min) * torch.rand_like(gt_exp)
    starts.append(("random", rand_scale, rand_exp))

    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wpca-ckpt", required=True)

    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--candidate-n", type=int, default=2048)
    ap.add_argument("--surface-n", type=int, default=2048)
    ap.add_argument("--eval-surface-n", type=int, default=4096)

    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=1.0)
    ap.add_argument("--exp-max", type=float, default=8.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    wpca = WeightedPCANet().to(device)
    ckpt = torch.load(args.wpca_ckpt, map_location=device)
    wpca.load_state_dict(ckpt["model"])
    wpca.eval()

    shape_names = [x[0] for x in SHAPES]
    gt_scale_orig = torch.tensor([x[1] for x in SHAPES], dtype=torch.float32, device=device)
    gt_exp_orig = torch.tensor([x[2] for x in SHAPES], dtype=torch.float32, device=device)

    world, gt_R = make_inputs(gt_scale_orig, gt_exp_orig, args, device)

    with torch.no_grad():
        frame, _, _ = wpca(world)
        canon = world @ frame

        perm = best_perm(frame, gt_R)
        gt_scale = gt_scale_orig.gather(1, perm)
        gt_exp = gt_exp_orig.gather(1, perm)
        obs_scale = observed_scale_init(canon, args.scale_min, args.scale_max)

    all_inputs = []
    all_gt_scale = []
    all_gt_exp = []
    all_shape_idx = []
    all_start_name = []
    init_scale_list = []
    init_exp_list = []

    for i in range(len(SHAPES)):
        starts = make_starts(gt_scale[i], gt_exp[i], obs_scale[i], args)
        for start_name, s0, e0 in starts:
            all_inputs.append(canon[i])
            all_gt_scale.append(gt_scale[i])
            all_gt_exp.append(gt_exp[i])
            all_shape_idx.append(i)
            all_start_name.append(start_name)
            init_scale_list.append(s0)
            init_exp_list.append(e0)

    x = torch.stack(all_inputs, dim=0)
    target_scale = torch.stack(all_gt_scale, dim=0)
    target_exp = torch.stack(all_gt_exp, dim=0)
    init_scale = torch.stack(init_scale_list, dim=0)
    init_exp = torch.stack(init_exp_list, dim=0)

    raw_scale = torch.nn.Parameter(logit_from_range(init_scale, args.scale_min, args.scale_max))
    raw_exp = torch.nn.Parameter(logit_from_range(init_exp, args.exp_min, args.exp_max))

    opt = torch.optim.Adam([raw_scale, raw_exp], lr=args.lr)

    dirs = fibonacci_sphere(args.surface_n, device)

    best_loss = torch.full((x.shape[0],), float("inf"), device=device)
    best_scale = init_scale.clone()
    best_exp = init_exp.clone()

    for step in range(1, args.steps + 1):
        pred_scale = value_from_logit(raw_scale, args.scale_min, args.scale_max)
        pred_exp = value_from_logit(raw_exp, args.exp_min, args.exp_max)

        pred_pts = sample_gensq_newton(pred_scale, pred_exp, dirs)
        cd_vec = chamfer(pred_pts, x)
        loss = cd_vec.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            improved = cd_vec < best_loss
            best_loss[improved] = cd_vec[improved]
            best_scale[improved] = pred_scale[improved]
            best_exp[improved] = pred_exp[improved]

        if step % 50 == 0 or step == 1:
            print(f"step={step:04d} loss={float(loss):.8f} best_mean={float(best_loss.mean()):.8f}", flush=True)

    with torch.no_grad():
        eval_dirs = fibonacci_sphere(args.eval_surface_n, device)
        gt_surf = sample_gensq_newton(target_scale, target_exp, eval_dirs)
        best_surf = sample_gensq_newton(best_scale, best_exp, eval_dirs)

        cd_input_pred = chamfer(best_surf, x)
        cd_gt_pred = chamfer(gt_surf, best_surf)

        scale_abs = (best_scale - target_scale).abs()
        exp_abs = (best_exp - target_exp).abs()

        rows = []
        for j in range(x.shape[0]):
            rows.append({
                "shape": shape_names[all_shape_idx[j]],
                "shape_idx": all_shape_idx[j],
                "start": all_start_name[j],
                "gt_scale": [round(float(v), 6) for v in target_scale[j].cpu()],
                "gt_exp": [round(float(v), 6) for v in target_exp[j].cpu()],
                "init_scale": [round(float(v), 6) for v in init_scale[j].cpu()],
                "init_exp": [round(float(v), 6) for v in init_exp[j].cpu()],
                "pred_scale": [round(float(v), 6) for v in best_scale[j].cpu()],
                "pred_exp": [round(float(v), 6) for v in best_exp[j].cpu()],
                "scale_l1": float(scale_abs[j].mean()),
                "exp_l1": float(exp_abs[j].mean()),
                "cd_input_pred": float(cd_input_pred[j]),
                "cd_gt_pred_surface": float(cd_gt_pred[j]),
            })

    csv_path = out / "all_starts.csv"
    with open(csv_path, "w", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # concise summaries
    lines = []
    for i, name in enumerate(shape_names):
        rr = [r for r in rows if r["shape_idx"] == i]
        best = min(rr, key=lambda r: r["cd_input_pred"])
        gtstart = [r for r in rr if r["start"] == "gt_exact"][0]
        boxystart = [r for r in rr if r["start"] == "obs_boxy7"][0]

        lines.append(f"\n=== {name} ===")
        lines.append(f"GT scale={best['gt_scale']} exp={best['gt_exp']}")
        lines.append(
            f"best_cd start={best['start']:11s} "
            f"pred_exp={best['pred_exp']} exp_l1={best['exp_l1']:.4f} "
            f"scale_l1={best['scale_l1']:.5f} "
            f"cd_in={best['cd_input_pred']:.8f} cd_gt={best['cd_gt_pred_surface']:.8f}"
        )
        lines.append(
            f"gt_exact             "
            f"pred_exp={gtstart['pred_exp']} exp_l1={gtstart['exp_l1']:.4f} "
            f"scale_l1={gtstart['scale_l1']:.5f} "
            f"cd_in={gtstart['cd_input_pred']:.8f} cd_gt={gtstart['cd_gt_pred_surface']:.8f}"
        )
        lines.append(
            f"obs_boxy7            "
            f"pred_exp={boxystart['pred_exp']} exp_l1={boxystart['exp_l1']:.4f} "
            f"scale_l1={boxystart['scale_l1']:.5f} "
            f"cd_in={boxystart['cd_input_pred']:.8f} cd_gt={boxystart['cd_gt_pred_surface']:.8f}"
        )

    summary = "\n".join(lines) + "\n"
    (out / "summary_short.txt").write_text(summary)
    print(summary)
    print(f"Wrote {csv_path}")
    print(f"Wrote {out / 'summary_short.txt'}")


if __name__ == "__main__":
    main()
