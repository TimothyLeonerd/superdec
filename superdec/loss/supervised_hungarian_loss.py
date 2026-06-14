import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from superdec.loss.loss import sampling_from_parametric_space_to_equivalent_points
from superdec.loss.sampler import EqualDistanceSamplerSQ

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    linear_sum_assignment = None
    _SCIPY_IMPORT_ERROR = exc


class NaiveGridSamplerSQ:
    """Uniform parameter-grid sampler matching SQ-Zero naive sampling.

    eta/theta:
      -pi/2 + pi/(2*n_theta), ..., pi/2 - pi/(2*n_theta)

    omega/phi:
      -pi + pi/n_phi, ..., pi - pi/n_phi

    This is not equal-distance surface sampling. It intentionally uses a
    parameter-space grid, like the SQ-Zero naive generator.
    """

    def __init__(self, n_theta=12, n_phi=12):
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        if self.n_theta <= 0 or self.n_phi <= 0:
            raise ValueError(
                f"NaiveGridSamplerSQ needs positive n_theta/n_phi, got "
                f"{self.n_theta}/{self.n_phi}"
            )

        self._n_samples = self.n_theta * self.n_phi

        d_eta = np.pi / self.n_theta
        d_omega = 2.0 * np.pi / self.n_phi

        eta0 = -np.pi / 2.0 + np.pi / (2.0 * self.n_theta)
        omega0 = -np.pi + np.pi / self.n_phi

        eta = eta0 + d_eta * np.arange(self.n_theta, dtype=np.float32)
        omega = omega0 + d_omega * np.arange(self.n_phi, dtype=np.float32)

        # Flatten in theta-outer, omega-inner order, matching sample_SQ_naive.
        eta_grid = np.repeat(eta[:, None], self.n_phi, axis=1)
        omega_grid = np.repeat(omega[None, :], self.n_theta, axis=0)

        self._etas = eta_grid.reshape(-1).astype(np.float32)
        self._omegas = omega_grid.reshape(-1).astype(np.float32)

    @property
    def n_samples(self):
        return self._n_samples

    def sample(self, **kwargs):
        return self._etas.copy(), self._omegas.copy()

    def sample_on_batch(self, shapes, epsilons):
        B = int(shapes.shape[0])
        M = int(shapes.shape[1])
        etas = np.broadcast_to(self._etas.reshape(1, 1, -1), (B, M, self._n_samples))
        omegas = np.broadcast_to(self._omegas.reshape(1, 1, -1), (B, M, self._n_samples))
        return etas.copy(), omegas.copy()


class SupervisedHungarianLoss(nn.Module):
    """Supervised DETR-style loss for SQ-Zero / part-labeled point clouds.

    Terms:
      - supervised existence BCE
      - supervised point-to-slot assignment NLL
      - optional geometry loss:
          surface_target="gt_surface":
              predicted SQ surface <-> sampled GT SQ surface
          surface_target="part_points":
              actual labeled part points -> predicted SQ surface
              plus beta * predicted SQ surface -> actual labeled part points

    Expected model outputs:
      out_dict["assign_matrix"]: [B, N, P]
      out_dict["exist"]:         [B, P, 1]
      out_dict["scale"]:         [B, P, 3]
      out_dict["shape"]:         [B, P, 2]
      out_dict["rotate"]:        [B, P, 3, 3]
      out_dict["trans"]:         [B, P, 3]

    Expected batch fields:
      batch["points"]:           [B, N, 3]
      batch["labels"]:           [B, N]
      batch["K"]:                [B]

    Additional fields for surface_target="gt_surface":
      batch["gt_scale"]:         [B, Kmax, 3]
      batch["gt_shape"]:         [B, Kmax, 2]
      batch["gt_rotate"]:        [B, Kmax, 3, 3]
      batch["gt_trans"]:         [B, Kmax, 3]
    """

    requires_batch = True

    def __init__(self, cfg):
        super().__init__()

        if linear_sum_assignment is None:
            raise ImportError(
                "SupervisedHungarianLoss requires scipy.optimize.linear_sum_assignment. "
                f"Original import error: {_SCIPY_IMPORT_ERROR}"
            )

        # Differentiable loss weights.
        self.w_exist = float(getattr(cfg, "w_sup_exist", 1.0))
        self.w_assign = float(getattr(cfg, "w_sup_assign", 1.0))
        self.w_surface = float(getattr(cfg, "w_sup_surface", 0.0))

        # Synthetic-only helper loss for SQ-Zero:
        # sign-invariant supervision of SQ local z-axis after Hungarian matching.
        self.w_z_axis = float(getattr(cfg, "w_sup_z_axis", 0.0))

        # Synthetic-only helper loss for SQ-Zero:
        # directly supervise eps_1, eps_2 after Hungarian matching.
        # Keep at 0.0 for real datasets without GT SQ parameters.
        self.w_shape_param = float(getattr(cfg, "w_sup_shape_param", 0.0))

        # Signed part-normal auxiliary loss:
        #   L = mean(1 - dot(input_normal, nearest_pred_sq_normal))
        # Requires sqzero_lmdb.normal_mode=sidecar.
        self.w_normal = float(getattr(cfg, "w_sup_normal", 0.0))
        self.normal_unsigned = bool(getattr(cfg, "normal_unsigned", False))
        self.normal_max_points_per_part = int(
            getattr(cfg, "normal_max_points_per_part", 512)
        )
        self.normal_eps = float(getattr(cfg, "normal_eps", 1e-8))

        # Location-independent normal-direction Chamfer.
        # Compares only normal directions per matched primitive, not xyz positions.
        self.w_normal_dir = float(getattr(cfg, "w_sup_normal_dir", 0.0))
        self.normal_dir_beta = float(getattr(cfg, "normal_dir_beta", 0.25))

        # If > 0, downweight normal correspondences whose nearest predicted
        # surface point is far away:
        #   gate = exp(-nn_dist^2 / tau^2)
        # The gate is detached by default so the model cannot game the loss by
        # changing distances only to change weights.

        # Strict staged SQ geometry loss, v1.  This is optional and is meant
        # to be enabled together with superdec.decoder.staged_params=true.
        # The three stage losses reuse the final Hungarian matching but build
        # surfaces with restricted parameter ownership:
        #   stage 1: translation only, fixed isotropic sphere
        #   stage 2: scale + rotation only, detached stage-1 translation
        #   stage 3: shape only, detached stage-1 translation and stage-2 scale/rotation
        self.use_staged_surface = bool(getattr(cfg, "use_staged_surface", False))
        self.stage1_fixed_radius = float(getattr(cfg, "stage1_fixed_radius", 0.575))
        self.w_stage1_surface = float(getattr(cfg, "w_stage1_surface", 0.0))
        self.w_stage2_surface = float(getattr(cfg, "w_stage2_surface", 0.0))
        self.w_stage3_surface = float(getattr(cfg, "w_stage3_surface", 0.0))

        # Matching-cost weights. These affect only the discrete Hungarian step.
        self.match_w_assign = float(getattr(cfg, "match_w_assign", 1.0))
        self.match_w_exist = float(getattr(cfg, "match_w_exist", 0.0))

        self.eps = float(getattr(cfg, "eps", 1e-8))

        self.w_shape_oracle_surface = float(
            getattr(cfg, "w_sup_shape_oracle_surface", 0.0)
        )

        # Synthetic diagnostic:
        # choose eta/omega samples from high-curvature regions of the GT SQ,
        # then compare predicted-vs-GT points at exactly those same eta/omega.
        self.w_highcurv_pointwise_surface = float(
            getattr(cfg, "w_sup_highcurv_pointwise_surface", 0.0)
        )
        self.w_highcurv_chamfer_surface = float(
            getattr(cfg, "w_sup_highcurv_chamfer_surface", 0.0)
        )
        self.highcurv_n_samples = int(getattr(cfg, "highcurv_n_samples", 128))
        self.highcurv_eta_bins = int(getattr(cfg, "highcurv_eta_bins", 32))
        self.highcurv_omega_bins = int(getattr(cfg, "highcurv_omega_bins", 64))
        self.highcurv_alpha = float(getattr(cfg, "highcurv_alpha", 2.0))
        self.highcurv_uniform_mix = float(getattr(cfg, "highcurv_uniform_mix", 0.0))
        self.highcurv_jitter = float(getattr(cfg, "highcurv_jitter", 1.0))
        self.highcurv_probe_delta = float(getattr(cfg, "highcurv_probe_delta", 0.05))

        # Geometry target.
        self.surface_target = str(getattr(cfg, "surface_target", "gt_surface"))
        valid_targets = {"gt_surface", "part_points"}
        if self.surface_target not in valid_targets:
            raise ValueError(
                f"Unknown loss.surface_target={self.surface_target!r}. "
                f"Expected one of {sorted(valid_targets)}."
            )

        if self.use_staged_surface and self.surface_target != "gt_surface":
            raise ValueError(
                "loss.use_staged_surface=true currently supports only "
                "loss.surface_target=gt_surface for strict staged v1."
            )

        # For part-points Chamfer:
        # L = d(part -> pred) + beta * d(pred -> part)
        self.part_cd_beta = float(getattr(cfg, "part_cd_beta", 0.0))
        self.part_cd_max_points_per_part = int(
            getattr(cfg, "part_cd_max_points_per_part", 512)
        )

        # Surface sampler.
        self.surface_sampler_type = str(
            getattr(cfg, "surface_sampler_type", "equal_distance")
        )

        if self.surface_sampler_type == "equal_distance":
            self.surface_n_samples = int(getattr(cfg, "surface_n_samples", 128))
            self.surface_sampler = EqualDistanceSamplerSQ(
                n_samples=self.surface_n_samples,
                D_eta=float(getattr(cfg, "surface_D_eta", 0.05)),
                D_omega=float(getattr(cfg, "surface_D_omega", 0.05)),
            )
        elif self.surface_sampler_type == "naive":
            self.surface_naive_n_theta = int(
                getattr(cfg, "surface_naive_n_theta", 12)
            )
            self.surface_naive_n_phi = int(
                getattr(cfg, "surface_naive_n_phi", 12)
            )
            self.surface_sampler = NaiveGridSamplerSQ(
                n_theta=self.surface_naive_n_theta,
                n_phi=self.surface_naive_n_phi,
            )
            self.surface_n_samples = self.surface_sampler.n_samples
        else:
            raise ValueError(
                f"Unknown loss.surface_sampler_type={self.surface_sampler_type!r}. "
                "Expected 'equal_distance' or 'naive'."
            )

        # Debug logging.
        self.debug_match_every = int(getattr(cfg, "debug_match_every", 0))
        self.debug_match_max_samples = int(getattr(cfg, "debug_match_max_samples", 4))
        self._forward_calls = 0

    @staticmethod
    def _local_to_world(local_points, rotate, trans):
        """Transform local primitive surface points into normalized object frame.

        Args:
            local_points: [B, M, S, 3]
            rotate:       [B, M, 3, 3], local-to-world rotation
            trans:        [B, M, 3]

        Returns:
            world_points: [B, M, S, 3]
        """
        return (
            torch.einsum("bmij,bmsj->bmsi", rotate, local_points)
            + trans.unsqueeze(2)
        )

    @staticmethod
    def _chamfer_squared(x, y):
        """Symmetric squared Chamfer distance.

        Args:
            x: [Nx, 3]
            y: [Ny, 3]
        """
        if x.numel() == 0 or y.numel() == 0:
            return x.new_tensor(0.0)

        d2 = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2.0)[0] ** 2
        return d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean()

    @staticmethod
    def _directional_chamfer_squared(src, dst):
        """One directional squared Chamfer: src -> dst.

        Args:
            src: [Ns, 3]
            dst: [Nd, 3]

        Returns:
            mean_s min_d ||src_s - dst_d||^2
        """
        if src.numel() == 0 or dst.numel() == 0:
            return src.new_tensor(0.0)

        d2 = torch.cdist(src.unsqueeze(0), dst.unsqueeze(0), p=2.0)[0] ** 2
        return d2.min(dim=1).values.mean()

    def _subsample_part_points(self, points):
        """Deterministically subsample part points for cheaper part-point Chamfer."""
        max_points = self.part_cd_max_points_per_part

        if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
            return points

        idx = torch.linspace(
            0,
            points.shape[0] - 1,
            steps=max_points,
            device=points.device,
        ).long()
        return points[idx]

    def _subsample_part_points_and_normals(self, points, normals):
        """Deterministically subsample paired part points/normals."""
        max_points = self.normal_max_points_per_part

        if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
            return points, normals

        idx = torch.linspace(
            0,
            points.shape[0] - 1,
            steps=max_points,
            device=points.device,
        ).long()
        return points[idx], normals[idx]

    @torch.no_grad()
    def _hungarian_for_one(self, assign_b, exist_b, labels_b, K_b):
        """Compute GT primitive -> predicted slot matching for one sample.

        Returns:
            matched_slots: [K], matched_slots[k] = predicted slot p.
        """
        _, P = assign_b.shape
        K = int(K_b)

        if K < 1:
            raise ValueError("K must be >= 1 for supervised samples.")
        if K > P:
            raise ValueError(f"K={K} cannot exceed number of predicted slots P={P}.")

        cost = assign_b.new_zeros((P, K))  # cost[p, k]

        for k in range(K):
            mask = labels_b == k
            n_k = int(mask.sum().item())

            if n_k == 0:
                cost[:, k] = 1e6
                continue

            probs = assign_b[mask, :]  # [n_k, P]
            assign_cost = -torch.log(probs.clamp_min(self.eps)).mean(dim=0)  # [P]
            exist_cost = -torch.log(exist_b.clamp_min(self.eps))             # [P]

            cost[:, k] = (
                self.match_w_assign * assign_cost
                + self.match_w_exist * exist_cost
            )

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        matched_slots_np = np.empty((K,), dtype=np.int64)
        for p, k in zip(row_ind, col_ind):
            matched_slots_np[k] = p

        return torch.as_tensor(
            matched_slots_np,
            device=assign_b.device,
            dtype=torch.long,
        )

    def _sample_predicted_surfaces_and_normals(self, out_dict):
        """Sample predicted SQ surfaces and outward normals for all slots.

        Normals are transformed by rotation only. Translation must never be
        applied to normals.
        """
        pred_local, pred_norm_local = sampling_from_parametric_space_to_equivalent_points(
            out_dict["scale"],
            out_dict["shape"],
            self.surface_sampler,
        )

        pred_world = self._local_to_world(
            pred_local,
            out_dict["rotate"],
            out_dict["trans"],
        )

        pred_norm_local = F.normalize(pred_norm_local, dim=-1, eps=self.normal_eps)
        pred_norm_world = torch.einsum(
            "bpij,bpsj->bpsi",
            out_dict["rotate"],
            pred_norm_local,
        )
        pred_norm_world = F.normalize(pred_norm_world, dim=-1, eps=self.normal_eps)

        return pred_world, pred_norm_world

    def _sample_predicted_surfaces(self, out_dict):
        """Sample predicted SQ surfaces for all predicted slots."""
        pred_world, _ = self._sample_predicted_surfaces_and_normals(out_dict)
        return pred_world

    def _compute_gt_surface_loss(self, out_dict, batch, all_matched_slots):
        """Matched predicted-vs-GT sampled SQ surface Chamfer.

        This is the previous/current SQ-Zero synthetic geometry loss.
        """
        pred_world = self._sample_predicted_surfaces(out_dict)

        pred_scale = out_dict["scale"]

        gt_scale = batch["gt_scale"].to(pred_scale.device).float()
        gt_shape = batch["gt_shape"].to(pred_scale.device).float()
        gt_rotate = batch["gt_rotate"].to(pred_scale.device).float()
        gt_trans = batch["gt_trans"].to(pred_scale.device).float()
        K = batch["K"].to(pred_scale.device).long()

        B = pred_scale.shape[0]

        total = pred_scale.new_tensor(0.0)
        n_pairs = 0

        for b in range(B):
            K_b = int(K[b].item())
            matched_slots = all_matched_slots[b]

            gt_local_b, _ = sampling_from_parametric_space_to_equivalent_points(
                gt_scale[b:b + 1, :K_b, :],
                gt_shape[b:b + 1, :K_b, :],
                self.surface_sampler,
            )

            gt_world_b = self._local_to_world(
                gt_local_b,
                gt_rotate[b:b + 1, :K_b, :, :],
                gt_trans[b:b + 1, :K_b, :],
            )[0]  # [K_b, S, 3]

            for k in range(K_b):
                p = int(matched_slots[k].item())
                x_pred = pred_world[b, p]  # [S, 3]
                x_gt = gt_world_b[k]       # [S, 3]

                total = total + self._chamfer_squared(x_pred, x_gt)
                n_pairs += 1

        if n_pairs == 0:
            return pred_scale.new_tensor(0.0)

        return total / n_pairs

    def _compute_part_point_surface_loss(self, pc, batch, out_dict, all_matched_slots):
        """Matched predicted SQ surface to actual labeled part points.

        For each GT part k, with matched predicted slot p:

            L_k = d(part_points_k -> pred_surface_p)
                  + beta * d(pred_surface_p -> part_points_k)

        beta=0.0 is intentionally very asymmetric and means:
            every observed part point should be near the predicted SQ,
            but extra/unobserved predicted SQ surface is not penalized.

        This target is more compatible with real part-labeled datasets than
        sampled GT SQ surfaces.
        """
        pred_world = self._sample_predicted_surfaces(out_dict)

        labels = batch["labels"].to(pc.device).long()
        K = batch["K"].to(pc.device).long()

        B = pc.shape[0]

        total = pc.new_tensor(0.0)
        n_pairs = 0

        for b in range(B):
            K_b = int(K[b].item())
            matched_slots = all_matched_slots[b]

            for k in range(K_b):
                part_mask = labels[b] == k
                if int(part_mask.sum().item()) == 0:
                    continue

                part_points = pc[b, part_mask, :]  # [Nk, 3]
                part_points = self._subsample_part_points(part_points)

                p = int(matched_slots[k].item())
                pred_points = pred_world[b, p]     # [S, 3]

                part_to_pred = self._directional_chamfer_squared(
                    src=part_points,
                    dst=pred_points,
                )

                if self.part_cd_beta > 0.0:
                    pred_to_part = self._directional_chamfer_squared(
                        src=pred_points,
                        dst=part_points,
                    )
                    loss_k = part_to_pred + self.part_cd_beta * pred_to_part
                else:
                    loss_k = part_to_pred

                total = total + loss_k
                n_pairs += 1

        if n_pairs == 0:
            return pc.new_tensor(0.0)

        return total / n_pairs

    def _compute_part_normal_loss(self, pc, normals, batch, out_dict, all_matched_slots):
        """Signed matched part-normal loss against predicted SQ normals."""
        pred_world, pred_norm_world = self._sample_predicted_surfaces_and_normals(out_dict)

        labels = batch["labels"].to(pc.device).long()
        K = batch["K"].to(pc.device).long()
        normals = F.normalize(normals.to(pc.device).float(), dim=-1, eps=self.normal_eps)

        B = pc.shape[0]
        total = pc.new_tensor(0.0)
        n_pairs = 0

        dot_sum = 0.0
        abs_dot_sum = 0.0
        nn_dist_sum = 0.0

        for b in range(B):
            K_b = int(K[b].item())
            matched_slots = all_matched_slots[b]

            for k in range(K_b):
                part_mask = labels[b] == k
                if int(part_mask.sum().item()) == 0:
                    continue

                part_points = pc[b, part_mask, :]
                part_normals = normals[b, part_mask, :]
                part_points, part_normals = self._subsample_part_points_and_normals(
                    part_points,
                    part_normals,
                )

                p = int(matched_slots[k].item())
                pred_points = pred_world[b, p]
                pred_normals = pred_norm_world[b, p]

                d2 = torch.cdist(
                    part_points.unsqueeze(0),
                    pred_points.unsqueeze(0),
                    p=2.0,
                )[0] ** 2

                nn_dist2, nn_idx = d2.min(dim=1)
                nearest_pred_normals = pred_normals[nn_idx]

                dots = (part_normals * nearest_pred_normals).sum(dim=-1).clamp(-1.0, 1.0)

                if self.normal_unsigned:
                    loss_k = (1.0 - dots.abs()).mean()
                else:
                    loss_k = (1.0 - dots).mean()

                total = total + loss_k
                n_pairs += 1

                with torch.no_grad():
                    dot_sum += float(dots.mean().detach().cpu().item())
                    abs_dot_sum += float(dots.abs().mean().detach().cpu().item())
                    nn_dist_sum += float(torch.sqrt(nn_dist2.clamp_min(0.0)).mean().detach().cpu().item())

        if n_pairs == 0:
            zero = pc.new_tensor(0.0)
            return zero, {
                "sup_normal_dot_mean": 0.0,
                "sup_normal_absdot_mean": 0.0,
                "sup_normal_nn_dist_mean": 0.0,
            }

        return total / n_pairs, {
            "sup_normal_dot_mean": dot_sum / n_pairs,
            "sup_normal_absdot_mean": abs_dot_sum / n_pairs,
            "sup_normal_nn_dist_mean": nn_dist_sum / n_pairs,
        }

    def _compute_normal_direction_loss(self, normals, batch, out_dict, all_matched_slots):
        """Location-independent normal-direction Chamfer per matched primitive."""
        _, pred_norm_world = self._sample_predicted_surfaces_and_normals(out_dict)

        device = pred_norm_world.device
        labels = batch["labels"].to(device).long()
        K = batch["K"].to(device).long()
        normals = F.normalize(normals.to(device).float(), dim=-1, eps=self.normal_eps)

        B = normals.shape[0]
        total = normals.new_tensor(0.0)
        n_pairs = 0
        gt2pred_dot_sum = 0.0
        pred2gt_dot_sum = 0.0

        for b in range(B):
            K_b = int(K[b].item())
            matched_slots = all_matched_slots[b]

            for k in range(K_b):
                part_mask = labels[b] == k
                if int(part_mask.sum().item()) == 0:
                    continue

                part_normals = normals[b, part_mask, :]

                max_points = self.normal_max_points_per_part
                if max_points is not None and max_points > 0 and part_normals.shape[0] > max_points:
                    idx = torch.linspace(
                        0,
                        part_normals.shape[0] - 1,
                        steps=max_points,
                        device=part_normals.device,
                    ).long()
                    part_normals = part_normals[idx]

                p = int(matched_slots[k].item())
                pred_normals = pred_norm_world[b, p]

                sim = torch.matmul(part_normals, pred_normals.transpose(0, 1)).clamp(-1.0, 1.0)
                if self.normal_unsigned:
                    sim = sim.abs()

                gt2pred_best = sim.max(dim=1).values
                pred2gt_best = sim.max(dim=0).values

                loss_k = (1.0 - gt2pred_best).mean()
                if self.normal_dir_beta > 0.0:
                    loss_k = loss_k + self.normal_dir_beta * (1.0 - pred2gt_best).mean()

                total = total + loss_k
                n_pairs += 1

                with torch.no_grad():
                    gt2pred_dot_sum += float(gt2pred_best.mean().detach().cpu().item())
                    pred2gt_dot_sum += float(pred2gt_best.mean().detach().cpu().item())

        if n_pairs == 0:
            zero = normals.new_tensor(0.0)
            return zero, {
                "sup_normal_dir_gt2pred_dot_mean": 0.0,
                "sup_normal_dir_pred2gt_dot_mean": 0.0,
            }

        return total / n_pairs, {
            "sup_normal_dir_gt2pred_dot_mean": gt2pred_dot_sum / n_pairs,
            "sup_normal_dir_pred2gt_dot_mean": pred2gt_dot_sum / n_pairs,
        }

    def _sq_local_from_eta_omega(self, shape_params, epsilons, etas, omegas):
        """Evaluate SQ local surface points/normals at fixed eta/omega.

        Args:
            shape_params: [B, M, 3]
            epsilons:     [B, M, 2]
            etas:         [B, M, S]
            omegas:       [B, M, S]
        """
        def fexp(x, p):
            return torch.sign(x) * (torch.abs(x) ** p)

        a1 = shape_params[:, :, 0].unsqueeze(-1)
        a2 = shape_params[:, :, 1].unsqueeze(-1)
        a3 = shape_params[:, :, 2].unsqueeze(-1)
        e1 = epsilons[:, :, 0].unsqueeze(-1)
        e2 = epsilons[:, :, 1].unsqueeze(-1)

        x = a1 * fexp(torch.cos(etas), e1) * fexp(torch.cos(omegas), e2)
        y = a2 * fexp(torch.cos(etas), e1) * fexp(torch.sin(omegas), e2)
        z = a3 * fexp(torch.sin(etas), e1)

        # Match the numerical guard used by the original SuperDec sampler.
        tiny = x.new_tensor(1e-6)
        x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), tiny)
        y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), tiny)
        z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), tiny)

        nx = (torch.cos(etas) ** 2) * (torch.cos(omegas) ** 2) / x
        ny = (torch.cos(etas) ** 2) * (torch.sin(omegas) ** 2) / y
        nz = (torch.sin(etas) ** 2) / z

        points = torch.stack([x, y, z], dim=-1)
        normals = F.normalize(torch.stack([nx, ny, nz], dim=-1), dim=-1, eps=self.normal_eps)
        return points, normals

    @torch.no_grad()
    def _sample_gt_highcurv_eta_omega(self, gt_scale, gt_shape):
        """Sample fixed-count eta/omega locations biased to GT high-curvature regions.

        Curvature proxy = local normal variation on a dense parameter grid.
        Output is [B, M, S] eta/omega. Selection is non-differentiable on purpose.
        """
        B, M, _ = gt_scale.shape
        device = gt_scale.device
        dtype = gt_scale.dtype

        eta_bins = self.highcurv_eta_bins
        omega_bins = self.highcurv_omega_bins
        n_samples = self.highcurv_n_samples

        if eta_bins <= 1 or omega_bins <= 1 or n_samples <= 0:
            raise ValueError(
                "highcurv_eta_bins, highcurv_omega_bins, highcurv_n_samples must be positive"
            )

        d_eta = np.pi / eta_bins
        d_omega = 2.0 * np.pi / omega_bins

        eta_centers_1d = torch.linspace(
            -np.pi / 2.0 + d_eta / 2.0,
            np.pi / 2.0 - d_eta / 2.0,
            eta_bins,
            device=device,
            dtype=dtype,
        )
        omega_centers_1d = torch.linspace(
            -np.pi + d_omega / 2.0,
            np.pi - d_omega / 2.0,
            omega_bins,
            device=device,
            dtype=dtype,
        )

        eta_grid, omega_grid = torch.meshgrid(
            eta_centers_1d,
            omega_centers_1d,
            indexing="ij",
        )
        eta_flat = eta_grid.reshape(-1)
        omega_flat = omega_grid.reshape(-1)
        G = eta_flat.numel()

        etas = eta_flat.view(1, 1, G).expand(B, M, G)
        omegas = omega_flat.view(1, 1, G).expand(B, M, G)

        delta_eta = min(float(self.highcurv_probe_delta), 0.45 * d_eta)
        delta_omega = min(float(self.highcurv_probe_delta), 0.45 * d_omega)

        eta_min = -np.pi / 2.0 + 1e-4
        eta_max = np.pi / 2.0 - 1e-4

        eta_p = torch.clamp(etas + delta_eta, eta_min, eta_max)
        eta_m = torch.clamp(etas - delta_eta, eta_min, eta_max)
        omega_p = omegas + delta_omega
        omega_m = omegas - delta_omega

        _, n_eta_p = self._sq_local_from_eta_omega(gt_scale, gt_shape, eta_p, omegas)
        _, n_eta_m = self._sq_local_from_eta_omega(gt_scale, gt_shape, eta_m, omegas)
        _, n_omega_p = self._sq_local_from_eta_omega(gt_scale, gt_shape, etas, omega_p)
        _, n_omega_m = self._sq_local_from_eta_omega(gt_scale, gt_shape, etas, omega_m)

        curv = (
            torch.linalg.norm(n_eta_p - n_eta_m, dim=-1)
            + torch.linalg.norm(n_omega_p - n_omega_m, dim=-1)
        )

        weights = (curv.clamp_min(0.0) + 1e-8) ** self.highcurv_alpha
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        mix = float(self.highcurv_uniform_mix)
        if mix > 0.0:
            mix = max(0.0, min(1.0, mix))
            weights = (1.0 - mix) * weights + mix * (1.0 / G)

        idx = torch.multinomial(
            weights.reshape(B * M, G),
            num_samples=n_samples,
            replacement=True,
        )

        eta_sel = eta_flat[idx].view(B, M, n_samples)
        omega_sel = omega_flat[idx].view(B, M, n_samples)

        jitter = float(self.highcurv_jitter)
        if jitter > 0.0:
            eta_sel = eta_sel + (torch.rand_like(eta_sel) - 0.5) * d_eta * jitter
            omega_sel = omega_sel + (torch.rand_like(omega_sel) - 0.5) * d_omega * jitter
            eta_sel = torch.clamp(eta_sel, eta_min, eta_max)

        return eta_sel, omega_sel

    def _compute_highcurv_chamfer_surface_loss(self, out_dict, batch, all_matched_slots):
        """GT-conditioned high-curvature Chamfer surface loss.

        Uses GT SQ only to choose high-curvature eta/omega sample locations.
        Then evaluates GT and predicted SQs at those eta/omega locations, but
        compares the resulting point sets with Chamfer instead of pointwise L2.
        """
        required = ["gt_scale", "gt_shape", "gt_rotate", "gt_trans"]
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "loss.w_sup_highcurv_chamfer_surface > 0 requires GT SQ sidecars; "
                f"missing batch keys: {missing}"
            )

        pred_scale = out_dict["scale"]
        pred_shape = out_dict["shape"]
        pred_rotate = out_dict["rotate"]
        pred_trans = out_dict["trans"]

        device = pred_scale.device
        gt_scale = batch["gt_scale"].to(device).float()
        gt_shape = batch["gt_shape"].to(device).float()
        gt_rotate = batch["gt_rotate"].to(device).float()
        gt_trans = batch["gt_trans"].to(device).float()
        K = batch["K"].to(device).long()

        B = pred_scale.shape[0]
        total = pred_scale.new_tensor(0.0)
        n_pairs = 0

        for b in range(B):
            K_b = int(K[b].item())
            if K_b <= 0:
                continue

            matched_slots = all_matched_slots[b][:K_b]

            gt_scale_b = gt_scale[b:b + 1, :K_b, :]
            gt_shape_b = gt_shape[b:b + 1, :K_b, :]
            gt_rotate_b = gt_rotate[b:b + 1, :K_b, :, :]
            gt_trans_b = gt_trans[b:b + 1, :K_b, :]

            pred_scale_b = pred_scale[b:b + 1, matched_slots, :]
            pred_shape_b = pred_shape[b:b + 1, matched_slots, :]
            pred_rotate_b = pred_rotate[b:b + 1, matched_slots, :, :]
            pred_trans_b = pred_trans[b:b + 1, matched_slots, :]

            etas, omegas = self._sample_gt_highcurv_eta_omega(gt_scale_b, gt_shape_b)

            pred_local_b, _ = self._sq_local_from_eta_omega(
                pred_scale_b,
                pred_shape_b,
                etas,
                omegas,
            )
            gt_local_b, _ = self._sq_local_from_eta_omega(
                gt_scale_b,
                gt_shape_b,
                etas,
                omegas,
            )

            pred_world_b = self._local_to_world(
                pred_local_b,
                pred_rotate_b,
                pred_trans_b,
            )[0]
            gt_world_b = self._local_to_world(
                gt_local_b,
                gt_rotate_b,
                gt_trans_b,
            )[0]

            for k in range(K_b):
                total = total + self._chamfer_squared(pred_world_b[k], gt_world_b[k])
                n_pairs += 1

        if n_pairs == 0:
            return pred_scale.new_tensor(0.0)

        return total / n_pairs

    def _compute_highcurv_pointwise_surface_loss(self, out_dict, batch, all_matched_slots):
        """GT-conditioned high-curvature pointwise surface loss.

        For each matched pair (GT primitive k, predicted slot p):
          1. pick eta/omega from high-curvature regions of the GT primitive;
          2. evaluate GT and prediction at the same eta/omega;
          3. use pointwise squared distance.

        This gives explicit parameter-space correspondence.
        """
        required = ["gt_scale", "gt_shape", "gt_rotate", "gt_trans"]
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "loss.w_sup_highcurv_pointwise_surface > 0 requires GT SQ sidecars; "
                f"missing batch keys: {missing}"
            )

        pred_scale = out_dict["scale"]
        pred_shape = out_dict["shape"]
        pred_rotate = out_dict["rotate"]
        pred_trans = out_dict["trans"]

        device = pred_scale.device
        gt_scale = batch["gt_scale"].to(device).float()
        gt_shape = batch["gt_shape"].to(device).float()
        gt_rotate = batch["gt_rotate"].to(device).float()
        gt_trans = batch["gt_trans"].to(device).float()
        K = batch["K"].to(device).long()

        B = pred_scale.shape[0]
        total = pred_scale.new_tensor(0.0)
        n_pairs = 0

        for b in range(B):
            K_b = int(K[b].item())
            if K_b <= 0:
                continue

            matched_slots = all_matched_slots[b][:K_b]

            gt_scale_b = gt_scale[b:b + 1, :K_b, :]
            gt_shape_b = gt_shape[b:b + 1, :K_b, :]
            gt_rotate_b = gt_rotate[b:b + 1, :K_b, :, :]
            gt_trans_b = gt_trans[b:b + 1, :K_b, :]

            pred_scale_b = pred_scale[b:b + 1, matched_slots, :]
            pred_shape_b = pred_shape[b:b + 1, matched_slots, :]
            pred_rotate_b = pred_rotate[b:b + 1, matched_slots, :, :]
            pred_trans_b = pred_trans[b:b + 1, matched_slots, :]

            etas, omegas = self._sample_gt_highcurv_eta_omega(gt_scale_b, gt_shape_b)

            pred_local_b, _ = self._sq_local_from_eta_omega(
                pred_scale_b,
                pred_shape_b,
                etas,
                omegas,
            )
            gt_local_b, _ = self._sq_local_from_eta_omega(
                gt_scale_b,
                gt_shape_b,
                etas,
                omegas,
            )

            pred_world_b = self._local_to_world(
                pred_local_b,
                pred_rotate_b,
                pred_trans_b,
            )[0]
            gt_world_b = self._local_to_world(
                gt_local_b,
                gt_rotate_b,
                gt_trans_b,
            )[0]

            per_primitive = ((pred_world_b - gt_world_b) ** 2).sum(dim=-1).mean(dim=-1)
            total = total + per_primitive.sum()
            n_pairs += K_b

        if n_pairs == 0:
            return pred_scale.new_tensor(0.0)

        return total / n_pairs

    def _compute_shape_oracle_surface_loss(self, out_dict, batch, all_matched_slots):
        """Matched GT-pose/scale + predicted-shape surface Chamfer.

        For each matched GT primitive k and predicted slot p=sigma(k), build:

            pred_oracle = SQ(gt_scale[k], pred_shape[p], gt_rotate[k], gt_trans[k])
            gt_surface  = SQ(gt_scale[k], gt_shape[k], gt_rotate[k], gt_trans[k])

        Thus the only predicted SQ parameter inside this surface loss is shape
        = [eps_1, eps_2]. Translation, scale, and rotation are oracle GT values.
        This is a synthetic diagnostic loss, not a real-data-compatible loss.
        """
        required = ["gt_scale", "gt_shape", "gt_rotate", "gt_trans"]
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "loss.w_sup_shape_oracle_surface > 0 requires GT SQ sidecars; "
                f"missing batch keys: {missing}"
            )

        pred_shape = out_dict["shape"]  # [B, P, 2]
        device = pred_shape.device

        gt_scale = batch["gt_scale"].to(device).float()
        gt_shape = batch["gt_shape"].to(device).float()
        gt_rotate = batch["gt_rotate"].to(device).float()
        gt_trans = batch["gt_trans"].to(device).float()
        K = batch["K"].to(device).long()

        B = pred_shape.shape[0]
        total = pred_shape.new_tensor(0.0)
        n_pairs = 0

        for b in range(B):
            K_b = int(K[b].item())
            if K_b <= 0:
                continue

            matched_slots = all_matched_slots[b][:K_b]

            gt_scale_b = gt_scale[b:b + 1, :K_b, :]
            gt_shape_b = gt_shape[b:b + 1, :K_b, :]
            gt_rotate_b = gt_rotate[b:b + 1, :K_b, :, :]
            gt_trans_b = gt_trans[b:b + 1, :K_b, :]

            # Reorder predicted shapes into GT-primitive order using the same
            # final Hungarian matching as assignment/existence.
            pred_shape_b = pred_shape[b:b + 1, matched_slots, :]

            pred_local_b, _ = sampling_from_parametric_space_to_equivalent_points(
                gt_scale_b,
                pred_shape_b,
                self.surface_sampler,
            )
            gt_local_b, _ = sampling_from_parametric_space_to_equivalent_points(
                gt_scale_b,
                gt_shape_b,
                self.surface_sampler,
            )

            pred_world_b = self._local_to_world(
                pred_local_b,
                gt_rotate_b,
                gt_trans_b,
            )[0]
            gt_world_b = self._local_to_world(
                gt_local_b,
                gt_rotate_b,
                gt_trans_b,
            )[0]

            for k in range(K_b):
                total = total + self._chamfer_squared(pred_world_b[k], gt_world_b[k])
                n_pairs += 1

        if n_pairs == 0:
            return pred_shape.new_tensor(0.0)

        return total / n_pairs

    def _compute_z_axis_loss(self, out_dict, batch, all_matched_slots):
        """Sign-invariant local z-axis supervision.

        L_z = mean_k 1 - (dot(z_pred, z_gt)^2)

        z_pred and -z_pred are treated as equivalent.
        """
        if "gt_rotate" not in batch:
            raise KeyError(
                "loss.w_sup_z_axis > 0 requires batch['gt_rotate']. "
                "Disable this loss for datasets without GT SQ rotations."
            )

        pred_rotate = out_dict["rotate"]  # [B, P, 3, 3], local-to-world
        gt_rotate = batch["gt_rotate"].to(pred_rotate.device).float()
        K = batch["K"].to(pred_rotate.device).long()

        total = pred_rotate.new_tensor(0.0)
        total_absdot = 0.0
        n_pairs = 0

        B = pred_rotate.shape[0]
        for b in range(B):
            K_b = int(K[b].item())
            matched_slots = all_matched_slots[b]

            for k in range(K_b):
                p = int(matched_slots[k].item())
                z_pred = pred_rotate[b, p, :, 2]
                z_gt = gt_rotate[b, k, :, 2]

                z_pred = torch.nn.functional.normalize(z_pred, dim=0)
                z_gt = torch.nn.functional.normalize(z_gt, dim=0)

                dot = torch.clamp(torch.dot(z_pred, z_gt), -1.0, 1.0)
                total = total + (1.0 - dot * dot)
                total_absdot += float(torch.abs(dot).detach().cpu().item())
                n_pairs += 1

        if n_pairs == 0:
            z = pred_rotate.new_tensor(0.0)
            return z, 0.0

        return total / n_pairs, total_absdot / n_pairs

    def _compute_shape_param_loss(self, out_dict, batch, all_matched_slots):
        """Direct L1 supervision for SQ shape exponents eps_1, eps_2.

        This is a synthetic SQ-Zero diagnostic loss. It should remain disabled
        for real part datasets that do not have GT SQ parameters.

        For matched GT primitive k and predicted slot p=sigma(k):

            loss_k = mean(|pred_shape[p] - gt_shape[k]|)

        where shape = [eps_1, eps_2].
        """
        if "gt_shape" not in batch:
            raise KeyError(
                "loss.w_sup_shape_param > 0 requires batch['gt_shape']. "
                "Disable this loss for real datasets without GT SQ parameters."
            )

        pred_shape = out_dict["shape"]  # [B, P, 2]
        gt_shape = batch["gt_shape"].to(pred_shape.device).float()
        K = batch["K"].to(pred_shape.device).long()

        total = pred_shape.new_tensor(0.0)
        n_pairs = 0

        B = pred_shape.shape[0]

        for b in range(B):
            K_b = int(K[b].item())
            matched_slots = all_matched_slots[b]

            for k in range(K_b):
                p = int(matched_slots[k].item())
                total = total + torch.abs(pred_shape[b, p] - gt_shape[b, k]).mean()
                n_pairs += 1

        if n_pairs == 0:
            return pred_shape.new_tensor(0.0)

        return total / n_pairs

    @staticmethod
    def _identity_rotation_like(trans):
        """Create [B, P, 3, 3] identity rotations on the same device/dtype."""
        B, P, _ = trans.shape
        eye = torch.eye(3, device=trans.device, dtype=trans.dtype)
        return eye.view(1, 1, 3, 3).expand(B, P, 3, 3)

    def _make_stage1_sphere_outdict(self, stage1):
        """Stage 1: translation-only fixed isotropic sphere."""
        trans = stage1["trans"]
        scale = torch.full_like(stage1["scale"], self.stage1_fixed_radius)
        shape = torch.ones_like(stage1["shape"])
        rotate = self._identity_rotation_like(trans)

        return {
            "scale": scale,
            "shape": shape,
            "rotate": rotate,
            "trans": trans,
        }

    def _make_stage2_ellipsoid_outdict(self, stage1, stage2):
        """Stage 2: scale + rotation only, using detached stage-1 translation."""
        return {
            "scale": stage2["scale"],
            "shape": torch.ones_like(stage2["shape"]),
            "rotate": stage2["rotate"],
            "trans": stage1["trans"].detach(),
        }

    def _make_stage3_shape_outdict(self, stage1, stage2, stage3):
        """Stage 3: shape only, with detached translation/scale/rotation."""
        return {
            "scale": stage2["scale"].detach(),
            "shape": stage3["shape"],
            "rotate": stage2["rotate"].detach(),
            "trans": stage1["trans"].detach(),
        }

    def _compute_staged_surface_losses(self, out_dict, batch, all_matched_slots):
        """Compute strict staged GT-surface losses.

        Matching is intentionally not recomputed here.  The final layer's
        assignment/existence already determined all_matched_slots in forward().
        """
        if "staged_outdicts" not in out_dict:
            raise KeyError(
                "loss.use_staged_surface=true requires model output "
                "out_dict['staged_outdicts']. Enable "
                "superdec.decoder.staged_params=true."
            )

        staged = out_dict["staged_outdicts"]
        if len(staged) < 3:
            raise ValueError(
                "Strict staged SQ v1 requires at least 3 staged decoder outputs, "
                f"but got {len(staged)}."
            )

        stage1 = staged[0]
        stage2 = staged[1]
        stage3 = staged[-1]

        stage1_out = self._make_stage1_sphere_outdict(stage1)
        stage2_out = self._make_stage2_ellipsoid_outdict(stage1, stage2)
        stage3_out = self._make_stage3_shape_outdict(stage1, stage2, stage3)

        stage1_loss = self._compute_gt_surface_loss(
            out_dict=stage1_out,
            batch=batch,
            all_matched_slots=all_matched_slots,
        )
        stage2_loss = self._compute_gt_surface_loss(
            out_dict=stage2_out,
            batch=batch,
            all_matched_slots=all_matched_slots,
        )
        stage3_loss = self._compute_gt_surface_loss(
            out_dict=stage3_out,
            batch=batch,
            all_matched_slots=all_matched_slots,
        )

        return stage1_loss, stage2_loss, stage3_loss

    def _compute_surface_loss(self, pc, out_dict, batch, all_matched_slots):
        if self.surface_target == "gt_surface":
            return self._compute_gt_surface_loss(
                out_dict=out_dict,
                batch=batch,
                all_matched_slots=all_matched_slots,
            )

        if self.surface_target == "part_points":
            return self._compute_part_point_surface_loss(
                pc=pc,
                batch=batch,
                out_dict=out_dict,
                all_matched_slots=all_matched_slots,
            )

        raise RuntimeError(f"Unhandled surface_target={self.surface_target!r}")

    def forward(self, pc, normals, out_dict, batch):
        assign = out_dict["assign_matrix"]          # [B, N, P]
        exist = out_dict["exist"].squeeze(-1)       # [B, P]

        labels = batch["labels"].to(assign.device).long()
        K = batch["K"].to(assign.device).long()

        B, N, _ = assign.shape

        self._forward_calls += 1
        should_debug_print = (
            self.debug_match_every > 0
            and self._forward_calls % self.debug_match_every == 0
        )

        phase = "train" if torch.is_grad_enabled() else "eval"

        total_exist_loss = assign.new_tensor(0.0)
        total_assign_loss = assign.new_tensor(0.0)

        total_assign_acc = 0.0
        total_count_acc = 0.0
        total_pred_count = 0.0
        total_soft_count = 0.0
        total_true_count = 0.0

        total_exist_pos_mean = 0.0
        total_exist_neg_mean = 0.0

        all_matched_slots = []
        debug_lines = []

        for b in range(B):
            K_b = int(K[b].item())

            matched_slots = self._hungarian_for_one(
                assign_b=assign[b],
                exist_b=exist[b],
                labels_b=labels[b],
                K_b=K_b,
            )
            all_matched_slots.append(matched_slots)

            # --------------------
            # Existence target
            # --------------------
            exist_target = torch.zeros_like(exist[b])
            exist_target[matched_slots] = 1.0

            exist_loss_b = nn.functional.binary_cross_entropy(
                exist[b],
                exist_target,
                reduction="mean",
            )

            # --------------------
            # Assignment target
            # --------------------
            target_slot = matched_slots[labels[b]]  # [N]
            point_indices = torch.arange(N, device=assign.device)
            chosen_probs = assign[b, point_indices, target_slot]

            assign_loss_b = -torch.log(chosen_probs.clamp_min(self.eps)).mean()

            # --------------------
            # Metrics
            # --------------------
            with torch.no_grad():
                pred_slot = assign[b].argmax(dim=1)
                assign_acc_b = (pred_slot == target_slot).float().mean().item()

                pred_count_b = (exist[b] > 0.5).sum().item()
                soft_count_b = exist[b].sum().item()
                true_count_b = K_b
                count_acc_b = float(pred_count_b == true_count_b)

                pos_vals = exist[b][exist_target > 0.5]
                neg_vals = exist[b][exist_target < 0.5]

                pos_mean_b = pos_vals.mean().item() if pos_vals.numel() > 0 else float("nan")
                neg_mean_b = neg_vals.mean().item() if neg_vals.numel() > 0 else float("nan")

                if should_debug_print and b < self.debug_match_max_samples:
                    match_str = ", ".join(
                        f"gt{k}->slot{int(p)}"
                        for k, p in enumerate(matched_slots.detach().cpu().tolist())
                    )

                    exist_str = ", ".join(
                        f"{x:.3f}" for x in exist[b].detach().cpu().tolist()
                    )

                    model_ids = batch.get("model_id", None)
                    if model_ids is not None:
                        model_id_b = model_ids[b]
                    else:
                        model_id_b = "unknown"

                    debug_lines.append(
                        f"[HungarianDebug] call={self._forward_calls} "
                        f"phase={phase} sample={b} model_id={model_id_b} K={K_b} "
                        f"matches=[{match_str}] "
                        f"exist=[{exist_str}] "
                        f"soft_count={soft_count_b:.3f} "
                        f"pred_count={int(pred_count_b)} "
                        f"assign_acc={assign_acc_b:.3f}"
                    )

            total_exist_loss = total_exist_loss + exist_loss_b
            total_assign_loss = total_assign_loss + assign_loss_b

            total_assign_acc += assign_acc_b
            total_count_acc += count_acc_b
            total_pred_count += float(pred_count_b)
            total_soft_count += float(soft_count_b)
            total_true_count += float(true_count_b)

            total_exist_pos_mean += float(pos_mean_b)
            total_exist_neg_mean += float(neg_mean_b)

        if should_debug_print and debug_lines:
            print("\n".join(debug_lines), flush=True)

        exist_loss = total_exist_loss / B
        assign_loss = total_assign_loss / B

        if self.w_surface > 0.0:
            surface_loss = self._compute_surface_loss(
                pc=pc,
                out_dict=out_dict,
                batch=batch,
                all_matched_slots=all_matched_slots,
            )
        else:
            surface_loss = assign.new_tensor(0.0)

        if self.use_staged_surface:
            (
                stage1_surface_loss,
                stage2_surface_loss,
                stage3_surface_loss,
            ) = self._compute_staged_surface_losses(
                out_dict=out_dict,
                batch=batch,
                all_matched_slots=all_matched_slots,
            )
        else:
            stage1_surface_loss = assign.new_tensor(0.0)
            stage2_surface_loss = assign.new_tensor(0.0)
            stage3_surface_loss = assign.new_tensor(0.0)

        if self.w_z_axis > 0.0:
            z_axis_loss, z_axis_absdot_mean = self._compute_z_axis_loss(
                out_dict=out_dict,
                batch=batch,
                all_matched_slots=all_matched_slots,
            )
        else:
            z_axis_loss = assign.new_tensor(0.0)
            z_axis_absdot_mean = 0.0

        if self.w_shape_param > 0.0:
            shape_param_loss = self._compute_shape_param_loss(
                out_dict=out_dict,
                batch=batch,
                all_matched_slots=all_matched_slots,
            )
        else:
            shape_param_loss = assign.new_tensor(0.0)

        if self.w_highcurv_chamfer_surface > 0.0:
            highcurv_chamfer_surface_loss = self._compute_highcurv_chamfer_surface_loss(
                out_dict=out_dict,
                batch=batch,
                all_matched_slots=all_matched_slots,
            )
        else:
            highcurv_chamfer_surface_loss = assign.new_tensor(0.0)

        if self.w_highcurv_pointwise_surface > 0.0:
            highcurv_pointwise_surface_loss = self._compute_highcurv_pointwise_surface_loss(
                out_dict=out_dict,
                batch=batch,
                all_matched_slots=all_matched_slots,
            )
        else:
            highcurv_pointwise_surface_loss = assign.new_tensor(0.0)

        if self.w_shape_oracle_surface > 0.0:
            shape_oracle_surface_loss = self._compute_shape_oracle_surface_loss(
                out_dict=out_dict,
                batch=batch,
                all_matched_slots=all_matched_slots,
            )
        else:
            shape_oracle_surface_loss = assign.new_tensor(0.0)

        if self.w_normal > 0.0:
            normal_loss, normal_stats = self._compute_part_normal_loss(
                pc=pc,
                normals=normals,
                batch=batch,
                out_dict=out_dict,
                all_matched_slots=all_matched_slots,
            )
        else:
            normal_loss = assign.new_tensor(0.0)
            normal_stats = {
                "sup_normal_dot_mean": 0.0,
                "sup_normal_absdot_mean": 0.0,
                "sup_normal_nn_dist_mean": 0.0,
            }

        if self.w_normal_dir > 0.0:
            normal_dir_loss, normal_dir_stats = self._compute_normal_direction_loss(
                normals=normals,
                batch=batch,
                out_dict=out_dict,
                all_matched_slots=all_matched_slots,
            )
        else:
            normal_dir_loss = assign.new_tensor(0.0)
            normal_dir_stats = {
                "sup_normal_dir_gt2pred_dot_mean": 0.0,
                "sup_normal_dir_pred2gt_dot_mean": 0.0,
            }

        loss = (
            self.w_exist * exist_loss
            + self.w_assign * assign_loss
            + self.w_surface * surface_loss
            + self.w_stage1_surface * stage1_surface_loss
            + self.w_stage2_surface * stage2_surface_loss
            + self.w_stage3_surface * stage3_surface_loss
            + self.w_z_axis * z_axis_loss
            + self.w_shape_param * shape_param_loss
            + self.w_shape_oracle_surface * shape_oracle_surface_loss
            + self.w_highcurv_pointwise_surface * highcurv_pointwise_surface_loss
            + self.w_highcurv_chamfer_surface * highcurv_chamfer_surface_loss
            + self.w_normal * normal_loss
            + self.w_normal_dir * normal_dir_loss
        )

        loss_dict = {
            "sup_exist_loss": float(exist_loss.detach().cpu().item()),
            "sup_assign_loss": float(assign_loss.detach().cpu().item()),
            "sup_surface_loss": float(surface_loss.detach().cpu().item()),
            "sup_stage1_surface_loss": float(stage1_surface_loss.detach().cpu().item()),
            "sup_stage2_surface_loss": float(stage2_surface_loss.detach().cpu().item()),
            "sup_stage3_surface_loss": float(stage3_surface_loss.detach().cpu().item()),
            "sup_z_axis_loss": float(z_axis_loss.detach().cpu().item()),
            "sup_z_axis_absdot_mean": z_axis_absdot_mean,
            "sup_shape_param_loss": float(shape_param_loss.detach().cpu().item()),
            "sup_shape_oracle_surface_loss": float(
                shape_oracle_surface_loss.detach().cpu().item()
            ),
            "sup_highcurv_pointwise_surface_loss": float(
                highcurv_pointwise_surface_loss.detach().cpu().item()
            ),
            "sup_highcurv_chamfer_surface_loss": float(
                highcurv_chamfer_surface_loss.detach().cpu().item()
            ),
            "sup_normal_loss": float(normal_loss.detach().cpu().item()),
            "sup_normal_dot_mean": normal_stats["sup_normal_dot_mean"],
            "sup_normal_absdot_mean": normal_stats["sup_normal_absdot_mean"],
            "sup_normal_nn_dist_mean": normal_stats["sup_normal_nn_dist_mean"],
            "sup_normal_dir_loss": float(normal_dir_loss.detach().cpu().item()),
            "sup_normal_dir_gt2pred_dot_mean": normal_dir_stats["sup_normal_dir_gt2pred_dot_mean"],
            "sup_normal_dir_pred2gt_dot_mean": normal_dir_stats["sup_normal_dir_pred2gt_dot_mean"],
            "sup_assign_acc": total_assign_acc / B,
            "sup_count_acc": total_count_acc / B,
            "sup_pred_count": total_pred_count / B,
            "sup_soft_count": total_soft_count / B,
            "sup_true_count": total_true_count / B,
            "sup_exist_pos_mean": total_exist_pos_mean / B,
            "sup_exist_neg_mean": total_exist_neg_mean / B,
            "all": float(loss.detach().cpu().item()),
        }

        return loss, loss_dict