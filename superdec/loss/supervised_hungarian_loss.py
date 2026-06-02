import numpy as np
import torch
import torch.nn as nn

from superdec.loss.loss import sampling_from_parametric_space_to_equivalent_points
from superdec.loss.sampler import EqualDistanceSamplerSQ

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    linear_sum_assignment = None
    _SCIPY_IMPORT_ERROR = exc


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

        # Matching-cost weights. These affect only the discrete Hungarian step.
        self.match_w_assign = float(getattr(cfg, "match_w_assign", 1.0))
        self.match_w_exist = float(getattr(cfg, "match_w_exist", 0.0))

        self.eps = float(getattr(cfg, "eps", 1e-8))

        # Geometry target.
        self.surface_target = str(getattr(cfg, "surface_target", "gt_surface"))
        valid_targets = {"gt_surface", "part_points"}
        if self.surface_target not in valid_targets:
            raise ValueError(
                f"Unknown loss.surface_target={self.surface_target!r}. "
                f"Expected one of {sorted(valid_targets)}."
            )

        # For part-points Chamfer:
        # L = d(part -> pred) + beta * d(pred -> part)
        self.part_cd_beta = float(getattr(cfg, "part_cd_beta", 0.0))
        self.part_cd_max_points_per_part = int(
            getattr(cfg, "part_cd_max_points_per_part", 512)
        )

        # Surface sampler.
        self.surface_n_samples = int(getattr(cfg, "surface_n_samples", 128))
        self.surface_sampler = EqualDistanceSamplerSQ(
            n_samples=self.surface_n_samples,
            D_eta=float(getattr(cfg, "surface_D_eta", 0.05)),
            D_omega=float(getattr(cfg, "surface_D_omega", 0.05)),
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

    def _sample_predicted_surfaces(self, out_dict):
        """Sample predicted SQ surfaces for all predicted slots.

        Returns:
            pred_world: [B, P, S, 3]
        """
        pred_local, _ = sampling_from_parametric_space_to_equivalent_points(
            out_dict["scale"],
            out_dict["shape"],
            self.surface_sampler,
        )

        pred_world = self._local_to_world(
            pred_local,
            out_dict["rotate"],
            out_dict["trans"],
        )

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

        loss = (
            self.w_exist * exist_loss
            + self.w_assign * assign_loss
            + self.w_surface * surface_loss
        )

        loss_dict = {
            "sup_exist_loss": float(exist_loss.detach().cpu().item()),
            "sup_assign_loss": float(assign_loss.detach().cpu().item()),
            "sup_surface_loss": float(surface_loss.detach().cpu().item()),
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