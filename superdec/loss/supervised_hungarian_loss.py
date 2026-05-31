import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    linear_sum_assignment = None
    _SCIPY_IMPORT_ERROR = exc


class SupervisedHungarianLoss(nn.Module):
    """First supervised DETR-style loss for SQ-Zero.

    Uses only:
      - out_dict["assign_matrix"]: [B, N, P], point-to-slot probabilities
      - out_dict["exist"]:         [B, P, 1], slot existence probabilities
      - batch["labels"]:           [B, N], GT primitive id per point
      - batch["K"]:                [B], number of GT primitives

    Matching is computed per sample with Hungarian assignment.
    Matching itself is non-differentiable and done under no_grad, as in DETR.
    The losses after matching are differentiable.
    """

    requires_batch = True

    def __init__(self, cfg):
        super().__init__()

        if linear_sum_assignment is None:
            raise ImportError(
                "SupervisedHungarianLoss requires scipy.optimize.linear_sum_assignment. "
                f"Original import error: {_SCIPY_IMPORT_ERROR}"
            )

        self.w_exist = float(getattr(cfg, "w_sup_exist", 1.0))
        self.w_assign = float(getattr(cfg, "w_sup_assign", 1.0))

        self._forward_calls = 0

        # Matching-cost weights. These only affect the discrete matching step.
        self.match_w_assign = float(getattr(cfg, "match_w_assign", 1.0))
        self.match_w_exist = float(getattr(cfg, "match_w_exist", 0.1))

        # Debug logging. If > 0, print Hungarian matches every N forward calls.
        # For tiny-overfit with batch_size=16 and one batch per epoch,
        # debug_match_every=10 means roughly every 10 epochs.
        self.debug_match_every = int(getattr(cfg, "debug_match_every", 0))
        self.debug_match_max_samples = int(getattr(cfg, "debug_match_max_samples", 4))

        self.eps = float(getattr(cfg, "eps", 1e-8))

    @torch.no_grad()
    def _hungarian_for_one(self, assign_b, exist_b, labels_b, K_b):
        """Compute slot-to-GT matching for one batch item.

        Args:
            assign_b: [N, P]
            exist_b:  [P]
            labels_b: [N]
            K_b: int

        Returns:
            matched_slots: [K] long tensor.
                matched_slots[k] = predicted slot p matched to GT primitive k.
        """
        N, P = assign_b.shape
        K = int(K_b)

        if K < 1:
            raise ValueError("K must be >= 1 for SQ-Zero supervised samples.")
        if K > P:
            raise ValueError(f"K={K} cannot exceed number of predicted slots P={P}.")

        # cost[p, k]
        cost = assign_b.new_zeros((P, K))

        for k in range(K):
            mask = labels_b == k
            n_k = int(mask.sum().item())
            if n_k == 0:
                # This should not happen for your current data/sampling, but make
                # it very costly if a primitive has no sampled points.
                cost[:, k] = 1e6
                continue

            # Low cost if slot p assigns high probability to points of GT primitive k.
            probs = assign_b[mask, :]  # [n_k, P]
            assign_cost = -torch.log(probs.clamp_min(self.eps)).mean(dim=0)  # [P]

            # Low cost if slot p has high existence probability.
            exist_cost = -torch.log(exist_b.clamp_min(self.eps))  # [P]

            cost[:, k] = (
                self.match_w_assign * assign_cost
                + self.match_w_exist * exist_cost
            )

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        # linear_sum_assignment returns arbitrary order. We want matched_slots[k] = p.
        matched_slots_np = np.empty((K,), dtype=np.int64)
        for p, k in zip(row_ind, col_ind):
            matched_slots_np[k] = p

        return torch.as_tensor(
            matched_slots_np,
            device=assign_b.device,
            dtype=torch.long,
        )

    def forward(self, pc, normals, out_dict, batch):
        assign = out_dict["assign_matrix"]          # [B, N, P], probabilities
        exist = out_dict["exist"].squeeze(-1)       # [B, P], probabilities

        labels = batch["labels"].to(assign.device).long()  # [B, N]
        K = batch["K"].to(assign.device).long()            # [B]

        B, N, P = assign.shape

        self._forward_calls += 1
        should_debug_print = (
            self.debug_match_every > 0
            and self._forward_calls % self.debug_match_every == 0
        )

        # torch.is_grad_enabled() is false during Trainer.evaluate(),
        # because evaluate is wrapped in @torch.no_grad().
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

        debug_lines = []

        for b in range(B):
            K_b = int(K[b].item())

            matched_slots = self._hungarian_for_one(
                assign_b=assign[b],
                exist_b=exist[b],
                labels_b=labels[b],
                K_b=K_b,
            )  # [K_b], matched_slots[k] = p

            # --------------------
            # Existence target
            # --------------------
            exist_target = torch.zeros_like(exist[b])  # [P]
            exist_target[matched_slots] = 1.0

            exist_loss_b = nn.functional.binary_cross_entropy(
                exist[b],
                exist_target,
                reduction="mean",
            )

            # --------------------
            # Assignment target
            # --------------------
            # For each point i, GT primitive labels[b,i] = k.
            # The target predicted slot is matched_slots[k].
            target_slot = matched_slots[labels[b]]  # [N]

            point_indices = torch.arange(N, device=assign.device)
            chosen_probs = assign[b, point_indices, target_slot]  # [N]

            assign_loss_b = -torch.log(chosen_probs.clamp_min(self.eps)).mean()

            # --------------------
            # Metrics
            # --------------------
            with torch.no_grad():
                pred_slot = assign[b].argmax(dim=1)  # [N]
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
                    # Show GT primitive k -> predicted slot p.
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

        loss = self.w_exist * exist_loss + self.w_assign * assign_loss

        loss_dict = {
            "sup_exist_loss": float(exist_loss.detach().cpu().item()),
            "sup_assign_loss": float(assign_loss.detach().cpu().item()),
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
