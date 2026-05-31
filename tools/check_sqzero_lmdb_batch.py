#!/usr/bin/env python3
import argparse

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from superdec.data.sqzero_lmdb import SQZeroLMDB


def describe_tensor(name, x):
    print(f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}")
    if torch.is_floating_point(x):
        print(f"  min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
    else:
        print(f"  min={x.min().item()}, max={x.max().item()}")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    print("=== Effective SQZero config ===")
    print(OmegaConf.to_yaml(cfg.sqzero_lmdb))

    train_ds = SQZeroLMDB(split="train", cfg=cfg)
    val_ds = SQZeroLMDB(split="val", cfg=cfg)

    print("=== Dataset lengths ===")
    print("train:", len(train_ds))
    print("val:", len(val_ds))

    loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    batch = next(iter(loader))

    print("=== Batch keys ===")
    print(sorted(batch.keys()))

    print("=== Tensor fields ===")
    for k in [
        "points", "normals", "labels", "part_ids", "sq_params",
        "gt_scale", "gt_shape", "gt_rotate", "gt_trans",
        "valid_mask", "K"
    ]:
        if k in batch:
            describe_tensor(k, batch[k])

    print("=== Per-sample checks ===")
    labels = batch["labels"]
    sq_params = batch["sq_params"]
    valid_mask = batch["valid_mask"]
    K = batch["K"]

    batch_size = labels.shape[0]

    for b in range(batch_size):
        unique_labels = torch.unique(labels[b]).cpu().tolist()
        k = int(K[b].item())
        valid_count = int(valid_mask[b].sum().item())

        print(f"sample {b}:")
        print(f"  K={k}")
        print(f"  valid_mask.sum={valid_count}")
        print(f"  unique_labels={unique_labels}")
        print(f"  label_min={int(labels[b].min().item())}")
        print(f"  label_max={int(labels[b].max().item())}")
        print(f"  sq_params_valid_shape={tuple(sq_params[b, :k].shape)}")

        assert valid_count == k, f"valid_mask count {valid_count} != K {k}"
        assert int(labels[b].min().item()) >= 0
        assert int(labels[b].max().item()) < k
        assert sq_params[b, k:].abs().sum().item() == 0.0

    print("=== OK: supervised SQ-Zero LMDB batch is structurally valid ===")


if __name__ == "__main__":
    main()
