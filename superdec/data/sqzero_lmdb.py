import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from superdec.data.dataloader import normalize_points, get_transforms

_LMDB_ENV_CACHE = {}

def _load_npy_from_lmdb_value(value: bytes) -> np.ndarray:
    if value is None:
        raise KeyError("Tried to decode a missing LMDB value.")
    return np.load(io.BytesIO(value), allow_pickle=False)


def _make_radial_normals(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centered, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (centered / denom).astype(np.float32)


class SQZeroLMDB(Dataset):
    """SQ-Zero LMDB dataset adapter for the existing SuperDec trainer.

    Expected LMDB keys:
        <key>              -> points, [N, 3], float32
        <key>.labels       -> primitive labels, [N], int32      optional for now
        <key>.sq_params    -> SQ params, [K, 11], float32       optional for now

    For original SuperDec training we only use points. Since the current SQ-Zero
    LMDB does not store normals, we synthesize radial normals and recommend
    setting loss.w_cub=0.0 for the first native LMDB run.
    """

    def __init__(self, split: str, cfg):
        super().__init__()

        self.split = split
        self.root = Path(cfg.sqzero_lmdb.path)
        self.n_points = int(getattr(cfg.sqzero_lmdb, "n_points", 4096))
        self.normalize = bool(getattr(cfg.sqzero_lmdb, "normalize", True))
        self.load_sidecars = bool(getattr(cfg.sqzero_lmdb, "load_sidecars", False))
        self.normal_mode = str(getattr(cfg.sqzero_lmdb, "normal_mode", "radial"))

        if split == "train":
            split_file = getattr(cfg.sqzero_lmdb, "train_split", "train.txt")
        elif split == "val":
            # SQ-Zero generator writes test.txt, while SuperDec trainer expects val.
            split_file = getattr(cfg.sqzero_lmdb, "val_split", "test.txt")
        else:
            split_file = f"{split}.txt"

        self.split_path = self.root / split_file
        if not self.split_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_path}")

        self.keys = [
            line.strip()
            for line in self.split_path.read_text().splitlines()
            if line.strip()
        ]

        if len(self.keys) == 0:
            raise RuntimeError(f"No keys found in split file: {self.split_path}")

        self.shard_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and (p / "data.mdb").exists()
        )
        if len(self.shard_dirs) == 0:
            raise RuntimeError(f"No LMDB shards found under: {self.root}")

        # LMDB environments are opened lazily per worker/process.
        self._envs: Optional[List[lmdb.Environment]] = None
        self._key_to_shard: Dict[str, int] = {}

        # Reuse SuperDec's existing augmentation function.
        self.transform = get_transforms(split, cfg)

    def __len__(self):
        return len(self.keys)

    def _open_envs(self):
        if self._envs is not None:
            return

        envs = []

        for shard_dir in self.shard_dirs:
            shard_path = str(shard_dir.resolve())

            if shard_path in _LMDB_ENV_CACHE:
                env = _LMDB_ENV_CACHE[shard_path]
            else:
                env = lmdb.open(
                    shard_path,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=256,
                )
                _LMDB_ENV_CACHE[shard_path] = env

            envs.append(env)

        self._envs = envs

    def _get_value(self, key: str) -> bytes:
        self._open_envs()

        # Fast path if we have already found this key before in this worker.
        shard_idx = self._key_to_shard.get(key)
        if shard_idx is not None:
            with self._envs[shard_idx].begin(write=False) as txn:
                value = txn.get(key.encode("utf-8"))
            if value is not None:
                return value

        # Robust path: try all shards. Fine for one/few shards.
        encoded = key.encode("utf-8")
        for i, env in enumerate(self._envs):
            with env.begin(write=False) as txn:
                value = txn.get(encoded)
            if value is not None:
                self._key_to_shard[key] = i
                return value

        raise KeyError(f"Key not found in any shard: {key}")

    def _load_array(self, key: str) -> np.ndarray:
        return _load_npy_from_lmdb_value(self._get_value(key))

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        points = points.astype(np.float32, copy=False)
        n = points.shape[0]

        if n >= self.n_points:
            idx = np.random.choice(n, self.n_points, replace=False)
        else:
            idx = np.random.choice(n, self.n_points, replace=True)

        return points[idx].astype(np.float32, copy=False)

    def __getitem__(self, idx):
        key = self.keys[idx]

        points = self._load_array(key)
        points = self._sample_points(points)

        if self.normalize:
            points, translation, scale = normalize_points(points)
        else:
            translation = np.zeros(3, dtype=np.float32)
            scale = np.float32(1.0)

        points = points.astype(np.float32, copy=False)

        if self.normal_mode == "radial":
            normals = _make_radial_normals(points)
        elif self.normal_mode == "zeros":
            normals = np.zeros_like(points, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported normal_mode: {self.normal_mode}")

        if self.transform is not None:
            t_data = self.transform(points=points, normals=normals)
            points = t_data["points"].astype(np.float32, copy=False)
            normals = t_data["normals"].astype(np.float32, copy=False)

        item = {
            "points": torch.from_numpy(points),
            "normals": torch.from_numpy(normals),
            "translation": torch.from_numpy(np.asarray(translation, dtype=np.float32)),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "point_num": points.shape[0],
            "model_id": key,
        }

        if self.load_sidecars:
            # Not used by original SuperDec. Useful later for supervised Hungarian.
            labels_key = f"{key}.labels"
            params_key = f"{key}.sq_params"

            item["labels"] = torch.from_numpy(self._load_array(labels_key).astype(np.int64))
            item["sq_params"] = torch.from_numpy(self._load_array(params_key).astype(np.float32))

        return item

    def name(self):
        return "SQZeroLMDB"
