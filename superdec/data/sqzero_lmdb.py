import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from superdec.data.dataloader import normalize_points, get_transforms


# Per-process LMDB environment cache.
# Needed because train_ds and val_ds may open the same shard in the same process.
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

def _euler_xyz_to_matrix_np(euler: np.ndarray) -> np.ndarray:
    """Convert SQ-Zero xyz Euler angles to rotation matrices.

    This matches:
        scipy.spatial.transform.Rotation.from_euler("xyz", euler).as_matrix()

    With column-vector convention:
        p_world = R @ p_local + t

    Your sampler uses row vectors:
        P = P @ R.T + t

    which is equivalent.
    """
    euler = euler.astype(np.float32, copy=False)

    x = euler[:, 0]
    y = euler[:, 1]
    z = euler[:, 2]

    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    R = np.empty((euler.shape[0], 3, 3), dtype=np.float32)

    # R = Rz @ Ry @ Rx, matching scipy Rotation.from_euler("xyz", ...)
    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = cz * sx * sy - cx * sz
    R[:, 0, 2] = sx * sz + cx * cz * sy

    R[:, 1, 0] = cy * sz
    R[:, 1, 1] = cx * cz + sx * sy * sz
    R[:, 1, 2] = cx * sy * sz - cz * sx

    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cx * cy

    return R


def _convert_sqzero_params_to_superdec_targets(
    sq_params: np.ndarray,
    kmax: int,
    translation: np.ndarray,
    scale: float,
):
    """Convert raw SQ-Zero [K,11] params into normalized SuperDec GT tensors.

    SQ-Zero raw convention:
        [a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z]

    SuperDec target convention:
        gt_scale:  [Kmax, 3]
        gt_shape:  [Kmax, 2]
        gt_rotate: [Kmax, 3, 3]
        gt_trans:  [Kmax, 3]
        valid_mask:[Kmax]
    """
    sq_params = sq_params.astype(np.float32, copy=False)

    if sq_params.ndim != 2 or sq_params.shape[1] != 11:
        raise ValueError(f"Expected sq_params shape [K, 11], got {sq_params.shape}")

    k = int(sq_params.shape[0])
    if k > kmax:
        raise ValueError(f"K={k} exceeds kmax={kmax}")

    gt_scale = np.zeros((kmax, 3), dtype=np.float32)
    gt_shape = np.zeros((kmax, 2), dtype=np.float32)
    gt_rotate = np.zeros((kmax, 3, 3), dtype=np.float32)
    gt_trans = np.zeros((kmax, 3), dtype=np.float32)
    valid_mask = np.zeros((kmax,), dtype=np.bool_)

    raw_scale = sq_params[:, 0:3]
    raw_shape = sq_params[:, 3:5]
    raw_euler = sq_params[:, 5:8]
    raw_trans = sq_params[:, 8:11]

    scale = np.float32(scale)
    translation = translation.astype(np.float32, copy=False).reshape(1, 3)

    gt_scale[:k] = raw_scale / scale
    gt_shape[:k] = raw_shape
    gt_rotate[:k] = _euler_xyz_to_matrix_np(raw_euler)
    gt_trans[:k] = (raw_trans - translation) / scale
    valid_mask[:k] = True

    return gt_scale, gt_shape, gt_rotate, gt_trans, valid_mask, k

class SQZeroLMDB(Dataset):
    """SQ-Zero LMDB dataset adapter for SuperDec.

    Expected LMDB keys:
        <key>              -> points, [N, 3], float32
        <key>.labels       -> primitive labels, [N], int32
        <key>.sq_params    -> SQ params, [K, 11], float32
        <key>.normals      -> optional point normals, [N, 3], float32

    For original SuperDec training, only points/normals are used.
    For supervised Hungarian training later, use:
        load_sidecars: true
    which additionally returns sampled labels, padded SQ params, valid_mask, and K.
    """

    def __init__(self, split: str, cfg):
        super().__init__()

        self.split = split
        self.root = Path(cfg.sqzero_lmdb.path)
        self.n_points = int(getattr(cfg.sqzero_lmdb, "n_points", 4096))
        self.normalize = bool(getattr(cfg.sqzero_lmdb, "normalize", True))
        self.load_sidecars = bool(getattr(cfg.sqzero_lmdb, "load_sidecars", False))
        self.normal_mode = str(getattr(cfg.sqzero_lmdb, "normal_mode", "radial"))
        self.kmax = int(getattr(cfg.sqzero_lmdb, "kmax", 8))

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

        if split == "train":
            max_samples = getattr(cfg.sqzero_lmdb, "max_train_samples", None)
        elif split == "val":
            max_samples = getattr(cfg.sqzero_lmdb, "max_val_samples", None)
        else:
            max_samples = None

        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples > 0:
                self.keys = self.keys[:max_samples]

        self.shard_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and (p / "data.mdb").exists()
        )

        print(f"[SQZeroLMDB] split={split} split_file={self.split_path}")
        print(f"[SQZeroLMDB] n_keys={len(self.keys)} first_keys={self.keys[:5]}")

        if len(self.shard_dirs) == 0:
            raise RuntimeError(f"No LMDB shards found under: {self.root}")

        # Open lazily per process / worker.
        self._envs: Optional[List[lmdb.Environment]] = None
        self._key_to_shard: Dict[str, int] = {}

        self.transform = get_transforms(split, cfg)

        # Important: if we return GT SQ params, random point-cloud augmentations
        # would also need to update gt_rotate / gt_trans. That is not implemented yet.
        if self.load_sidecars and self.transform is not None:
            raise ValueError(
                "SQZeroLMDB load_sidecars=true is incompatible with "
                "trainer.augmentations=true for now, because GT SQ "
                "rotations/translations are not updated under augmentation."
            )

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

        shard_idx = self._key_to_shard.get(key)
        if shard_idx is not None:
            with self._envs[shard_idx].begin(write=False) as txn:
                value = txn.get(key.encode("utf-8"))
            if value is not None:
                return value

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

    def _sample_indices(self, n_available: int) -> np.ndarray:
        if n_available >= self.n_points:
            return np.random.choice(n_available, self.n_points, replace=False)
        return np.random.choice(n_available, self.n_points, replace=True)

    def _pad_sq_params(self, sq_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        sq_params = sq_params.astype(np.float32, copy=False)

        if sq_params.ndim != 2 or sq_params.shape[1] != 11:
            raise ValueError(f"Expected sq_params shape [K, 11], got {sq_params.shape}")

        k = int(sq_params.shape[0])

        if k > self.kmax:
            raise ValueError(
                f"Sample has K={k} primitives, but dataset kmax={self.kmax}. "
                f"Increase sqzero_lmdb.kmax."
            )

        padded = np.zeros((self.kmax, sq_params.shape[1]), dtype=np.float32)
        valid_mask = np.zeros((self.kmax,), dtype=np.bool_)

        padded[:k] = sq_params
        valid_mask[:k] = True

        return padded, valid_mask, k

    def __getitem__(self, idx):
        key = self.keys[idx]

        raw_points = self._load_array(key).astype(np.float32, copy=False)

        if raw_points.ndim != 2 or raw_points.shape[1] != 3:
            raise ValueError(
                f"Expected points shape [N, 3], got {raw_points.shape} for key={key}"
            )

        raw_normals = None
        if self.normal_mode == "sidecar":
            raw_normals = self._load_array(f"{key}.normals").astype(np.float32, copy=False)
            if raw_normals.ndim != 2 or raw_normals.shape[1] != 3:
                raise ValueError(
                    f"Expected normals shape [N, 3], got {raw_normals.shape} for key={key}"
                )
            if raw_normals.shape[0] != raw_points.shape[0]:
                raise ValueError(
                    f"normals length {raw_normals.shape[0]} does not match "
                    f"points length {raw_points.shape[0]} for key={key}"
                )

        # Sample point indices once, then use the same indices for labels/normals.
        sample_idx = self._sample_indices(raw_points.shape[0])
        points = raw_points[sample_idx].astype(np.float32, copy=False)

        labels = None
        sq_params_padded = None
        raw_valid_mask = None
        raw_k = None

        gt_scale = None
        gt_shape = None
        gt_rotate = None
        gt_trans = None
        valid_mask = None
        k = None

        if self.load_sidecars:
            raw_labels = self._load_array(f"{key}.labels").astype(np.int64, copy=False)
            raw_sq_params = self._load_array(f"{key}.sq_params").astype(np.float32, copy=False)

            if raw_labels.ndim != 1:
                raise ValueError(
                    f"Expected labels shape [N], got {raw_labels.shape} for key={key}"
                )

            if raw_labels.shape[0] != raw_points.shape[0]:
                raise ValueError(
                    f"labels length {raw_labels.shape[0]} does not match "
                    f"points length {raw_points.shape[0]} for key={key}"
                )

            labels = raw_labels[sample_idx].astype(np.int64, copy=False)

            sq_params_padded, raw_valid_mask, raw_k = self._pad_sq_params(raw_sq_params)

            if labels.min() < 0:
                raise ValueError(f"Negative labels found for key={key}: min={labels.min()}")

            if labels.max() >= raw_k:
                raise ValueError(
                    f"Label max {labels.max()} >= K={raw_k} for key={key}. "
                    f"Labels and sq_params are inconsistent."
                )

        # Normalize points using SuperDec's own convention.
        #
        # normalize_points returns:
        #   points_norm = (points_raw - translation) / scale
        #
        # Therefore GT SQ parameters must be transformed into the same frame:
        #   gt_scale = raw_scale / scale
        #   gt_trans = (raw_trans - translation) / scale
        #   gt_rotate = raw_rotate
        #   gt_shape = raw_shape
        if self.normalize:
            points, translation, scale = normalize_points(points)
        else:
            translation = np.zeros(3, dtype=np.float32)
            scale = np.float32(1.0)

        points = points.astype(np.float32, copy=False)

        if self.load_sidecars:
            gt_scale, gt_shape, gt_rotate, gt_trans, valid_mask, k = (
                _convert_sqzero_params_to_superdec_targets(
                    raw_sq_params,
                    kmax=self.kmax,
                    translation=np.asarray(translation, dtype=np.float32),
                    scale=float(scale),
                )
            )

            if k != raw_k:
                raise RuntimeError(
                    f"Internal K mismatch for key={key}: converted K={k}, raw K={raw_k}"
                )

            if not np.array_equal(valid_mask, raw_valid_mask):
                raise RuntimeError(f"Internal valid_mask mismatch for key={key}")

        if self.normal_mode == "radial":
            normals = _make_radial_normals(points)
        elif self.normal_mode == "zeros":
            normals = np.zeros_like(points, dtype=np.float32)
        elif self.normal_mode == "sidecar":
            normals = raw_normals[sample_idx].astype(np.float32, copy=False)
            # Normalization is translation + isotropic scale only, so normals are
            # unchanged apart from numerical renormalization. Do not translate or
            # scale normals.
            denom = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(denom, 1e-8)
            normals = normals.astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"Unsupported normal_mode: {self.normal_mode}. "
                "Expected one of: radial, zeros, sidecar."
            )

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
            item["labels"] = torch.from_numpy(labels)
            item["part_ids"] = torch.from_numpy(labels)

            # Raw padded SQ-Zero parameters:
            # [a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z]
            # Useful for debugging / traceability.
            item["sq_params"] = torch.from_numpy(sq_params_padded)

            # SuperDec-style normalized GT parameters.
            # These are the fields to use for future supervised losses.
            item["gt_scale"] = torch.from_numpy(gt_scale)
            item["gt_shape"] = torch.from_numpy(gt_shape)
            item["gt_rotate"] = torch.from_numpy(gt_rotate)
            item["gt_trans"] = torch.from_numpy(gt_trans)
            item["valid_mask"] = torch.from_numpy(valid_mask)
            item["K"] = torch.tensor(k, dtype=torch.long)

        return item

    def name(self):
        return "SQZeroLMDB"