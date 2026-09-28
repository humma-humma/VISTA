"""
prepare_gt.py — Extract ground-truth feat67 and joints for unified evaluation.

For each sample in the manifest (styled + base), loads the raw GT motion .npy,
normalizes it to [T, 67], recovers 3-D joints [T, 22, 3], and saves both.

Usage:
    python prepare_gt.py --manifest manifest.json
    python prepare_gt.py --manifest manifest.json --resume
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import recover_from_ric from MM_MARDM
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFEST_DEFAULT = _SCRIPT_DIR / "manifest.json"


def _setup_mm_mardm_import(mm_mardm_root: Path):
    """Add MM_MARDM to sys.path so we can import utils.motion_process."""
    root_str = str(mm_mardm_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_recover_from_ric(mm_mardm_root: Path):
    _setup_mm_mardm_import(mm_mardm_root)
    from utils.motion_process import recover_from_ric  # noqa: E402
    return recover_from_ric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_humanml3d_stats(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load HumanML3D Mean and Std (shape [263]) from the mm_mardm paths."""
    mm_root = Path(manifest["path_roots"]["mm_mardm"])
    rel = manifest["relative_paths"]["humanml3d"]["mm_mardm"]
    mean = np.load(mm_root / rel["mean"])  # (263,)
    std = np.load(mm_root / rel["std"])    # (263,)
    return mean, std


def snap_length(length: int, unit: int = 4, cap: int = 196) -> int:
    """Snap length down to nearest multiple of *unit*, then cap."""
    return min((length // unit) * unit, cap)


def normalize_feat67(raw_67: np.ndarray, mean_67: np.ndarray, std_67: np.ndarray) -> np.ndarray:
    """Normalize raw [T, 67] features: (x - mean) / std."""
    return (raw_67 - mean_67) / std_67


def feat67_to_joints(feat67_normed: np.ndarray, mean_67: np.ndarray, std_67: np.ndarray,
                     recover_fn) -> np.ndarray:
    """Denormalize feat67 and recover world-space joints [T, 22, 3]."""
    denormed = feat67_normed * std_67 + mean_67
    tensor = torch.from_numpy(denormed).float()
    joints = recover_fn(tensor, 22)  # [T, 22, 3]
    return joints.numpy()


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def process_styled_samples(manifest: dict, mean: np.ndarray, std: np.ndarray,
                           recover_fn, out_dir: Path, resume: bool) -> tuple[int, int]:
    """Process all styled samples. Returns (success_count, fail_count)."""
    mm_root = Path(manifest["path_roots"]["mm_mardm"])
    motion_dir = mm_root / manifest["relative_paths"]["100style"]["mm_mardm"]["motion_dir"]

    mean_67 = mean[:67]
    std_67 = std[:67]

    meta = manifest["metadata"]
    unit_length = meta["unit_length"]
    max_length = meta["max_motion_length"]

    styled_dir = out_dir / "styled"
    styled_dir.mkdir(parents=True, exist_ok=True)

    samples = manifest["styled_samples"]
    ok, fail = 0, 0

    for s in tqdm(samples, desc="Styled GT", unit="sample"):
        mid = s["manifest_id"]
        feat_path = styled_dir / f"feat67_{mid:05d}.npy"
        joints_path = styled_dir / f"joints_{mid:05d}.npy"

        if resume and feat_path.exists() and joints_path.exists():
            ok += 1
            continue

        try:
            sample_id = s["sample_id"]
            raw_path = motion_dir / f"{sample_id}.npy"
            raw = np.load(raw_path)  # [T_raw, 263], float64

            # Snap length
            T_raw = raw.shape[0]
            T = snap_length(T_raw, unit=unit_length, cap=max_length)
            if T == 0:
                logging.warning("Styled %s (mid=%d): snapped length is 0, skipping", sample_id, mid)
                fail += 1
                continue

            # Crop from beginning, slice to 67 dims
            raw_67 = raw[:T, :67].astype(np.float32)

            # Normalize with HumanML3D stats
            feat67 = normalize_feat67(raw_67, mean_67.astype(np.float32), std_67.astype(np.float32))

            # Recover joints
            joints = feat67_to_joints(feat67, mean_67.astype(np.float32), std_67.astype(np.float32), recover_fn)

            # Save
            np.save(feat_path, feat67)
            np.save(joints_path, joints)
            ok += 1

        except Exception:
            logging.exception("Styled sample_id=%s manifest_id=%d failed", s.get("sample_id", "?"), mid)
            fail += 1

    return ok, fail


def process_base_samples(manifest: dict, mean: np.ndarray, std: np.ndarray,
                         recover_fn, out_dir: Path, resume: bool) -> tuple[int, int]:
    """Process all base (HumanML3D) samples. Returns (success_count, fail_count)."""
    mm_root = Path(manifest["path_roots"]["mm_mardm"])
    sliced_dir = mm_root / manifest["relative_paths"]["humanml3d"]["mm_mardm"]["sliced_motion_dir"]

    mean_67 = mean[:67].astype(np.float32)
    std_67 = std[:67].astype(np.float32)

    max_length = manifest["metadata"]["max_motion_length"]

    base_dir = out_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    samples = manifest["base_samples"]
    ok, fail = 0, 0

    for s in tqdm(samples, desc="Base GT", unit="sample"):
        mid = s["manifest_id"]
        feat_path = base_dir / f"feat67_{mid:05d}.npy"
        joints_path = base_dir / f"joints_{mid:05d}.npy"

        if resume and feat_path.exists() and joints_path.exists():
            ok += 1
            continue

        try:
            sliced_id = s["sliced_id"]
            raw_path = sliced_dir / f"{sliced_id}.npy"
            raw = np.load(raw_path)  # [T_raw, 67], float32 (already 67-dim)

            # Handle both 263-dim and 67-dim sliced files
            if raw.shape[1] > 67:
                raw = raw[:, :67]

            # Use length_raw_frames from manifest (= length_latent_frames * unit_length).
            # This is the frame count the models produce, so GT must match exactly.
            T = min(s["length_raw_frames"], max_length)

            if T == 0:
                logging.warning("Base %s (mid=%d): length is 0, skipping", sliced_id, mid)
                fail += 1
                continue

            raw_67 = raw[:T].astype(np.float32)

            # Normalize with HumanML3D stats
            feat67 = normalize_feat67(raw_67, mean_67, std_67)

            # Recover joints
            joints = feat67_to_joints(feat67, mean_67, std_67, recover_fn)

            # Save
            np.save(feat_path, feat67)
            np.save(joints_path, joints)
            ok += 1

        except Exception:
            logging.exception("Base sliced_id=%s manifest_id=%d failed", s.get("sliced_id", "?"), mid)
            fail += 1

    return ok, fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare ground-truth feat67 and joints for unified evaluation."
    )
    parser.add_argument(
        "--manifest", type=str, default=str(_MANIFEST_DEFAULT),
        help="Path to manifest.json (default: %(default)s)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override GT output directory (default: <manifest_dir>/gt)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip samples whose output files already exist."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load manifest
    manifest_path = Path(args.manifest)
    logging.info("Loading manifest from %s", manifest_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Load HumanML3D stats
    mean, std = load_humanml3d_stats(manifest)
    logging.info("Loaded HumanML3D Mean/Std (shape %s)", mean.shape)

    # Import recover_from_ric
    mm_root = Path(manifest["path_roots"]["mm_mardm"])
    recover_fn = _load_recover_from_ric(mm_root)
    logging.info("Imported recover_from_ric from %s", mm_root)

    # Output directory
    out_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent / "gt"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", out_dir)

    # Process styled samples
    styled_ok, styled_fail = process_styled_samples(
        manifest, mean, std, recover_fn, out_dir, args.resume
    )

    # Process base samples
    base_ok, base_fail = process_base_samples(
        manifest, mean, std, recover_fn, out_dir, args.resume
    )

    # Summary
    total_ok = styled_ok + base_ok
    total_fail = styled_fail + base_fail
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("  Styled: %d OK, %d failed (of %d)",
                 styled_ok, styled_fail, len(manifest["styled_samples"]))
    logging.info("  Base:   %d OK, %d failed (of %d)",
                 base_ok, base_fail, len(manifest["base_samples"]))
    logging.info("  Total:  %d OK, %d failed", total_ok, total_fail)
    logging.info("=" * 60)

    if total_fail > 0:
        logging.warning("%d sample(s) failed — check logs above for details.", total_fail)
        sys.exit(1)


if __name__ == "__main__":
    main()
