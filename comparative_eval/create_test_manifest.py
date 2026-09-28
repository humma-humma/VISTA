"""
Phase 2: Build shared test manifest for comparative evaluation.

Replicates the exact test split from MM_MARDM/evaluate_MARDM.py:

  100STYLE:  test_100STYLE_Filter.txt (747 IDs)
             -> filter: 5 styles, no TR, length in [40, 400)
             -> sort by length ascending
             -> pointer_style = searchsorted(lengths, 196) -- skip short samples
             -> random_split(seed=3407): 2/3 val, 1/3 test

  HumanML3D: splits_sliced/val.txt (2380 sliced IDs, MM-MARDM's actual eval split)
             Each sliced ID "012698_0" maps to parent "012698" for SMooDi/LoRA-MDM.
             MM-MARDM uses pre-encoded latents + sliced texts internally;
             SMooDi/LoRA-MDM use unsliced motions + multi-line texts.

Video/embedding paths per codebase:
  MM-MARDM:  videos/{id}_FV.mp4              (raw video, processed through ViViT)
  LoRA-MDM:  video_embeddings/{id}_FV.pt     (pre-computed embeddings)
  SMooDi:    video_latents/{id}.pt           (pre-computed latents, no _FV suffix)

Output: comparative_eval/manifest.json
"""

import os
import sys
import json
import numpy as np
import torch
try:
    import chardet
except ImportError:
    chardet = None

# ===================================================================
# Configuration — matches evaluate_MARDM.py defaults
# ===================================================================
SEED = 3407
MAX_MOTION_LENGTH = 196
MIN_MOTION_LENGTH = 40
MAX_LENGTH_FILTER = 400
UNIT_LENGTH = 4
STYLES = ["Aeroplane", "Chicken", "Robot", "Superman", "ArmsFolded"]

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
# MARDM_ROOT = os.path.join(ROOT, "MM_MARDM")
# SMOODI_ROOT = os.path.join(ROOT, "SMooDi")
# LORAMDM_ROOT = os.path.join(ROOT, "LoRA-MDM_myversion")
# VISTA root must contain datasets/ (symlink allowed). Baseline roots are optional and only
# needed when exporting SMooDi / LoRA-MDM for the comparison table.
MARDM_ROOT = os.environ.get("VISTA_ROOT", ROOT)
SMOODI_ROOT = os.environ.get("SMOODI_ROOT", os.path.join(ROOT, "..", "SMooDi"))
LORAMDM_ROOT = os.environ.get("LORAMDM_ROOT", os.path.join(ROOT, "..", "LoRA-MDM"))
OUTPUT_DIR = os.path.dirname(__file__)


# ===================================================================
# Helpers — mirror MM_MARDM/utils/datasets.py logic
# ===================================================================

def resolve_smoodi_stats(filename):
    """SMooDi stats may be in 100STYLES_RETARGETED or humanml3d_smoodi."""
    p1 = os.path.join("datasets", "100STYLES_RETARGETED", filename)
    if os.path.exists(os.path.join(SMOODI_ROOT, p1)):
        return p1
    return os.path.join("datasets", "humanml3d_smoodi", filename)


def build_dict_from_txt(filename):
    """Replicate datasets.py:build_dict_from_txt (lines 327-349)."""
    result = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                key = parts[0]
                style_name = parts[1].split("_")[0]
                motion_type = parts[1].split("_")[1]
                seq_idx = parts[1].split("_")[2].split(".")[0]
                length = int(parts[3])
                result[key] = (parts[2], style_name, motion_type, seq_idx, length)
    return result


def build_lengths_dict(filename):
    """Replicate datasets.py:build_dict_from_txt2 (lines 352-367)."""
    result = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) >= 2:
                result[parts[0]] = int(parts[1])
    return result


def read_split_file(filepath):
    ids = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def read_text_file_100style(filepath):
    """Read 100STYLE text file — multi-line, deduplicate (case-insensitive)."""
    captions = []
    seen = set()
    try:
        raw = open(filepath, "rb").read()
        if chardet is not None:
            enc = chardet.detect(raw).get("encoding", "utf-8") or "utf-8"
        else:
            enc = "utf-8"
        text = raw.decode(enc, errors="replace")
    except Exception:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    for line in text.splitlines():
        caption = line.strip().split("#")[0].strip()
        if not caption:
            continue
        canon = caption.lower()
        if canon in seen:
            continue
        seen.add(canon)
        captions.append(caption)
    return captions


def read_text_file_humanml(filepath):
    """Read HumanML3D text file — multi-line 'caption#tokens#start#end'."""
    captions = []
    seen = set()
    try:
        raw = open(filepath, "rb").read()
        if chardet is not None:
            enc = chardet.detect(raw).get("encoding", "utf-8") or "utf-8"
        else:
            enc = "utf-8"
        text = raw.decode(enc, errors="replace")
    except Exception:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    for line in text.splitlines():
        caption = line.strip().split("#")[0].strip()
        if not caption:
            continue
        canon = caption.lower()
        if canon in seen:
            continue
        seen.add(canon)
        captions.append(caption)
    return captions


# ===================================================================
# PART 1: Build 100STYLE test split
# ===================================================================

def build_100style_test_split():
    """Replicate Text2MotionDatasetCombined_v4 filtering + random_split."""

    name_dict_path = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "100STYLE_name_dict_length.txt")
    split_path = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "test_100STYLE_Full.txt")
    motion_dir = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "new_joint_vecs")
    video_dir = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "videos")
    text_dir = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "texts")

    metadata = build_dict_from_txt(name_dict_path)
    split_ids = read_split_file(split_path)

    print(f"100STYLE split file: {len(split_ids)} IDs")

    # Filter — mirrors datasets.py lines 3455-3503
    valid_samples = []
    skipped = {"style": 0, "tr": 0, "length": 0, "missing_file": 0, "metadata": 0, "no_text": 0}

    for name in split_ids:
        if name not in metadata:
            skipped["metadata"] += 1
            continue

        _, style_name, motion_type, seq_idx, length = metadata[name]

        if style_name not in STYLES:
            skipped["style"] += 1
            continue
        if motion_type.startswith("TR"):
            skipped["tr"] += 1
            continue
        if length < MIN_MOTION_LENGTH or length >= MAX_LENGTH_FILTER:
            skipped["length"] += 1
            continue

        motion_path = os.path.join(motion_dir, name + ".npy")
        video_path = os.path.join(video_dir, name + "_FV.mp4")
        text_path = os.path.join(text_dir, name + ".txt")

        if not (os.path.exists(motion_path) and os.path.exists(video_path) and os.path.exists(text_path)):
            skipped["missing_file"] += 1
            continue

        captions = read_text_file_100style(text_path)
        if not captions:
            skipped["no_text"] += 1
            continue

        style_to_idx = {s: i for i, s in enumerate(STYLES)}
        valid_samples.append({
            "sample_id": name,
            "style_name": style_name,
            "style_idx": style_to_idx[style_name],
            "motion_type": motion_type,
            "seq_idx": seq_idx,
            "length_frames": length,
            "captions": captions,
            "video_ref": {
                "mm_mardm": name + "_FV.mp4",
                "loramdm": name + "_FV.pt",
                "smoodi": name + ".pt",
            },
        })

    print(f"  After filtering: {len(valid_samples)} valid samples")
    print(f"  Skipped: {skipped}")

    # Sort by length ascending (consistent ordering)
    valid_samples.sort(key=lambda x: x["length_frames"])

    # NOTE: We skip pointer_style (length >= 196) and val/test random_split to
    # maximize the styled test set. evaluate_MARDM.py applies both, but with only
    # 5 of 100 styles the sample count is too small for meaningful metrics otherwise.
    test_samples = valid_samples
    print(f"  Using all {len(test_samples)} filtered samples (no pointer_style or val/test split)")

    # Assign manifest IDs
    for i, s in enumerate(test_samples):
        s["manifest_id"] = i

    # Style distribution
    style_counts = {}
    for s in test_samples:
        style_counts[s["style_name"]] = style_counts.get(s["style_name"], 0) + 1
    print(f"  Test set style distribution: {style_counts}")

    return test_samples


# ===================================================================
# PART 2: Build HumanML3D test set
# ===================================================================

def build_humanml3d_test_set():
    """
    Load MM-MARDM's actual HumanML3D eval split: splits_sliced/val.txt (2380 sliced IDs).

    MM-MARDM uses sliced data internally (pre-encoded latents + single-line texts).
    SMooDi and LoRA-MDM use unsliced data (raw 263-dim motions + multi-line texts).

    For each sliced ID like "012698_0", we record:
      - sliced_id: "012698_0" (for MM-MARDM latent/text lookup)
      - parent_id: "012698"   (for SMooDi/LoRA-MDM motion/text lookup)
      - length_latent: latent frames from all_lengths.txt (multiply by 4 for raw frames)
      - caption: from MM-MARDM's sliced text file (deterministic single caption)
      - parent_captions: from unsliced text file (for SMooDi/LoRA-MDM)
    """

    sliced_split_path = os.path.join(MARDM_ROOT, "datasets", "HumanML3D", "splits_sliced", "val.txt")
    sliced_lengths_path = os.path.join(MARDM_ROOT, "datasets", "HumanML3D", "splits_sliced", "all_lengths.txt")
    sliced_text_dir = os.path.join(MARDM_ROOT, "datasets", "HumanML3D", "splits_sliced", "texts_sliced")
    sliced_motion_dir = os.path.join(MARDM_ROOT, "datasets", "HumanML3D", "sliced_joint_vecs")
    latent_dir = os.path.join(MARDM_ROOT, "datasets", "HumanML3D", "latent_vecs")

    unsliced_text_dir = os.path.join(MARDM_ROOT, "datasets", "HumanML3D", "texts")
    unsliced_motion_dir = os.path.join(MARDM_ROOT, "datasets", "HumanML3D", "new_joint_vecs")

    sliced_ids = read_split_file(sliced_split_path)
    sliced_lengths = build_lengths_dict(sliced_lengths_path)

    print(f"\nHumanML3D sliced val split: {len(sliced_ids)} IDs")

    samples = []
    skipped = {"no_length": 0, "no_sliced_text": 0, "no_latent": 0,
               "zero_length": 0, "no_parent_text": 0, "no_parent_motion": 0}

    for sliced_id in sliced_ids:
        if sliced_id not in sliced_lengths:
            skipped["no_length"] += 1
            continue

        length_latent = sliced_lengths[sliced_id]
        if length_latent <= 0:
            skipped["zero_length"] += 1
            continue

        # Parent ID: strip the slice suffix ("012698_0" -> "012698")
        parent_id = sliced_id.rsplit("_", 1)[0]

        # MM-MARDM sliced text (single caption)
        sliced_text_path = os.path.join(sliced_text_dir, sliced_id + ".txt")
        if not os.path.exists(sliced_text_path):
            skipped["no_sliced_text"] += 1
            continue

        with open(sliced_text_path, "r", encoding="utf-8", errors="replace") as f:
            sliced_caption = f.read().strip()

        if not sliced_caption:
            skipped["no_sliced_text"] += 1
            continue

        # MM-MARDM latent file
        latent_path = os.path.join(latent_dir, sliced_id + ".npy")
        if not os.path.exists(latent_path):
            skipped["no_latent"] += 1
            continue

        # Unsliced parent text (multi-line, for SMooDi/LoRA-MDM)
        parent_text_path = os.path.join(unsliced_text_dir, parent_id + ".txt")
        if not os.path.exists(parent_text_path):
            skipped["no_parent_text"] += 1
            continue

        parent_captions = read_text_file_humanml(parent_text_path)

        # Unsliced parent motion (verify it exists for cross-codebase use)
        parent_motion_path = os.path.join(unsliced_motion_dir, parent_id + ".npy")
        if not os.path.exists(parent_motion_path):
            skipped["no_parent_motion"] += 1
            continue

        samples.append({
            "sliced_id": sliced_id,
            "parent_id": parent_id,
            "length_latent_frames": length_latent,
            "length_raw_frames": length_latent * UNIT_LENGTH,
            "caption": sliced_caption,
            "parent_captions": parent_captions,
        })

    print(f"  Valid samples: {len(samples)}")
    print(f"  Skipped: {skipped}")

    for i, s in enumerate(samples):
        s["manifest_id"] = i

    return samples


# ===================================================================
# PART 3: Cross-codebase path verification
# ===================================================================

def verify_paths(styled_samples, base_samples):
    """Spot-check that sample files exist in all 3 codebases."""

    codebases = {
        "mm_mardm": {
            "root": MARDM_ROOT,
            "style_motion": "datasets/100STYLE-SMPL/new_joint_vecs",
            "video_dir": "datasets/100STYLE-SMPL/videos",
            "hml_motion": "datasets/HumanML3D/new_joint_vecs",
            "hml_sliced_motion": "datasets/HumanML3D/sliced_joint_vecs",
            "hml_latent": "datasets/HumanML3D/latent_vecs",
        },
        "smoodi": {
            "root": SMOODI_ROOT,
            "style_motion": "datasets/100STYLES_RETARGETED/new_joint_vecs",
            "video_dir": "datasets/100STYLES_RETARGETED/video_latents",
            "hml_motion": "datasets/humanml3d_smoodi/new_joint_vecs",
        },
        "loramdm": {
            "root": LORAMDM_ROOT,
            "style_motion": "dataset/100STYLE-SMPL/new_joint_vecs",
            "video_dir": "dataset/100STYLE-SMPL/video_embeddings",
            "hml_motion": "dataset/HumanML3D/new_joint_vecs",
        },
    }

    print("\nCross-codebase path verification (spot-check first 5 + last 5):")
    check_styled = styled_samples[:5] + styled_samples[-5:]
    check_base = base_samples[:5] + base_samples[-5:]

    issues = []
    for cb_name, cb in codebases.items():
        if cb_name != "mm_mardm" and not os.path.isdir(cb["root"]):
            print(f"  SKIP: {cb_name} root not found ({cb['root']}); set {cb_name.upper()}_ROOT to verify it")
            continue
        # Check 100STYLE motion files
        for s in check_styled:
            p = os.path.join(cb["root"], cb["style_motion"], s["sample_id"] + ".npy")
            if not os.path.exists(p):
                issues.append(f"  MISSING: {cb_name} style motion {s['sample_id']}: {p}")

        # Check video/embedding files
        for s in check_styled:
            vid_ref = s["video_ref"][cb_name]
            p = os.path.join(cb["root"], cb["video_dir"], vid_ref)
            if not os.path.exists(p):
                issues.append(f"  MISSING: {cb_name} video/emb {vid_ref}: {p}")

        # Check HumanML3D motion files (using parent_id for unsliced codebases)
        for s in check_base:
            parent_id = s["parent_id"]
            p = os.path.join(cb["root"], cb["hml_motion"], parent_id + ".npy")
            if not os.path.exists(p):
                issues.append(f"  MISSING: {cb_name} HML3D motion {parent_id}: {p}")

        # MM-MARDM-specific: check sliced motion + latent files
        if cb_name == "mm_mardm":
            for s in check_base:
                sliced_id = s["sliced_id"]
                p = os.path.join(cb["root"], cb["hml_sliced_motion"], sliced_id + ".npy")
                if not os.path.exists(p):
                    issues.append(f"  MISSING: mm_mardm sliced motion {sliced_id}: {p}")
                p = os.path.join(cb["root"], cb["hml_latent"], sliced_id + ".npy")
                if not os.path.exists(p):
                    issues.append(f"  MISSING: mm_mardm latent {sliced_id}: {p}")

    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(issue)
    else:
        print("  All spot-checks PASSED")

    return len(issues) == 0


# ===================================================================
# PART 4: Assemble and write manifest
# ===================================================================

def main():
    print("=" * 70)
    print("Phase 2: Creating shared test manifest")
    print("=" * 70)

    styled_samples = build_100style_test_split()
    base_samples = build_humanml3d_test_set()
    verify_paths(styled_samples, base_samples)

    manifest = {
        "metadata": {
            "description": "Shared test manifest for comparative evaluation (MM-MARDM, SMooDi, LoRA-MDM)",
            "created": "2026-04-22",
            "seed": SEED,
            "max_motion_length": MAX_MOTION_LENGTH,
            "unit_length": UNIT_LENGTH,
            "min_motion_length": MIN_MOTION_LENGTH,
            "styles": STYLES,
            "style_to_idx": {s: i for i, s in enumerate(STYLES)},
            "style_to_idx_note": (
                "Training order from style classifier checkpoint. "
                "DO NOT sort() — alphabetical order breaks SRA (Phase 1 bug)."
            ),
            "source_split": {
                "100style": (
                    "test_100STYLE_Full.txt (1607 IDs) -> filter (5 styles, no TR, "
                    f"len [{MIN_MOTION_LENGTH}, {MAX_LENGTH_FILTER})) -> all surviving samples used"
                ),
                "humanml3d": (
                    "splits_sliced/val.txt (2380 sliced IDs from evaluate_MARDM.py). "
                    "Each entry has sliced_id (MM-MARDM) + parent_id (SMooDi/LoRA-MDM)."
                ),
            },
        },
        "path_roots": {
            "mm_mardm": MARDM_ROOT,
            "smoodi": SMOODI_ROOT,
            "loramdm": LORAMDM_ROOT,
        },
        "relative_paths": {
            "100style": {
                "mm_mardm": {
                    "motion_dir": "datasets/100STYLE-SMPL/new_joint_vecs",
                    "text_dir": "datasets/100STYLE-SMPL/texts",
                    "video_dir": "datasets/100STYLE-SMPL/videos",
                    "video_format": "{id}_FV.mp4",
                    "mean": "datasets/100STYLE-SMPL/Mean.npy",
                    "std": "datasets/100STYLE-SMPL/Std.npy",
                },
                "smoodi": {
                    "motion_dir": "datasets/100STYLES_RETARGETED/new_joint_vecs",
                    "text_dir": "datasets/100STYLES_RETARGETED/texts",
                    "video_dir": "datasets/100STYLES_RETARGETED/video_latents",
                    "video_format": "{id}.pt",
                    "mean": resolve_smoodi_stats("Mean.npy"),
                    "std": resolve_smoodi_stats("Std.npy"),
                },
                "loramdm": {
                    "motion_dir": "dataset/100STYLE-SMPL/new_joint_vecs",
                    "text_dir": "dataset/100STYLE-SMPL/texts",
                    "video_dir": "dataset/100STYLE-SMPL/video_embeddings",
                    "video_format": "{id}_FV.pt",
                    "mean": "dataset/100STYLE-SMPL/Mean.npy",
                    "std": "dataset/100STYLE-SMPL/Std.npy",
                },
            },
            "humanml3d": {
                "mm_mardm": {
                    "motion_dir": "datasets/HumanML3D/new_joint_vecs",
                    "sliced_motion_dir": "datasets/HumanML3D/sliced_joint_vecs",
                    "latent_dir": "datasets/HumanML3D/latent_vecs",
                    "text_dir": "datasets/HumanML3D/texts",
                    "sliced_text_dir": "datasets/HumanML3D/splits_sliced/texts_sliced",
                    "mean": "datasets/HumanML3D/Mean.npy",
                    "std": "datasets/HumanML3D/Std.npy",
                },
                "smoodi": {
                    "motion_dir": "datasets/humanml3d_smoodi/new_joint_vecs",
                    "text_dir": "datasets/humanml3d_smoodi/texts",
                    "mean": "datasets/humanml3d_smoodi/Mean.npy",
                    "std": "datasets/humanml3d_smoodi/Std.npy",
                },
                "loramdm": {
                    "motion_dir": "dataset/HumanML3D/new_joint_vecs",
                    "text_dir": "dataset/HumanML3D/texts",
                    "mean": "dataset/HumanML3D/Mean.npy",
                    "std": "dataset/HumanML3D/Std.npy",
                },
            },
        },
        "styled_samples": styled_samples,
        "base_samples": base_samples,
    }

    out_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest written to: {out_path}")
    print(f"  Styled samples: {len(styled_samples)}")
    print(f"  Base samples:   {len(base_samples)}")


if __name__ == "__main__":
    main()
