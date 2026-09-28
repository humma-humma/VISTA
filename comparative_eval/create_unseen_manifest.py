"""
create_unseen_manifest.py
-------------------------
Build manifest for 5 UNSEEN styles evaluation (MM-MARDM only).

Styles: Cat (5), HandsBetweenLegs (7), BigSteps (11), LegsApart (14), CrowdAvoidance (15)
These were NOT used in MARDM training (which used Aeroplane/Chicken/Robot/Superman/ArmsFolded)
but ARE recognized by the 20-class style classifier checkpoint.

Produces: manifest_unseen.json with:
  - styled_samples: all valid test-split samples for the 5 unseen styles
  - transfer_samples: 10 per style = 50 (HumanML3D text + unseen style ref)
  - No base_samples (base eval is style-independent, already covered by main manifest)

Usage:
    python create_unseen_manifest.py
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict

try:
    import chardet
except ImportError:
    chardet = None

# ===================================================================
# Configuration
# ===================================================================
SEED = 3407
MAX_MOTION_LENGTH = 196
MIN_MOTION_LENGTH = 40
MAX_LENGTH_FILTER = 400
UNIT_LENGTH = 4

UNSEEN_STYLES = ["Cat", "HandsBetweenLegs", "BigSteps", "LegsApart", "CrowdAvoidance"]

# Indices from the 20-class style classifier checkpoint (training order)
CLASSIFIER_STYLE_TO_IDX = {
    "Cat": 5,
    "HandsBetweenLegs": 7,
    "BigSteps": 11,
    "LegsApart": 14,
    "CrowdAvoidance": 15,
}

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
# MARDM_ROOT = os.path.join(ROOT, "MM_MARDM")
MARDM_ROOT = os.environ.get("VISTA_ROOT", ROOT)
OUTPUT_DIR = os.path.dirname(__file__)
ORIGINAL_MANIFEST = os.path.join(OUTPUT_DIR, "manifest.json")

SAMPLES_PER_STYLE_TRANSFER = 10


# ===================================================================
# Helpers (same as create_test_manifest.py)
# ===================================================================

def build_dict_from_txt(filename):
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
                result[key] = (style_name, motion_type, seq_idx, length)
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


def snap_to_multiple(n, unit, cap):
    capped = min(n, cap)
    return (capped // unit) * unit


# ===================================================================
# PART 1: Build unseen-style styled samples
# ===================================================================

def build_unseen_styled_samples():
    name_dict_path = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "100STYLE_name_dict_length.txt")
    split_path = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "test_100STYLE_Full.txt")
    motion_dir = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "new_joint_vecs")
    video_dir = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "videos_OOD")
    text_dir = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "texts")

    metadata = build_dict_from_txt(name_dict_path)
    split_ids = read_split_file(split_path)

    print(f"100STYLE test split: {len(split_ids)} IDs")

    valid_samples = []
    skipped = {"style": 0, "tr": 0, "length": 0, "missing_file": 0, "metadata": 0, "no_text": 0}

    for name in split_ids:
        if name not in metadata:
            skipped["metadata"] += 1
            continue

        style_name, motion_type, seq_idx, length = metadata[name]

        if style_name not in UNSEEN_STYLES:
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

        valid_samples.append({
            "sample_id": name,
            "style_name": style_name,
            "style_idx": CLASSIFIER_STYLE_TO_IDX[style_name],
            "motion_type": motion_type,
            "seq_idx": seq_idx,
            "length_frames": length,
            "captions": captions,
            "video_ref": {
                "mm_mardm": name + "_FV.mp4",
            },
        })

    print(f"  After filtering: {len(valid_samples)} valid samples")
    print(f"  Skipped: {skipped}")

    valid_samples.sort(key=lambda x: x["length_frames"])

    for i, s in enumerate(valid_samples):
        s["manifest_id"] = i

    style_counts = {}
    for s in valid_samples:
        style_counts[s["style_name"]] = style_counts.get(s["style_name"], 0) + 1
    print(f"  Style distribution: {style_counts}")

    return valid_samples


# ===================================================================
# PART 2: Build transfer samples (HumanML3D text + unseen style ref)
# ===================================================================

def build_transfer_samples(styled_samples):
    with open(ORIGINAL_MANIFEST, "r", encoding="utf-8") as f:
        original = json.load(f)
    base_samples = original["base_samples"]

    styled_by_style = defaultdict(list)
    for s in styled_samples:
        styled_by_style[s["style_name"]].append(s)

    rng = random.Random(SEED)
    all_base_indices = list(range(len(base_samples)))
    rng.shuffle(all_base_indices)

    next_id = len(styled_samples)

    transfer_samples = []
    used_base_indices = set()
    cursor = 0

    for style_name in UNSEEN_STYLES:
        style_idx = CLASSIFIER_STYLE_TO_IDX[style_name]
        style_refs = styled_by_style[style_name]
        if not style_refs:
            print(f"  WARNING: No styled samples for {style_name}, skipping transfer")
            continue
        count = 0

        while count < SAMPLES_PER_STYLE_TRANSFER:
            if cursor >= len(all_base_indices):
                raise RuntimeError("Ran out of base samples")
            idx = all_base_indices[cursor]
            cursor += 1

            if idx in used_base_indices:
                continue
            used_base_indices.add(idx)

            base = base_samples[idx]
            style_ref = rng.choice(style_refs)

            length_frames = snap_to_multiple(
                base["length_raw_frames"], UNIT_LENGTH, MAX_MOTION_LENGTH
            )

            entry = {
                "manifest_id": next_id,
                "text": base["caption"],
                "length_frames": length_frames,
                "style_name": style_name,
                "style_idx": style_idx,
                "style_ref_sample_id": style_ref["sample_id"],
                "video_ref": {
                    "mm_mardm": style_ref["video_ref"]["mm_mardm"],
                },
                "source_base_sample": {
                    "sliced_id": base["sliced_id"],
                    "parent_id": base["parent_id"],
                },
            }

            transfer_samples.append(entry)
            next_id += 1
            count += 1

    return transfer_samples


# ===================================================================
# PART 3: Verify MM-MARDM paths
# ===================================================================

def verify_paths(styled_samples):
    motion_dir = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "new_joint_vecs")
    video_dir = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "videos_OOD")
    text_dir = os.path.join(MARDM_ROOT, "datasets", "100STYLE-SMPL", "texts")

    print("\nPath verification (all styled samples):")
    issues = []
    for s in styled_samples:
        sid = s["sample_id"]
        for path, label in [
            (os.path.join(motion_dir, sid + ".npy"), "motion"),
            (os.path.join(video_dir, sid + "_FV.mp4"), "video"),
            (os.path.join(text_dir, sid + ".txt"), "text"),
        ]:
            if not os.path.exists(path):
                issues.append(f"  MISSING {label}: {path}")

    if issues:
        print(f"  {len(issues)} ISSUES:")
        for issue in issues[:10]:
            print(issue)
    else:
        print(f"  All {len(styled_samples)} samples verified OK")

    return len(issues) == 0


# ===================================================================
# PART 4: Assemble and write manifest
# ===================================================================

def main():
    print("=" * 70)
    print("Creating unseen-styles manifest (MM-MARDM only)")
    print(f"Styles: {UNSEEN_STYLES}")
    print(f"Classifier indices: {CLASSIFIER_STYLE_TO_IDX}")
    print("=" * 70)

    styled_samples = build_unseen_styled_samples()
    verify_paths(styled_samples)
    transfer_samples = build_transfer_samples(styled_samples)

    manifest = {
        "metadata": {
            "description": "Unseen-styles evaluation manifest (MM-MARDM only). "
                           "These 5 styles were NOT in MARDM training but ARE in the 20-class classifier.",
            "created": "2026-04-26",
            "seed": SEED,
            "max_motion_length": MAX_MOTION_LENGTH,
            "unit_length": UNIT_LENGTH,
            "min_motion_length": MIN_MOTION_LENGTH,
            "styles": UNSEEN_STYLES,
            "style_to_idx": CLASSIFIER_STYLE_TO_IDX,
            "style_to_idx_note": (
                "Indices match the 20-class style classifier checkpoint (training order). "
                "NOT 0-4 — these are the actual classifier output indices."
            ),
        },
        "path_roots": {
            "mm_mardm": MARDM_ROOT,
        },
        "relative_paths": {
            "100style": {
                "mm_mardm": {
                    "motion_dir": "datasets/100STYLE-SMPL/new_joint_vecs",
                    "text_dir": "datasets/100STYLE-SMPL/texts",
                    "video_dir": "datasets/100STYLE-SMPL/videos_OOD",
                    "video_format": "{id}_FV.mp4",
                    "mean": "datasets/100STYLE-SMPL/Mean.npy",
                    "std": "datasets/100STYLE-SMPL/Std.npy",
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
            },
        },
        "styled_samples": styled_samples,
        "base_samples": [],
        "transfer_samples": transfer_samples,
    }

    out_path = os.path.join(OUTPUT_DIR, "manifest_unseen.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest written to: {out_path}")
    print(f"  Styled samples:   {len(styled_samples)}")
    print(f"  Transfer samples: {len(transfer_samples)}")
    print(f"  Base samples:     0 (use main manifest for base eval)")

    print("\nTransfer breakdown by style:")
    from collections import Counter
    tc = Counter(s["style_name"] for s in transfer_samples)
    for style in UNSEEN_STYLES:
        print(f"  {style:20s}: {tc.get(style, 0)}")

    print("\nSample styled entry:")
    print(json.dumps(styled_samples[0], indent=2))
    print("\nSample transfer entry:")
    print(json.dumps(transfer_samples[0], indent=2))


if __name__ == "__main__":
    main()
