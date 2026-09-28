"""
add_transfer_samples.py
-----------------------
Adds ~50 transfer evaluation samples to the shared manifest.json.

Transfer = HumanML3D text prompt + 100STYLE style (novel combination).
  - 5 styles x 10 samples per style = 50 total
  - For each, we randomly pick a base_sample (HumanML3D) and pair it with
    a random styled_sample of the target style (as the style reference).
  - No ground-truth motion exists for these combinations.

Usage:
    python add_transfer_samples.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict

MANIFEST_PATH = Path(__file__).parent / "manifest.json"
SEED = 3407
SAMPLES_PER_STYLE = 10
MAX_MOTION_LENGTH = 196
UNIT_LENGTH = 4


def snap_to_multiple(n: int, unit: int, cap: int) -> int:
    """Cap at `cap`, then snap down to nearest multiple of `unit`."""
    capped = min(n, cap)
    return (capped // unit) * unit


def main():
    # ---- Load manifest ----
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    style_to_idx = manifest["metadata"]["style_to_idx"]
    styles = manifest["metadata"]["styles"]  # ordered list
    styled_samples = manifest["styled_samples"]
    base_samples = manifest["base_samples"]

    # ---- Determine next manifest_id ----
    max_styled = max(s["manifest_id"] for s in styled_samples)
    max_base = max(s["manifest_id"] for s in base_samples)
    next_id = max(max_styled, max_base) + 1
    print(f"Highest styled manifest_id : {max_styled}")
    print(f"Highest base manifest_id   : {max_base}")
    print(f"Transfer samples start at  : {next_id}")
    print()

    # ---- Group styled_samples by style for quick lookup ----
    styled_by_style: dict[str, list] = defaultdict(list)
    for s in styled_samples:
        styled_by_style[s["style_name"]].append(s)

    # ---- Seed RNG ----
    rng = random.Random(SEED)

    # ---- Select 10 random base samples per style ----
    # We draw 50 unique base samples total (10 per style, no repeats across styles)
    all_base_indices = list(range(len(base_samples)))
    rng.shuffle(all_base_indices)

    transfer_samples = []
    used_base_indices = set()
    cursor = 0  # position in shuffled list

    for style_name in styles:
        style_idx = style_to_idx[style_name]
        style_refs = styled_by_style[style_name]
        count = 0

        while count < SAMPLES_PER_STYLE:
            if cursor >= len(all_base_indices):
                raise RuntimeError("Ran out of base samples (should not happen with 2380 available)")
            idx = all_base_indices[cursor]
            cursor += 1

            if idx in used_base_indices:
                continue
            used_base_indices.add(idx)

            base = base_samples[idx]
            # Pick a random style reference from styled_samples of this style
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
                    "loramdm": style_ref["video_ref"]["loramdm"],
                    "smoodi": style_ref["video_ref"]["smoodi"],
                },
                "source_base_sample": {
                    "sliced_id": base["sliced_id"],
                    "parent_id": base["parent_id"],
                },
            }

            transfer_samples.append(entry)
            next_id += 1
            count += 1

    # ---- Insert into manifest ----
    manifest["transfer_samples"] = transfer_samples

    # ---- Save ----
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print(f"Added {len(transfer_samples)} transfer samples to manifest.")
    print(f"  manifest_id range: {transfer_samples[0]['manifest_id']} - {transfer_samples[-1]['manifest_id']}")
    print()
    print("Breakdown by style:")
    from collections import Counter
    style_counts = Counter(s["style_name"] for s in transfer_samples)
    for style_name in styles:
        print(f"  {style_name:15s}: {style_counts[style_name]} samples")
    print()
    print("Sample transfer entry:")
    print(json.dumps(transfer_samples[0], indent=2))
    print()
    print(f"Manifest saved to: {MANIFEST_PATH.resolve()}")


if __name__ == "__main__":
    main()
