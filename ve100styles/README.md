# VE-100STYLES: dataset generation

Standalone sub-repository that builds the **VE-100STYLES** video modality used by VISTA: every
100STYLE sequence (HumanML3D format, 263-dim) is fitted with SMPL and rendered as a 1920x1080
video from fixed front and left cameras. It shares no code with the VISTA training / evaluation
code and has its own environment.

```
ve100styles/
├── render_dataset.py        bulk renderer (style / id filtering, resumable)
├── render_motion.py         render one .npy file (debugging a single sequence)
├── build_length_index.py    100STYLE_name_dict.txt -> 100STYLE_name_dict_length.txt
├── download_assets.sh       SMPL + SMPLify priors -> ./assets
├── requirements.txt
└── vestyles/
    ├── features.py          263-dim HumanML3D features -> joints (recover_from_ric), Mean/Std handling
    ├── renderer.py          SMPL fit + pyrender scene + MP4 encoding
    ├── rotation_conversions.py
    ├── paths.py             asset locations (relative to this directory, or $VE100STYLES_ASSETS)
    └── smplify/             SMPLify-3D joint fitting (from MDM / joints2smpl)
```

## Setup

```bash
conda create -n ve100styles python=3.10 && conda activate ve100styles
pip install torch            # pick the build for your CUDA
pip install -r ve100styles/requirements.txt
bash ve100styles/download_assets.sh          # -> ve100styles/assets/{body_models,smplify}
export PYOPENGL_PLATFORM=egl                 # headless servers (or osmesa)
```

`ffmpeg` must be on `PATH`.

## Input: MM_MARDM / VISTA dataset layout

```
datasets/100STYLE-SMPL/
├── new_joint_vecs/<id>.npy        [T, 263] HumanML3D features, RAW (not normalised), 20 FPS
├── new_joints/<id>.npy            [T, 22, 3] joints (optional alternative input)
├── Mean.npy, Std.npy              263-dim statistics (only used with --normalized)
└── 100STYLE_name_dict.txt         "<id> <Style>_<Content>_<nn>.bvh <...>"
```

Ids prefixed with `M` are mirrored sequences; they are rendered by default (the MM_MARDM video
set contains them; `--no-include_mirrored` skips them).

Features are consumed as the full 263-dim representation: 67-dim inputs are rejected, and
normalisation (when `--normalized` is set) uses the 263-dim Mean/Std. Joint positions are
recovered with HumanML3D's `recover_from_ric`, which reads the root and RIC channels (0-66) of
that vector.

## Usage

```bash
# 1) length index read by the VISTA data loaders
python ve100styles/build_length_index.py --data_root datasets/100STYLE-SMPL

# 2) check the setup on a couple of sequences first
python ve100styles/render_dataset.py --data_root datasets/100STYLE-SMPL \
    --styles Aeroplane --views front left --limit 2 --output_dir /tmp/ve100_smoke

# 3) training styles, front + left views
python ve100styles/render_dataset.py --data_root datasets/100STYLE-SMPL \
    --styles Aeroplane ArmsFolded Chicken Robot Superman --views front left

# 4) held-out styles for the unseen-style study
python ve100styles/render_dataset.py --data_root datasets/100STYLE-SMPL \
    --styles <style> [<style> ...] --views front left --output_dir datasets/100STYLE-SMPL/videos_OOD
```

Output: `<output_dir>/<id>_FV.mp4` and `<id>_LV.mp4` (default `<data_root>/videos`), the names the
VISTA loaders expect. Existing videos are skipped (`--overwrite` re-renders); videos are
written to a temporary file and renamed, so an interrupted run leaves no truncated MP4. Failed
ids are listed in `<output_dir>/failed_renders.txt`. `--dry_run` prints the work list.

Rendering settings: camera distance 10, elevation 0.0, tilt -1 deg, FoV framing factor 1.2
(vertical FoV computed from the animation's bounding box). These are the values used for the
released videos (they appear in LoRA-MDM's `data_setup.ipynb`; the old bulk script defaulted to
elevation 0.3 / framing 2.0 / tilt 0, which yields a ~1.7x smaller body). Verified on `030001`
with the default settings, re-render vs. released video (263/263 frames each):
front view PSNR 50.6 dB, body-silhouette IoU 0.992, centroid offset 0.3 px;
left view PSNR 50.9 dB, IoU 0.991, centroid offset 0.3 px. Other settings:
floor at the lowest point of the animation, neutral SMPL shape (beta = 0), 150 SMPLify
iterations. One video frame is written per motion frame, so video frame *i* matches motion
frame *i*; `--fps` only sets the container rate (the existing set uses 15).

## Differences from the original scripts

Originally `generate_video_bulk.py` / `render_smpl_vid.py`, which imported modules from the
LoRA-MDM tree. Rendering and fitting logic are unchanged. Changes:

- Self-contained: kinematics, rotation utilities and SMPLify are vendored in `vestyles/`; asset
  paths no longer depend on the working directory. SMPL faces come from `smplx` directly
  instead of MDM's `Rotation2xyz`.
- Input handling: the original always applied `x * std + mean` to 263-dim input, which is wrong
  for the raw `new_joint_vecs` (it produces a different, distorted pose). Features are now
  treated as raw unless `--normalized` is given, with a warning when the data looks like the
  other case. Checked against the released set: joints recovered from raw `new_joint_vecs`
  equal `new_joints` exactly, and match the pose in the existing `030001_FV.mp4`; the
  double-normalised variant does not. `new_joints` input works as before.
- One SMPLify fit per sequence is shared by all requested views (previously refitted per view).
- Style filtering reads `<data_root>/100STYLE_name_dict.txt` by default; the hard-coded Windows
  `pending_files.txt` resume list is replaced by skip-existing plus optional `--ids_file`.
- Output names are `<id>_<view>.mp4` directly; MP4s have no audio track.
- The low-resolution GIF previewer (`render_smpl_gif.py`) is not included.

The thesis text (Sec. 5.1.1) describes rendering at 75 FPS with subsampling to 20 FPS; the
released videos have one frame per 20 FPS motion frame, stored at 15 FPS, which is what this
tool reproduces.
