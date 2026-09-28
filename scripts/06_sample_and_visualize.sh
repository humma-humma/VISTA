#!/usr/bin/env bash
# Qualitative generation (50 steps) and visualisation. Run from the repository root.
set -euo pipefail

STYLE_VIDEO=${STYLE_VIDEO:-datasets/100STYLE-SMPL/videos/030001_FV.mp4}

# Text + reference-video generation, 3-way guidance.
python sample_new.py --text_prompt "a person walks forward" --style_video "$STYLE_VIDEO" \
  --use_weight_schedule --cfg_mode 3way_additive --cfg_text 4.5 --cfg_style 2.0 \
  --timesteps 50 --motion_length 120 --output_dir ./generation

# Unconditional / text-only / style-only / text+style grid:
# python sample_new.py --text_prompt "a person walks forward" --style_video "$STYLE_VIDEO" --generate_quad

# SMPL mesh export (SMPLify fit) -> .obj sequence, then Blender render:
# python tools/export_meshes.py --input generation/<run>/joints.npy --output generation/<run>/meshes
# blender --background --python tools/render_blender.py -- --input generation/<run>/meshes --output renders/

# Interactive viewer (aitviewer):
# python tools/aitviewer_interactive_render.py --help
