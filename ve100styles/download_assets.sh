#!/usr/bin/env bash
# SMPL body model + SMPLify priors for the renderer (same sources as MDM).
# Installs into ./assets next to this script (or $VE100STYLES_ASSETS). Nothing is deleted or overwritten.
# By downloading SMPL you accept its license (https://smpl.is.tue.mpg.de).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS="${VE100STYLES_ASSETS:-$HERE/assets}"
mkdir -p "$ASSETS/body_models" "$ASSETS/smplify"

if [[ -f "$ASSETS/body_models/smpl/SMPL_NEUTRAL.pkl" ]]; then
  echo "SMPL already present - skipping"
else
  gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2" -O "$ASSETS/body_models/smpl.zip"
  unzip -n "$ASSETS/body_models/smpl.zip" -d "$ASSETS/body_models"
  rm "$ASSETS/body_models/smpl.zip"
fi

BASE=https://github.com/GuyTevet/motion-diffusion-model/raw/main/visualize/joints2smpl/smpl_models
for f in gmm_08.pkl neutral_smpl_mean_params.h5 SMPL_downsample_index.pkl smplx_parts_segm.pkl; do
  [[ -f "$ASSETS/smplify/$f" ]] || curl -L --fail -o "$ASSETS/smplify/$f" "$BASE/$f"
done
echo "Assets ready in $ASSETS"
