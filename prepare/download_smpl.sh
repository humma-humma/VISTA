#!/usr/bin/env bash
# SMPL assets for rendering / SMPLify fitting (same sources as MDM). Run from the repository root.
# Existing files are never deleted or overwritten. By downloading SMPL you accept its license
# (https://smpl.is.tue.mpg.de).
set -euo pipefail

# 1) body_models/smpl/{SMPL_NEUTRAL.pkl, J_regressor_extra.npy, kintree_table.pkl, smplfaces.npy}
if [[ -f body_models/smpl/SMPL_NEUTRAL.pkl ]]; then
  echo "body_models/smpl already present - skipping"
else
  mkdir -p body_models
  gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2" -O body_models/smpl.zip
  unzip -n body_models/smpl.zip -d body_models
  rm body_models/smpl.zip
fi

# 2) SMPLify priors used by visualize/joints2smpl (GMM pose prior, mean params, ...)
DST=visualize/joints2smpl/smpl_models
BASE=https://github.com/GuyTevet/motion-diffusion-model/raw/main/visualize/joints2smpl/smpl_models
mkdir -p "$DST"
for f in gmm_08.pkl neutral_smpl_mean_params.h5 SMPL_downsample_index.pkl smplx_parts_segm.pkl; do
  [[ -f "$DST/$f" ]] || curl -L --fail -o "$DST/$f" "$BASE/$f"
done
echo "SMPL assets ready."
