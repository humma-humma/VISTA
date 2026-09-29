#!/usr/bin/env bash
# Preprocessing for VISTA training/evaluation. Run from the repository root.
# Prerequisite: the VE-100STYLES dataset (videos + length index), built with the separate
# ve100styles/ sub-repository in its own environment - see ve100styles/README.md:
#   python ve100styles/build_length_index.py --data_root datasets/100STYLE-SMPL
#   python ve100styles/render_dataset.py --data_root datasets/100STYLE-SMPL \
#       --styles Aeroplane ArmsFolded Chicken Robot Superman --views front left
set -euo pipefail

# 1) Checkpoints: VISTA + evaluators + GloVe (Google Drive, sha256-verified), then the MARDM base model,
#    HumanML3D AE and length estimator from the original MARDM release.
python prepare/download_vista_checkpoints.py
python prepare/download_pretrained.py --skip_sit
bash prepare/download_smpl.sh     # SMPL body model + SMPLify priors (mesh export / viewers)

# 2) HumanML3D: slice by caption time tags, encode with the pretrained HumanML3D AE.
#    Writes sliced_joint_vecs/, latent_vecs/, splits_sliced/{train,val,test,all_lengths}.txt,
#    splits_sliced/texts_sliced/.
python preprocess/hml3d_encoder.py
