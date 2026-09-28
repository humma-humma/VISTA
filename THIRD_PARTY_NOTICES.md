# Third-Party Notices

VISTA includes third-party code and code adapted from third-party projects. Those components retain their
original copyright notices and license terms. Check the upstream repositories for the authoritative
license text.

## MARDM

Source: https://github.com/neu-vi/MARDM (MIT License; Copyright (c) Meta Platforms, Inc. and affiliates,
as stated in the MARDM license file)

Affected paths include:

- `models/MARDM.py`, `models/DiffMLPs.py`, `models/AE.py` (motion autoencoder), `models/LengthEstimator.py`
- `train_*.py`, `evaluate_*.py`, `sample_new.py`, `edit.py` (training / evaluation / sampling scaffolding)
- `utils/train_utils.py`, `utils/datasets.py`, `utils/eval_utils.py`, `utils/evaluators.py`
- `prepare/download_pretrained.py` (download links of the MARDM release)

## Diffusion and flow-matching backends (OpenAI diffusion code, DiT, SiT)

Sources: https://github.com/openai/guided-diffusion, https://github.com/openai/glide-text2im,
https://github.com/facebookresearch/DiT, https://github.com/willisma/SiT

Affected paths:

- `diffusions/diffusion/` (modified from OpenAI's diffusion code as adopted in DiT)
- `diffusions/transport/` (flow-matching / SiT backend)

## Text-to-motion evaluators and motion representation (HumanML3D / T2M)

Sources: https://github.com/EricGuo5513/text-to-motion, https://github.com/EricGuo5513/HumanML3D

Affected paths:

- `utils/evaluators.py`, `utils/eval_utils.py`, `utils/glove.py`, `utils/motion_process.py`
- `utils/eval_mean_std/` (evaluator normalisation statistics)

## MDM and ACTOR (SMPL layer, rotation utilities, joints-to-SMPL fitting)

Sources: https://github.com/GuyTevet/motion-diffusion-model (MIT License),
https://github.com/Mathux/ACTOR

Affected paths:

- `visualize/smpl.py`, `visualize/rotation2xyz.py`, `visualize/rotation_conversions.py`,
  `visualize/vis_utils.py`, `visualize/render_mesh.py`, `visualize/simplify_loc2rot.py`,
  `visualize/motions2hik.py`
- `ve100styles/vestyles/rotation_conversions.py`, `ve100styles/vestyles/smplify/joints2smpl.py`

`rotation_conversions.py` carries the notice "Copyright (c) Facebook, Inc. and its affiliates. All rights
reserved." (PyTorch3D).

## SMPLify / joints2smpl

Sources: https://github.com/wangsen1312/joints2smpl, https://smplify.is.tue.mpg.de

Affected paths:

- `visualize/joints2smpl/` (`fit_seq.py`, `src/config.py`, `src/customloss.py`, `src/prior.py`,
  `src/smplify.py`)
- `ve100styles/vestyles/smplify/` (`config.py`, `customloss.py`, `prior.py`, `smplify.py`)

`prior.py` carries the following notice from the Max Planck Institute for Intelligent Systems:

> Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
> on this computer program. You can only use this computer program if you have closed a license agreement
> with MPG or you get the right to use the computer program from someone who is authorized to grant you
> that right. Any use of the computer program without a valid license is prohibited and liable to
> prosecution. Copyright©2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG). acting
> on behalf of its Max Planck Institute for Intelligent Systems. All rights reserved.
> Contact: ps-license@tuebingen.mpg.de

## DuetGen

Source: DuetGen (global trajectory refinement)

Affected paths (optional refinement code):

- `utils/motion_smoothing.py` (adapted from `smooth_root_joint_deltas()`)
- `train_refinement.py` (loss design adapted from `GlobalTrajectoryTrainer`)

## Models and data used at run time (not included in this repository)

These are downloaded separately and are subject to their own licenses and terms of use:

- SMPL body model and SMPLify priors (https://smpl.is.tue.mpg.de, https://smplify.is.tue.mpg.de)
- CLIP (https://github.com/openai/CLIP)
- ViViT `google/vivit-b-16x2-kinetics400` (https://huggingface.co/google/vivit-b-16x2-kinetics400)
- HumanML3D (https://github.com/EricGuo5513/HumanML3D) and 100STYLE
  (https://www.ianxmason.com/100style/), and the 100STYLE SMPL retargeting released with SMooDi
  (https://github.com/neu-vi/SMooDi)
- Pretrained MARDM models, evaluators and GloVe vectors from the MARDM release
