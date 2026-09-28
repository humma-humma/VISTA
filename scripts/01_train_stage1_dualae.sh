#!/usr/bin/env bash
# Stage 1: DualAE pre-training on VE-100STYLES (thesis Sec. 4.2 / 5.1.2). Run from the repository root.
# Phase 1 (reconstruction + L_vel + L_embed + latent style CE) from epoch 0,
# Phase 2 (adversarial, 2-layer discriminator) from epoch 30, Phase 3 (latent cycle) from epoch 70.
# The released DualAE checkpoint (epoch_119_detach_nostyle_disc.tar) stores ep=120.
set -euo pipefail

python train_AE_adv.py \
  --dataset_name 100styles --name DAE --model DAE_Model \
  --style_classes Aeroplane Chicken Robot Superman ArmsFolded \
  --video_encoder vivit --window_size 64 --snippets_per_sequence 15 \
  --epoch 120 --batch_size 16 --lr 2e-4 --warm_up_iter 2000 \
  --disc_start_epoch 30 --cycle_start_epoch 70 \
  --seed 3407 --exp_name vista_stage1_dualae
# Output: checkpoints/100styles/DAE/model/{epoch_*.tar,final.tar}

# Stage-1 evaluation (Table 5.2 metrics: FID / RR-MPJPE / Global MPJPE, motion vs. video reconstruction):
# python evaluate_AE.py --dataset_name 100styles --name DAE --model DAE_Model \
#   --style_classes Aeroplane Chicken Robot Superman ArmsFolded --video_encoder vivit
