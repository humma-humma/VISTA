#!/usr/bin/env bash
# Auxiliary kinematic style classifier (frozen during Stage 2; provides L_style_feat and the SRA metric).
# Settings below match the args stored in the released style_classifier_final.pt (21 styles).
set -euo pipefail

python train_style_classification.py \
  --dataset_dir ./datasets --input_dim 67 \
  --latent_dim 512 --ff_size 1024 --num_layers 6 --num_heads 4 --dropout 0.1 \
  --epochs 200 --batch_size 128 --lr 2e-4 --weight_decay 1e-5 \
  --seed 3407 --output_dir ./checkpoints/style_classifier
# Output: checkpoints/style_classifier/style_classifier_final.pt (includes style_to_idx)
