#!/usr/bin/env bash
# Stage 2: VISTA diffusion fine-tuning (thesis Sec. 4.3-4.4 / 5.1.2). Run from the repository root.
# Final thesis configuration = late fusion (DiffMLP routing) + hybrid cross-batch objective (Table 5.4).
#   --is_continue            initialise from pretrained HumanML3D MARDM (checkpoints/t2m/MARDM-DDPM-XL/model/humanml3d_latest.tar)
#   --freeze_mode differential --mardm_lr_mult 0.1   new modules 1e-4, MARTransformer 1e-5, content DiffMLP ramped
#   --enable_cfg_dropout     5% both / 10% text / 10% style dropout
#   --use_weight_schedule    block-wise w: 0 -> 1 across the 24 DiffMLP blocks (default ON)
#   --enable_cross_batch --cross_batch_mode hybrid   Pass 3 (p=0.5, weight 0.1; the thesis run used --cross_batch_prob 0.4)
# Batch size: the thesis text says 64, but the released checkpoint shows 16 iterations/epoch
# (199,500 - 191,500 it over 500 epochs) = 528 training clips // 32. Use 32 to reproduce it.
set -euo pipefail

DAE_CKPT=${DAE_CKPT:-checkpoints/100styles/DAE/epoch_119_detach_nostyle_disc.tar}   # Stage-1 output

python train_MARDM.py \
  --name VISTA --model MARDM-DDPM-XL --dataset_name t2m \
  --ae_name DAE --ae_model DAE_Model --dae_ckpt "$DAE_CKPT" \
  --styles Aeroplane Chicken Robot Superman ArmsFolded --video_encoder vivit \
  --data_mode v4 --style_routing diffmlp --freeze_mode differential \
  --epoch 500 --batch_size 32 --lr 1e-4 --mardm_lr_mult 0.1 \
  --enable_cfg_dropout --use_weight_schedule \
  --enable_cross_batch --cross_batch_mode hybrid --cross_batch_prob 0.5 --cross_batch_weight 0.1 \
  --is_continue --seed 3407 --exp_name vista_stage2
# Output: checkpoints/t2m/VISTA/model/{final.tar, dae_final.tar, net_best_fid_styled.tar, epoch_*.tar}
#   final.tar      -> Stage-2 MARDM (use key 'ema_mardm' at inference)
#   dae_final.tar  -> DualAE with the jointly fine-tuned decoder
# Released equivalents: final_diffmlp_diff_v4_cfg_cross_batch_hybrid_ema_fix.tar /
#                       final_finetune_diffmlp_decoder_fixed_hybrid_ema_fix.tar
