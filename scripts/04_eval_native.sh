#!/usr/bin/env bash
# Native three-mode evaluation (base / styled / transfer) directly from the dataloaders.
# Run from the repository root. Override checkpoints with MARDM_CKPT / DAE_CKPT for your own runs, e.g.
#   MARDM_CKPT=checkpoints/t2m/VISTA/model/final.tar DAE_CKPT=checkpoints/t2m/VISTA/model/dae_final.tar
set -euo pipefail

EXTRA=()
[[ -n "${MARDM_CKPT:-}" ]] && EXTRA+=(--mardm_ckpt "$MARDM_CKPT")
[[ -n "${DAE_CKPT:-}"   ]] && EXTRA+=(--dae_ckpt "$DAE_CKPT")

COMMON=(--model MARDM-DDPM-XL --ae_name DAE --ae_model DAE_Model --checkpoint_key ema_mardm
        --style_routing diffmlp --use_weight_schedule --eval_mode full --timesteps 18 --seed 3407)

# VISTA-2way: joint guidance s = 4.5
python evaluate_MARDM.py "${COMMON[@]}" "${EXTRA[@]}" --cfg_mode 2way --cfg_scale 4.5

# VISTA-3way: s_text = 4.5, s_style = 2.0 (thesis Sec. 5.3)
python evaluate_MARDM.py "${COMMON[@]}" "${EXTRA[@]}" --cfg_mode 3way_additive --cfg_text 4.5 --cfg_style 2.0
