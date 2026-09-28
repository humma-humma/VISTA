#!/usr/bin/env bash
# Regression check against the thesis numbers (comparative_eval/reference/thesis_results_main.json).
# Needs a large GPU (A100: ~1-2 h; an 8 GB card takes days for the base set). Run from the repository root.
#   REF_EXPORT=/path/to/thesis/mardm_3way_additive_5styles   optional: also compare raw exports sample by sample
#   SKIP_BASE=1                                              optional: styled + transfer only (base: 2,380 samples)
set -euo pipefail

# Manifest + GT (identical sample selection to the thesis export; verified locally).
[[ -f comparative_eval/manifest.json ]] || {
  python comparative_eval/create_test_manifest.py
  python comparative_eval/add_transfer_samples.py
}
python comparative_eval/prepare_gt.py --manifest comparative_eval/manifest.json --resume

# Same settings as the thesis export (reference/thesis_export_3way_run_log.json):
# 3way_additive, text 4.5 / style 2.0, 18 steps, seed 3407, EMA weights, weight schedule ON.
SKIP=(); [[ "${SKIP_BASE:-0}" == 1 ]] && SKIP=(--skip_base)
python export_for_comparison.py --manifest comparative_eval/manifest.json --model MARDM-DDPM-XL --ae_name DAE \
  --cfg_mode 3way_additive --cfg_text 4.5 --cfg_style 2.0 --timesteps 18 --seed 3407 \
  --use_weight_schedule --output_dir comparative_eval "${SKIP[@]}"

MODES=(styled transfer); [[ "${SKIP_BASE:-0}" == 1 ]] || MODES+=(base)
python evaluate_comparative.py --manifest comparative_eval/manifest.json --comp_root comparative_eval \
  --models mardm_3way_t4.5_s2.0 --eval_modes "${MODES[@]}" --output comparative_eval/results_regression.json

if [[ -n "${REF_EXPORT:-}" ]]; then
  python tools/compare_exports.py exports --new comparative_eval/mardm_3way_t4.5_s2.0 --ref "$REF_EXPORT" --modes "${MODES[@]}"
fi
python tools/compare_exports.py metrics --new comparative_eval/results_regression.json \
  --model mardm_3way_t4.5_s2.0 --ref_model mardm_3way_additive
