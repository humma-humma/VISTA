#!/usr/bin/env bash
# Shared-manifest evaluation used for the thesis comparison tables (Sec. 5.3-5.4).
# base: HumanML3D val (n=2,380) | styled: VE-100STYLES, 5 styles, len in [40,400) (n=72) | transfer: n=50.
# Run from the repository root.
set -euo pipefail

# 1) Manifests (seed 3407). SMOODI_ROOT / LORAMDM_ROOT are only needed to spot-check baseline paths.
python comparative_eval/create_test_manifest.py        # -> comparative_eval/manifest.json
python comparative_eval/add_transfer_samples.py        # adds 5 x 10 transfer pairs to manifest.json
python comparative_eval/create_unseen_manifest.py      # -> comparative_eval/manifest_unseen.json (Sec. 5.4.3, videos_OOD)
python comparative_eval/prepare_gt.py --manifest comparative_eval/manifest.json

# 2) Export VISTA predictions (feat67 + joints) for every manifest sample.
COMMON=(--manifest comparative_eval/manifest.json --model MARDM-DDPM-XL --ae_name DAE
        --use_weight_schedule --timesteps 18 --seed 3407)
python export_for_comparison.py "${COMMON[@]}" --cfg_mode 2way --cfg_scale 4.5                              # -> comparative_eval/mardm_2way_c4.5/
python export_for_comparison.py "${COMMON[@]}" --cfg_mode 3way_additive --cfg_text 4.5 --cfg_style 2.0      # -> comparative_eval/mardm_3way_t4.5_s2.0/

# Guidance-scale sweep (Sec. 5.4.2):
# python export_for_comparison.py "${COMMON[@]}" --cfg_text 4.5 --output_dir comparative_eval/mardm_cfg_sweep \
#   --cfg_style_sweep 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0

# 3) Unified metrics (FID, R-Precision, Diversity, RR/Global MPJPE, foot-skating, SRA).
#    Baseline exports (smoodi/, loramdm/) are produced in their own repositories with the same manifest.
python evaluate_comparative.py --manifest comparative_eval/manifest.json --comp_root comparative_eval \
  --models mardm_2way_c4.5 mardm_3way_t4.5_s2.0 --eval_modes styled base transfer \
  --output comparative_eval/results.json

# 4) Plots (Sec. 5.4.1 / 5.4.2)
# python analysis/per_style_analysis.py comparative_eval/results.json
# python analysis/cfg_sweep_analysis.py comparative_eval/results_cfg_sweep.json
