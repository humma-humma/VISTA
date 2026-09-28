"""Asset locations for the VE-100STYLES renderer.

All paths resolve relative to this sub-repository, so the tools work from any working
directory. Set VE100STYLES_ASSETS to keep the assets elsewhere.

    <assets>/body_models/smpl/SMPL_NEUTRAL.pkl
    <assets>/smplify/{gmm_08.pkl, neutral_smpl_mean_params.h5, smplx_parts_segm.pkl, SMPL_downsample_index.pkl}
"""
import os

SUBREPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.environ.get("VE100STYLES_ASSETS", os.path.join(SUBREPO_ROOT, "assets"))

SMPL_MODEL_DIR = os.path.join(ASSETS_DIR, "body_models")           # smplx.create(<dir>, model_type="smpl")
SMPL_NEUTRAL_PKL = os.path.join(SMPL_MODEL_DIR, "smpl", "SMPL_NEUTRAL.pkl")
SMPLIFY_DIR = os.path.join(ASSETS_DIR, "smplify")


def check_assets():
    missing = [p for p in (SMPL_NEUTRAL_PKL,
                           os.path.join(SMPLIFY_DIR, "gmm_08.pkl"),
                           os.path.join(SMPLIFY_DIR, "neutral_smpl_mean_params.h5"))
               if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing SMPL/SMPLify assets:\n  " + "\n  ".join(missing)
            + "\nRun `bash download_assets.sh` in the ve100styles directory, or set VE100STYLES_ASSETS.")
