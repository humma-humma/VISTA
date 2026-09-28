"""Download third-party pretrained assets that VISTA builds on.

Fetches (from the original MARDM release, Google Drive via gdown):
  checkpoints/t2m/text_mot_match*/       HumanML3D evaluators (FID / R-Precision / MotionCLIP)
  checkpoints/t2m/MARDM-DDPM-XL/         pretrained text-to-motion MARDM (Stage-2 initialisation)
  checkpoints/t2m/MARDM-SiT-XL/          (SiT variant; not used by the thesis model)
  checkpoints/t2m/length_estimator/      length estimator
  checkpoints/t2m/AE/                    HumanML3D motion AE (used to pre-encode HumanML3D latents)
  glove/                                 GloVe vocabulary used by the evaluators

VISTA's own checkpoints (DualAE, Stage-2 MARDM, style classifier) are distributed
separately; see README.md -> "Checkpoints".

Nothing is deleted: existing files are kept and skipped. `checkpoints/` and `glove/` may be
symlinks to external storage.

Usage (from the repository root):
    python prepare/download_pretrained.py [--root .]
"""
import argparse
import os
import shutil
import subprocess
import zipfile

EVALUATORS_URL = "https://drive.google.com/file/d/1ejiz4NvyuoTj3BIdfNrTFFZBZ-zq4oKD/view?usp=sharing"
GLOVE_URL = "https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing"
# Order matches the original MARDM setup script: SiT-XL, DDPM-XL, length estimator, AE.
PRETRAINED = [
    ("https://drive.google.com/file/d/1TBybFByAd-kD4AuFgMyR3ZBt4VV43Sif/view?usp=sharing", "MARDM_SiT_XL.zip"),
    ("https://drive.google.com/file/d/1csjlxi0uOhfPPEwiThsR0gaj7_VDmgb6/view?usp=sharing", "MARDM_DDPM_XL.zip"),
    ("https://drive.google.com/file/d/1nWoEcN4rEFKi4Xyf_ObKinDmSQNPKXgU/view?usp=sharing", "length_estimator.zip"),
    ("https://drive.google.com/file/d/1nfX_j8VzMmynqKv8x68pXrsL3c0qWLXA/view?usp=sharing", "AE_humanml3d.zip"),
]


def fetch_and_unzip(url, zip_name, dest):
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, zip_name)
    if not os.path.exists(zip_path):
        subprocess.run(["gdown", "--fuzzy", url, "-O", zip_path], check=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = os.path.join(dest, member.filename)
            if os.path.exists(target) and not member.is_dir():
                continue  # never overwrite existing files
            zf.extract(member, dest)
    os.remove(zip_path)
    print(f"[ok] {zip_name} -> {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="Repository root (default: current directory)")
    ap.add_argument("--skip_sit", action="store_true", help="Skip the SiT-XL MARDM variant (unused by VISTA)")
    args = ap.parse_args()

    t2m = os.path.join(args.root, "checkpoints", "t2m")
    fetch_and_unzip(EVALUATORS_URL, "evaluators_humanml3d.zip", t2m)
    fetch_and_unzip(GLOVE_URL, "glove.zip", os.path.join(args.root, "glove"))
    for url, name in PRETRAINED:
        if args.skip_sit and name == "MARDM_SiT_XL.zip":
            continue
        fetch_and_unzip(url, name, t2m)

    # Stage-1 evaluation (--dataset_name 100styles) loads the same evaluators from checkpoints/100styles/
    for name in ("text_mot_match", "text_mot_match_clip"):
        src_f = os.path.join(t2m, name, "model", "finest.tar")
        dst_f = os.path.join(args.root, "checkpoints", "100styles", name, "model", "finest.tar")
        if os.path.exists(src_f) and not os.path.exists(dst_f):
            os.makedirs(os.path.dirname(dst_f), exist_ok=True)
            shutil.copy2(src_f, dst_f)
            print(f"[ok] copied {name} evaluator -> {dst_f}")

    # train_MARDM.py --is_continue expects the HumanML3D MARDM as humanml3d_latest.tar
    ddpm_dir = os.path.join(t2m, "MARDM-DDPM-XL", "model")
    src, dst = os.path.join(ddpm_dir, "latest.tar"), os.path.join(ddpm_dir, "humanml3d_latest.tar")
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
        print(f"[ok] renamed {src} -> {dst}")
    elif not os.path.exists(dst):
        print(f"[warn] {dst} not found: place the pretrained HumanML3D MARDM-DDPM-XL checkpoint there before Stage-2 training")
    print("Done. ViViT (google/vivit-b-16x2-kinetics400) and CLIP ViT-B/32 are fetched automatically on first use.")


if __name__ == "__main__":
    main()
