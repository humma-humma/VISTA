"""Render a single motion file (263-dim HumanML3D features or [T,22,3] joints) to MP4.

    python ve100styles/render_motion.py datasets/100STYLE-SMPL/new_joint_vecs/030001.npy \
        --views front left --output_dir /tmp/ve100_check
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this sub-repo only
from vestyles.features import load_stats, to_joints  # noqa: E402
from vestyles.renderer import VIEW_SUFFIX, fit_smpl, render_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('motion', help='.npy with [T,263] features or [T,22,3] joints')
    ap.add_argument('--output_dir', default='.')
    ap.add_argument('--name', default=None, help='output stem (default: input file stem)')
    ap.add_argument('--views', nargs='+', default=['front'], choices=list(VIEW_SUFFIX))
    ap.add_argument('--normalized', action='store_true', help='features are z-normalised; requires --mean/--std')
    ap.add_argument('--mean', default=None)
    ap.add_argument('--std', default=None)
    ap.add_argument('--fps', type=int, default=15)
    # Camera used for the released VE-100STYLES videos (reproduces them at ~50 dB PSNR, body IoU 0.99).
    # The original bulk-script defaults were elevation 0.3 / framing 2.0 / tilt 0.0.
    ap.add_argument('--camera_elevation', type=float, default=0.0)
    ap.add_argument('--camera_distance', type=float, default=10.0)
    ap.add_argument('--fov_framing_factor', type=float, default=1.2)
    ap.add_argument('--tilt_degrees', type=float, default=-1.0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    mean = std = None
    if args.normalized:
        if not (args.mean and args.std):
            ap.error('--normalized requires --mean and --std (263-dim)')
        mean, std = load_stats(args.mean, args.std)
    joints = to_joints(np.load(args.motion), normalized=args.normalized, mean=mean, std=std)
    fit = fit_smpl(joints, args.device)
    stem = args.name or os.path.splitext(os.path.basename(args.motion))[0]
    for view in args.views:
        path = render_video(joints, os.path.join(args.output_dir, f"{stem}_{VIEW_SUFFIX[view]}.mp4"), fit=fit,
                            viewpoint=view, fps=args.fps, device=args.device,
                            camera_elevation=args.camera_elevation, camera_distance=args.camera_distance,
                            fov_framing_factor=args.fov_framing_factor, tilt_degrees=args.tilt_degrees)
        print(path or f"[failed] {view}")


if __name__ == '__main__':
    main()
