"""Render the VE-100STYLES video set from a 100STYLE-SMPL dataset (MM_MARDM layout).

Input  (--data_root, default datasets/100STYLE-SMPL):
    new_joint_vecs/<id>.npy      263-dim HumanML3D features (raw, not normalised)
    100STYLE_name_dict.txt       "<id> <Style>_<Content>_<nn>.bvh <...>" (style of every id)
Output (--output_dir, default <data_root>/videos):
    <id>_FV.mp4 (front), <id>_LV.mp4 (left)  [also _BV / _RV for back / right]

Existing videos are skipped, so an interrupted run can simply be restarted.

Examples:
    # training styles, front + left views (thesis setup)
    python ve100styles/render_dataset.py --data_root datasets/100STYLE-SMPL \
        --styles Aeroplane ArmsFolded Chicken Robot Superman --views front left

    # held-out styles for the unseen-style study
    python ve100styles/render_dataset.py --data_root datasets/100STYLE-SMPL \
        --styles Cat FlickLegs --views front --output_dir datasets/100STYLE-SMPL/videos_OOD
"""
import argparse
import os
import sys
import traceback

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this sub-repo only
from vestyles.features import load_stats, to_joints  # noqa: E402
from vestyles.renderer import VIEW_SUFFIX, fit_smpl, render_video  # noqa: E402


def read_style_map(name_dict_path):
    """id -> style name (lower case), parsed from '<id> <Style>_<...>.bvh ...' lines."""
    id_to_style = {}
    with open(name_dict_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                id_to_style[parts[0]] = parts[1].split('_')[0].lower()
    return id_to_style


def select_ids(args):
    motion_dir = os.path.join(args.data_root, args.motion_subdir)
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(motion_dir) if f.endswith('.npy'))
    if args.ids_file:
        with open(args.ids_file) as f:
            wanted = {l.strip() for l in f if l.strip() and not l.lstrip().startswith('#')}
        ids = [i for i in ids if i in wanted]
    if args.styles:
        style_map = read_style_map(args.name_dict or os.path.join(args.data_root, '100STYLE_name_dict.txt'))
        wanted_styles = {s.lower() for s in args.styles}
        ids = [i for i in ids if style_map.get(i) in wanted_styles]
    if not args.include_mirrored:
        ids = [i for i in ids if not i.startswith('M')]
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', default=os.path.join('datasets', '100STYLE-SMPL'))
    ap.add_argument('--motion_subdir', default='new_joint_vecs',
                    help="new_joint_vecs (263-dim features, default) or new_joints ([T,22,3] joints)")
    ap.add_argument('--output_dir', default=None, help='default: <data_root>/videos')
    ap.add_argument('--styles', nargs='+', default=None, help='style names to render (default: all ids)')
    ap.add_argument('--name_dict', default=None, help='default: <data_root>/100STYLE_name_dict.txt')
    ap.add_argument('--ids_file', default=None, help='optional file with one motion id per line')
    ap.add_argument('--include_mirrored', action=argparse.BooleanOptionalAction, default=True,
                    help="also render mirrored 'M<id>' sequences (the MM_MARDM video set contains them)")
    ap.add_argument('--views', nargs='+', default=['front', 'left'], choices=list(VIEW_SUFFIX))
    ap.add_argument('--normalized', action='store_true',
                    help='inputs are z-normalised 263-dim features (then --mean/--std are used)')
    ap.add_argument('--mean', default=None, help='default: <data_root>/Mean.npy (263-dim)')
    ap.add_argument('--std', default=None, help='default: <data_root>/Std.npy (263-dim)')
    ap.add_argument('--fps', type=int, default=15,
                    help='container frame rate; one video frame per motion frame regardless (existing set: 15)')
    # Camera used for the released VE-100STYLES videos (reproduces them at ~50 dB PSNR, body IoU 0.99).
    # The original bulk-script defaults were elevation 0.3 / framing 2.0 / tilt 0.0.
    ap.add_argument('--camera_elevation', type=float, default=0.0)
    ap.add_argument('--camera_distance', type=float, default=10.0)
    ap.add_argument('--fov_framing_factor', type=float, default=1.2)
    ap.add_argument('--tilt_degrees', type=float, default=-1.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--overwrite', action='store_true', help='re-render videos that already exist')
    ap.add_argument('--limit', type=int, default=None, help='render at most N ids (smoke test)')
    ap.add_argument('--dry_run', action='store_true', help='list what would be rendered and exit')
    ap.add_argument('--store_frames', action='store_true', help='also write PNG frames to <output_dir>/<id>_<view>/')
    ap.add_argument('--export_mesh', action='store_true', help='also export SMPL meshes to <output_dir>/meshes/')
    ap.add_argument('--mesh_format', default='obj', choices=['obj', 'ply', 'npz'])
    ap.add_argument('--export_all_frames', action='store_true')
    args = ap.parse_args()

    out_dir = args.output_dir or os.path.join(args.data_root, 'videos')
    mean = std = None
    if args.normalized:
        mean, std = load_stats(args.mean or os.path.join(args.data_root, 'Mean.npy'),
                               args.std or os.path.join(args.data_root, 'Std.npy'))

    ids = select_ids(args)
    jobs = [(i, v) for i in ids for v in args.views
            if args.overwrite or not os.path.exists(os.path.join(out_dir, f"{i}_{VIEW_SUFFIX[v]}.mp4"))]
    if args.limit is not None:
        jobs = jobs[:args.limit * len(args.views)]
    print(f"{len(ids)} ids x {len(args.views)} views selected; {len(jobs)} videos to render -> {out_dir}")
    if args.dry_run:
        for i, v in jobs[:50]:
            print(f"  {i}_{VIEW_SUFFIX[v]}.mp4")
        return

    failed = []
    motion_dir = os.path.join(args.data_root, args.motion_subdir)
    cache = {}  # one SMPLify fit per motion, reused across views
    for motion_id, view in tqdm(jobs, desc='videos'):
        video_path = os.path.join(out_dir, f"{motion_id}_{VIEW_SUFFIX[view]}.mp4")
        try:
            if motion_id not in cache:
                cache.clear()
                joints = to_joints(np.load(os.path.join(motion_dir, motion_id + '.npy')),
                                   normalized=args.normalized, mean=mean, std=std)
                cache[motion_id] = (joints, fit_smpl(joints, args.device))
            joints, fit = cache[motion_id]
            ok = render_video(
                joints, video_path, fit=fit, viewpoint=view, fps=args.fps, device=args.device,
                camera_elevation=args.camera_elevation, camera_distance=args.camera_distance,
                fov_framing_factor=args.fov_framing_factor, tilt_degrees=args.tilt_degrees,
                frames_dir=os.path.join(out_dir, f"{motion_id}_{VIEW_SUFFIX[view]}") if args.store_frames else None,
                mesh_dir=os.path.join(out_dir, 'meshes') if args.export_mesh else None,
                mesh_format=args.mesh_format, export_all_frames=args.export_all_frames)
            if ok is None:
                failed.append(video_path)
        except Exception:
            traceback.print_exc()
            failed.append(video_path)

    print(f"done: {len(jobs) - len(failed)} rendered, {len(failed)} failed")
    if failed:
        fail_log = os.path.join(out_dir, 'failed_renders.txt')
        with open(fail_log, 'w') as f:
            f.write('\n'.join(failed) + '\n')
        print(f"failed paths written to {fail_log}")
        sys.exit(1)


if __name__ == '__main__':
    main()
