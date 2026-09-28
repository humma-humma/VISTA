"""Single-run interactive keyframe/camera selection + headless render.

Open the aitviewer window, orbit/zoom/step through frames, press 'M' to mark
keyframes, then close the window. The script then flips aitviewer to headless
mode and renders (in the same process):

  - a motion-strip PNG using the marked frames (or a linspace fallback)
  - a side-by-side video (.mp4/.gif) using the final camera pose

aitviewer_test_wip.py is left untouched; shared helpers are duplicated here on
purpose so this script can evolve independently.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path


import argparse
import os
import shutil
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join as pjoin

from aitviewer.configuration import CONFIG as C
C.update_conf({"window_type": "pyqt5"})

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.shaders import clear_shader_cache
from aitviewer.viewer import Viewer

from visualize.simplify_loc2rot import joints2smpl
from visualize.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle


_PALETTE = [
    (0.35, 0.60, 0.28),  # grass green  — GT
    (0.35, 0.45, 0.65),  # deep blue    — Motion
    (0.72, 0.54, 0.10),  # golden       — Video
    (0.60, 0.40, 0.55),
    (0.45, 0.45, 0.55),
]

_GT_COLOR_MAP = {
    'grass':  (0.35, 0.60, 0.28),
    'blue':   (0.35, 0.45, 0.65),
    'golden': (0.72, 0.54, 0.10),
    'red':    (0.88, 0.52, 0.52),
}


def temporal_interpolate_joints(joints, factor):
    if factor <= 1 or joints.shape[0] < 2:
        return joints
    t = torch.from_numpy(joints).permute(2, 1, 0).reshape(1, -1, joints.shape[0]).float()
    up = F.interpolate(t, scale_factor=factor, mode='linear', align_corners=True)
    up = up.reshape(3, joints.shape[1], -1).permute(2, 1, 0).contiguous()
    return up.numpy()


def load_data(root_path, source_file):
    return np.load(pjoin(root_path, source_file), allow_pickle=True)


def motion_to_smpl_params(motion_sequence, device_id):
    nframes = motion_sequence.shape[0]
    j2s = joints2smpl(num_frames=nframes, device_id=device_id)
    thetas, _ = j2s.joint2smpl(motion_sequence)
    params = thetas.squeeze(0)
    trans = params[24, :3, :].permute(1, 0)
    rot_6d = params[:24, :, :].permute(2, 0, 1)
    rot_mat = rotation_6d_to_matrix(rot_6d)
    rot_aa = matrix_to_axis_angle(rot_mat)
    poses_root = rot_aa[:, 0, :]
    poses_body = rot_aa[:, 1:, :].reshape(nframes, -1)
    return poses_body, poses_root, trans


def _to_numpy_params(p):
    pb, pr, tr = p
    return (pb.detach().cpu().numpy() if torch.is_tensor(pb) else pb,
            pr.detach().cpu().numpy() if torch.is_tensor(pr) else pr,
            tr.detach().cpu().numpy() if torch.is_tensor(tr) else tr)


def run_interactive(params_list, titles, device, spacing):
    """Return (marked_frames, final_cam_pos, final_cam_target)."""
    C.smplx_models = "body_models/"
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=device)
    N = len(params_list)
    offsets = [0.0] if N == 1 else np.linspace(-(N - 1) / 2.0, (N - 1) / 2.0, N) * spacing

    marked = []

    class _MarkedViewer(Viewer):
        def key_event(self, key, action, modifiers):
            super().key_event(key, action, modifiers)
            if action == self.wnd.keys.ACTION_PRESS and key == self.wnd.keys.M:
                frame = self.scene.current_frame_id
                pos = tuple(round(float(x), 4) for x in self.scene.camera.position)
                tgt = tuple(round(float(x), 4) for x in self.scene.camera.target)
                marked.append((frame, pos, tgt))
                print(f"[Mark] frame={frame:4d}  cam_pos={pos}  cam_target={tgt}")

    v = _MarkedViewer()
    for i, (params, offset, title) in enumerate(zip(params_list, offsets, titles)):
        pb, pr, tr = params
        n_frames = pb.shape[0]
        rgb = _PALETTE[i % len(_PALETTE)]
        tr_off = tr.copy()
        tr_off[:, 0] += offset
        seq = SMPLSequence(
            poses_body=pb.astype(np.float32),
            poses_root=pr.astype(np.float32),
            poses_left_hand=np.zeros((n_frames, 0), np.float32),
            poses_right_hand=np.zeros((n_frames, 0), np.float32),
            trans=tr_off.astype(np.float32),
            betas=np.zeros(10, np.float32),
            smpl_layer=smpl_layer,
            name=title,
            color=(rgb[0], rgb[1], rgb[2], 1.0),
        )
        seq.skeleton_seq.enabled = False
        v.scene.add(seq)

    print("\n--- Interactive Viewer ---")
    print("  Orbit: left-drag  |  Zoom: scroll  |  Pan: middle-drag")
    print("  Play/pause: Space  |  Step frames: Left / Right arrow")
    print("  M: mark current frame + camera")
    print("  Close window to trigger headless rendering.")
    print("--------------------------\n")
    v.run()

    pos = tuple(round(float(x), 4) for x in v.scene.camera.position)
    tgt = tuple(round(float(x), 4) for x in v.scene.camera.target)
    frames = [f for f, *_ in marked]

    print("\n=== Interactive session complete ===")
    print(f"  Marked frames: {frames if frames else '(none - linspace fallback)'}")
    print(f"  Camera: pos={pos}, target={tgt}")
    print("====================================\n")
    return frames, pos, tgt


def _switch_to_headless():
    C.update_conf({"window_type": "headless"})
    from aitviewer.headless import HeadlessRenderer
    return HeadlessRenderer


def _dedupe_preserve(indices, n_frames):
    seen, out = set(), []
    for k in indices:
        ki = max(0, min(n_frames - 1, int(k)))
        if ki not in seen:
            seen.add(ki)
            out.append(ki)
    return np.array(out, dtype=int)


def render_motion_strip(
    params_list, titles, output_path, device,
    keyframe_indices=None, n_keyframes=20, spacing=3.0,
    alpha_range=(0.50, 0.95),
    cam_pos=(0.0, 1.8, 12.0), cam_target=(0.0, 1.0, 0.0),
):
    HeadlessRenderer = _switch_to_headless()
    C.smplx_models = "body_models/"
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=device)

    N = len(params_list)
    offsets = [0.0] if N == 1 else np.linspace(-(N - 1) / 2.0, (N - 1) / 2.0, N) * spacing
    a_lo, a_hi = alpha_range

    if not output_path.lower().endswith((".png", ".jpg", ".jpeg")):
        output_path = os.path.splitext(output_path)[0] + ".png"

    v = HeadlessRenderer()
    last_nkf = 0
    for i, (params, offset, title) in enumerate(zip(params_list, offsets, titles)):
        pb, pr, tr = params
        n_frames = pb.shape[0]
        if keyframe_indices:
            idx = _dedupe_preserve(keyframe_indices, n_frames)
        else:
            idx = np.linspace(0, n_frames - 1, n_keyframes).astype(int)

        base_rgb = _PALETTE[i % len(_PALETTE)]
        n_kf = len(idx)
        last_nkf = n_kf
        for k, frame_i in enumerate(idx):
            t = k / max(1, n_kf - 1)
            alpha = a_lo + t * (a_hi - a_lo)
            tr_i = tr[frame_i:frame_i + 1].copy()
            tr_i[:, 0] += offset
            name = title if k == n_kf - 1 else ""
            seq = SMPLSequence(
                poses_body=pb[frame_i:frame_i + 1].astype(np.float32),
                poses_root=pr[frame_i:frame_i + 1].astype(np.float32),
                poses_left_hand=np.zeros((1, 0), np.float32),
                poses_right_hand=np.zeros((1, 0), np.float32),
                trans=tr_i.astype(np.float32),
                betas=np.zeros(10, np.float32),
                smpl_layer=smpl_layer,
                name=name,
                color=(base_rgb[0], base_rgb[1], base_rgb[2], alpha),
            )
            seq.skeleton_seq.enabled = False
            v.scene.add(seq)

    v.scene.camera.position = np.array(cam_pos, dtype=np.float32)
    v.scene.camera.target = np.array(cam_target, dtype=np.float32)
    print(f"Rendering strip ({N} agents x {last_nkf} keyframes) -> {output_path}")
    v.save_frame(file_path=output_path)
    v.on_close()
    clear_shader_cache()


def render_video(
    params_list, titles, output_path, device,
    fps=30.0, spacing=2.0,
    cam_pos=(0.0, 2.5, 6.5), cam_target=(0.0, 1.0, 0.0),
    gt_color=None,
):
    HeadlessRenderer = _switch_to_headless()
    C.smplx_models = "body_models/"
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=device)

    N = len(params_list)
    if N == 1:
        offsets = [0.0]
        colors = [(0.2, 0.7, 0.2, 1.0)]
    elif N == 2:
        offsets = [-spacing / 2.0, spacing / 2.0]
        colors = [(0.7, 0.2, 0.2, 1.0), (0.2, 0.7, 0.2, 1.0)]
    else:
        if N > 3:
            print(f"Warning: video mode supports up to 3 agents; dropping {N - 3}.")
            params_list = params_list[:3]
            titles = titles[:3]
            N = 3
        offsets = [-spacing, 0.0, spacing]
        gt_rgb = gt_color if gt_color is not None else (0.35, 0.60, 0.28)
        colors = [(gt_rgb[0], gt_rgb[1], gt_rgb[2], 0.82), (0.35, 0.45, 0.65, 1.0), (0.72, 0.54, 0.10, 1.0)]

    v = HeadlessRenderer()
    max_frames = 0
    for (params, offset, title, color) in zip(params_list, offsets, titles, colors):
        pb, pr, tr = params
        n_frames = pb.shape[0]
        max_frames = max(max_frames, n_frames)
        tr_off = tr.copy()
        tr_off[:, 0] += offset
        seq = SMPLSequence(
            poses_body=pb.astype(np.float32),
            poses_root=pr.astype(np.float32),
            poses_left_hand=np.zeros((n_frames, 0), np.float32),
            poses_right_hand=np.zeros((n_frames, 0), np.float32),
            trans=tr_off.astype(np.float32),
            betas=np.zeros(10, np.float32),
            smpl_layer=smpl_layer,
            name=title,
            color=color,
        )
        seq.skeleton_seq.enabled = False
        v.scene.add(seq)

    v.scene.camera.position = np.array(cam_pos, dtype=np.float32)
    v.scene.camera.target = np.array(cam_target, dtype=np.float32)
    print(f"Rendering {max_frames} frames -> {output_path}")
    abs_out = os.path.abspath(output_path)
    if os.name == 'nt' and len(abs_out) > 200:
        ext = os.path.splitext(output_path)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as _f:
            tmp_path = _f.name
        v.save_video(video_dir=tmp_path, output_fps=fps, animation_range=(0, max_frames))
        shutil.move(tmp_path, abs_out)
    else:
        v.save_video(video_dir=output_path, output_fps=fps, animation_range=(0, max_frames))
    v.on_close()
    clear_shader_cache()


def _load_motions(args):
    motions, titles = [], []
    if args.gt_file:
        motions.append(load_data(args.root_path, args.gt_file))
        titles.append("Ground-truth")
    if args.motion_file:
        motions.append(load_data(args.root_path, args.motion_file))
        titles.append("Motion")
    if args.video_file:
        vid = load_data(args.root_path, args.video_file)[1]
        if vid.ndim == 3 and vid.shape[-1] != 3:
            vid = vid.transpose(2, 0, 1)  # (J, 3, T) -> (T, J, 3)
        motions.append(vid)
        titles.append("Video")

    if args.upsample > 1:
        motions = [temporal_interpolate_joints(m, args.upsample) for m in motions]
    return motions, titles


def main(args):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.makedirs(args.root_path, exist_ok=True)
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

    gt_color = _GT_COLOR_MAP[args.gt_color]
    _PALETTE[0] = gt_color

    motions, titles = _load_motions(args)

    if args.params_cache:
        _raw_cache = args.params_cache if args.params_cache.endswith('.npz') else args.params_cache + '.npz'
    else:
        _stems = [os.path.splitext(f)[0] for f in [args.gt_file, args.motion_file, args.video_file] if f]
        _cache_name = '_'.join(_stems) + '_ik.npz'
        _raw_cache = pjoin(args.root_path, 'SMPL_mesh', _cache_name)
    params_cache = _raw_cache
    os.makedirs(os.path.dirname(params_cache), exist_ok=True)

    expected_n = len(motions)
    cache_valid = os.path.isfile(params_cache)

    if cache_valid:
        print(f"Loading cached IK params from {params_cache} (skipping IK)...")
        data = np.load(params_cache, allow_pickle=False)
        params_list = [(data[f'pb_{i}'], data[f'pr_{i}'], data[f'tr_{i}']) for i in range(expected_n)]
    else:
        print("Running IK for all motions...")
        for t, m in zip(titles, motions):
            print(f"  [{t}] shape={m.shape}, dtype={m.dtype}")
        params_list = [_to_numpy_params(motion_to_smpl_params(m, args.device_id)) for m in motions]
        save_dict = {'n': np.array(len(params_list))}
        for i, (pb, pr, tr) in enumerate(params_list):
            save_dict[f'pb_{i}'] = pb
            save_dict[f'pr_{i}'] = pr
            save_dict[f'tr_{i}'] = tr
        np.savez(params_cache, **save_dict)
        print(f"IK params cached to {params_cache}")

    marked_frames, cam_pos, cam_target = run_interactive(
        params_list, titles, device, args.spacing
    )

    strip_path = pjoin(args.root_path, args.strip_file)
    video_path = pjoin(args.root_path, args.video_out)

    strip_cam_pos = tuple(args.strip_cam_pos) if args.strip_cam_pos else cam_pos
    strip_cam_target = tuple(args.strip_cam_target) if args.strip_cam_target else cam_target

    if args.render in ('strip', 'both'):
        render_motion_strip(
            params_list, titles, strip_path, device,
            keyframe_indices=marked_frames or None,
            n_keyframes=args.n_keyframes,
            spacing=args.spacing,
            cam_pos=strip_cam_pos, cam_target=strip_cam_target,
        )

    if args.render in ('video', 'both'):
        render_video(
            params_list, titles, video_path, device,
            fps=args.fps, spacing=args.spacing,
            cam_pos=cam_pos, cam_target=cam_target,
            gt_color=gt_color,
        )

    print("\nDone.")
    if args.render in ('strip', 'both'):
        print(f"  Strip: {strip_path}")
    if args.render in ('video', 'both'):
        print(f"  Video: {video_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gt_file', type=str, default=None)
    p.add_argument('--motion_file', type=str, default=None)
    p.add_argument('--video_file', type=str, default=None)

    p.add_argument('--root_path', type=str, default='./visualizations/aitviewer_test')
    p.add_argument('--strip_file', type=str, default='strip.png')
    p.add_argument('--video_out', type=str, default='comparison.mp4')

    p.add_argument('--model_type', type=str, default='t2m', choices=['t2m', 'kit'])
    p.add_argument('--device_id', type=int, default=0)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--n_keyframes', type=int, default=20)
    p.add_argument('--spacing', type=float, default=3.0)
    p.add_argument('--upsample', type=int, default=1)
    p.add_argument('--render', type=str, default='both', choices=['strip', 'video', 'both'])
    p.add_argument('--gt_color', type=str, default='grass', choices=['grass', 'blue', 'golden', 'red'],
                   help='Color for the GT motion (default: grass)')
    p.add_argument('--params_cache', type=str, default=None, metavar='PATH',
                   help='Path to save/load IK params (.npz). Defaults to <root_path>/ik_cache.npz. Saves after IK if missing, loads to skip IK if present.')
    p.add_argument('--strip_cam_pos', type=float, nargs=3, default=None)
    p.add_argument('--strip_cam_target', type=float, nargs=3, default=None)

    main(p.parse_args())
