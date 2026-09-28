"""SMPL video renderer for VE-100STYLES (from the original render_smpl_vid.py).

Pipeline per motion: joints [T,22,3] -> SMPLify-3D -> SMPL (neutral, beta=0) meshes ->
pyrender offscreen at 1920x1080 with a fixed camera, two directional lights and a floor
anchored at the lowest point of the animation -> MP4 (libx264, CRF 18).

One video frame is rendered per motion frame, so frame i of the video is aligned with frame i
of the 20 FPS motion features. `fps` only sets the container frame rate (existing
VE-100STYLES videos: 15).
"""
import os
import shutil
import subprocess

import imageio
import numpy as np
import pyrender
import smplx
import torch
import trimesh
import trimesh.transformations
from tqdm import tqdm

try:
    from moviepy.editor import ImageSequenceClip  # moviepy 1.x
except ImportError:
    try:
        from moviepy import ImageSequenceClip  # moviepy >= 2.0
    except ImportError:
        ImageSequenceClip = None  # fall back to piping frames into the ffmpeg binary

from .paths import SMPL_NEUTRAL_PKL, check_assets
from .rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from .smplify import joints2smpl

VIEW_SUFFIX = {"front": "FV", "left": "LV", "back": "BV", "right": "RV"}
VIEWPORT_WIDTH, VIEWPORT_HEIGHT = 1920, 1080


def generate_rotation_matrix_from_viewpoint(viewpoint):
    if viewpoint == "front":
        return np.eye(3)
    if viewpoint == "back":
        return trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])[:3, :3]
    if viewpoint == "left":
        return trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])[:3, :3]
    if viewpoint == "right":
        return trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3]
    raise ValueError(f"Unknown viewpoint {viewpoint!r}")


def get_camera_pose(viewpoint, base_distance, camera_height_y=0.0, tilt_degrees=0.0):
    """Camera looking at the world origin from `viewpoint` at `base_distance`."""
    camera_pose = np.eye(4)
    rotation_mat = generate_rotation_matrix_from_viewpoint(viewpoint)
    tilt_matrix = trimesh.transformations.rotation_matrix(np.deg2rad(tilt_degrees), [1, 0, 0])[:3, :3]
    camera_pose[:3, :3] = np.dot(tilt_matrix, rotation_mat)
    position = {"front": (0, camera_height_y, base_distance),
                "back": (0, camera_height_y, -base_distance),
                "left": (-base_distance, camera_height_y, 0),
                "right": (base_distance, camera_height_y, 0)}[viewpoint]
    camera_pose[:3, 3] = position
    return camera_pose


def create_floor(size=120.0, thickness=0.05, color=(0.2, 0.2, 0.8)):
    floor_trimesh = trimesh.creation.box(extents=[size, thickness, size])
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*color, 1.0],
                                                  metallicFactor=0.2, roughnessFactor=0.4)
    return pyrender.Mesh.from_trimesh(floor_trimesh, material=material, smooth=False)


def calculate_overall_motion_bounds_and_yfov(all_frames_vertices, camera_distance_to_center, fov_framing_factor=1.0):
    """Centroid of the whole animation, vertical FoV that frames it, and its lowest (centred) Y."""
    all_points_flat = all_frames_vertices.reshape(-1, 3)
    overall_centroid = (all_points_flat.min(axis=0) + all_points_flat.max(axis=0)) / 2.0
    centered = all_points_flat - overall_centroid
    height = centered[:, 1].max() - centered[:, 1].min()
    lowest_y = centered[:, 1].min()
    if height <= 0 or camera_distance_to_center <= 0:
        return overall_centroid, np.pi / 3.0, lowest_y
    half_yfov = np.arctan((height * fov_framing_factor / 2.0) / camera_distance_to_center)
    return overall_centroid, float(np.clip(2 * half_yfov, np.deg2rad(5), np.deg2rad(120))), lowest_y


def fit_smpl(joints, device):
    """joints [T,22,3] -> SMPL vertices [T,6890,3], faces, and SMPL parameters."""
    check_assets()
    nframes = joints.shape[0]
    j2s = joints2smpl(num_frames=nframes, device=device)
    motion_tensor, _ = j2s.joint2smpl(torch.from_numpy(joints).float().to(device))

    params = motion_tensor.squeeze(0)                         # [25, 6, T]
    transl = params[24, :3, :].permute(1, 0)                 # [T, 3]
    pose_6d = params[:24, :, :].permute(2, 0, 1).float()     # [T, 24, 6]
    pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(pose_6d))
    smpl_params = {"global_orient": pose_aa[:, 0:1, :], "body_pose": pose_aa[:, 1:, :],
                   "transl": transl, "betas": torch.zeros((nframes, 10), device=device)}

    smpl_model = smplx.create(model_path=SMPL_NEUTRAL_PKL, model_type="smpl", gender="neutral",
                              batch_size=nframes).to(device)
    output = smpl_model(**{k: v.to(device) for k, v in smpl_params.items()})
    return output.vertices.detach().cpu().numpy(), smpl_model.faces, smpl_params


def render_video(joints, video_path, viewpoint="front", fps=15, device="cuda",
                 camera_elevation=0.0, camera_distance=10.0, fov_framing_factor=1.2, tilt_degrees=-1.0,
                 frames_dir=None, mesh_dir=None, mesh_format="obj", export_all_frames=False, fit=None):
    """Render joints [T,22,3] to `video_path`. Returns the path, or None if encoding failed.

    `fit` = output of fit_smpl(joints, device); pass it to reuse one SMPLify fit for several views.
    """
    vertices, faces, smpl_params = fit if fit is not None else fit_smpl(joints, device)
    nframes = vertices.shape[0]
    if mesh_dir is not None:
        export_smpl_mesh(vertices, faces, os.path.splitext(os.path.basename(video_path))[0],
                         mesh_dir, mesh_format, export_all_frames, smpl_params)

    centroid, yfov, lowest_y = calculate_overall_motion_bounds_and_yfov(vertices, camera_distance, fov_framing_factor)
    floor_mesh = create_floor(size=120.0, thickness=0.05, color=(0.545, 0.271, 0.075))
    floor_pose = np.eye(4)
    floor_pose[1, 3] = lowest_y - 0.05 / 2.0

    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=VIEWPORT_WIDTH / VIEWPORT_HEIGHT, znear=0.05, zfar=100.0)
    renderer = pyrender.OffscreenRenderer(viewport_width=VIEWPORT_WIDTH, viewport_height=VIEWPORT_HEIGHT)
    cam_pose = get_camera_pose(viewpoint, camera_distance, camera_height_y=camera_elevation, tilt_degrees=tilt_degrees)
    fill_pose = np.copy(cam_pose)
    fill_pose[:3, 3] = -fill_pose[:3, 3]
    body_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.4, 0.6, 0.9, 1.0],
                                                       metallicFactor=0.1, roughnessFactor=0.5)
    if frames_dir is not None:
        os.makedirs(frames_dir, exist_ok=True)

    frames = []
    for i in tqdm(range(nframes), desc=os.path.basename(video_path), leave=False):
        mesh = trimesh.Trimesh(vertices=vertices[i] - centroid, faces=faces)
        scene = pyrender.Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True, material=body_material))
        scene.add(floor_mesh, pose=floor_pose)
        scene.add(pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=2.0), pose=cam_pose)
        scene.add(pyrender.DirectionalLight(color=np.array([0.7, 0.7, 0.8]), intensity=1.0), pose=fill_pose)
        scene.add(camera, pose=cam_pose)
        color = renderer.render(scene)[0]
        frames.append(color)
        if frames_dir is not None:
            imageio.imwrite(os.path.join(frames_dir, f"frame_{i:03d}.png"), color)
    renderer.delete()

    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    tmp_path = video_path + ".part.mp4"   # write-then-rename: no truncated videos on interruption
    try:
        if ImageSequenceClip is not None:
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(tmp_path, codec="libx264", audio=False, threads=os.cpu_count(),
                                 preset="medium", logger=None, ffmpeg_params=["-crf", "18"])
        else:
            write_mp4_ffmpeg(frames, tmp_path, fps)
        os.replace(tmp_path, video_path)
        return video_path
    except Exception as e:
        print(f"[error] encoding {video_path} failed: {type(e).__name__}: {e} (is ffmpeg installed?)")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None


def write_mp4_ffmpeg(frames, path, fps):
    """Encode RGB frames with the ffmpeg binary (libx264, CRF 18, yuv420p) - used when moviepy is absent."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("neither moviepy nor an ffmpeg binary on PATH is available")
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-f", "mp4", path]
    # stream frame by frame: stacking 1080p clips would need several GB of extra RAM
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame[..., :3], dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg exited with {proc.returncode}: {err}")


def export_smpl_mesh(vertices, faces, name, mesh_dir, mesh_format="obj", export_all_frames=False, smpl_params=None):
    """Export SMPL meshes as .obj/.ply (first or all frames) or one .npz with vertices + parameters."""
    os.makedirs(mesh_dir, exist_ok=True)
    fmt = mesh_format.lower()
    if fmt == "npz":
        data = {"vertices": vertices, "faces": faces}
        if smpl_params is not None:
            data.update({k: v.detach().cpu().numpy() for k, v in smpl_params.items()})
        np.savez_compressed(os.path.join(mesh_dir, f"{name}_all_frames.npz"), **data)
    elif fmt in ("obj", "ply"):
        idx = range(vertices.shape[0]) if export_all_frames else [0]
        for i in idx:
            trimesh.Trimesh(vertices=vertices[i], faces=faces).export(
                os.path.join(mesh_dir, f"{name}_frame_{i:03d}.{fmt}"))
    else:
        raise ValueError(f"Unsupported mesh format {mesh_format!r}")
