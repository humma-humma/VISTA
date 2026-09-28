"""
export_meshes.py — Convert (T, 22, 3) joint positions to SMPL mesh .obj files.

Pipeline:
  1. Run SMPLify (IK) to recover SMPL pose parameters from joints
  2. Forward-pass through SMPL to get vertices + faces
  3. Export each frame as a numbered .obj file

Usage:
  python export_meshes.py \
      --input  path/to/joints.npy \
      --output path/to/mesh_frames/ \
      --device_id 0
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path


import argparse
from os.path import join as pjoin
import os
import time
import numpy as np
import torch
import trimesh
from pathlib import Path

# ── You already have these in your project ──
from visualize.simplify_loc2rot import joints2smpl
from visualize.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle

# ── SMPL forward pass (using smplx library) ──
# Install: pip install smplx
import smplx


def joints_to_smpl_params(joints: np.ndarray, device_id: int):
    """Inverse kinematics: (T,22,3) joints → SMPL pose params."""
    nframes = joints.shape[0]
    j2s = joints2smpl(num_frames=nframes, device_id=device_id)

    print(f"Running SMPLify on {nframes} frames …")
    t0 = time.time()
    thetas, _ = j2s.joint2smpl(joints)
    print(f"  Done in {time.time() - t0:.1f}s")

    params = thetas.squeeze(0)  # (25, 6, T)

    # Translation
    trans = params[24, :3, :].permute(1, 0)  # (T, 3)

    # Rotations: 6D → matrix → axis-angle
    rot_6d = params[:24, :, :].permute(2, 0, 1)  # (T, 24, 6)
    rot_mat = rotation_6d_to_matrix(rot_6d)
    rot_aa = matrix_to_axis_angle(rot_mat)  # (T, 24, 3)

    global_orient = rot_aa[:, 0, :]  # (T, 3)
    body_pose = rot_aa[:, 1:, :].reshape(nframes, -1)  # (T, 69)

    return global_orient, body_pose, trans


def export_obj_sequence(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    trans: torch.Tensor,
    out_dir: str,
    smpl_model_path: str = "body_models/smpl",
    gender: str = "neutral",
):
    """Forward SMPL and write one .obj per frame."""
    os.makedirs(out_dir, exist_ok=True)
    device = global_orient.device

    # Load SMPL via the smplx library
    model = smplx.create(
        model_path=smpl_model_path,
        model_type="smpl",
        gender=gender,
        batch_size=1,
    ).to(device)

    faces = model.faces  # (F, 3) numpy int array — constant across frames

    nframes = global_orient.shape[0]
    print(f"Exporting {nframes} .obj files to {out_dir} …")

    for i in range(nframes):
        output = model(
            global_orient=global_orient[i : i + 1],
            body_pose=body_pose[i : i + 1],
            transl=trans[i : i + 1],
        )
        verts = output.vertices.detach().cpu().numpy().squeeze()  # (6890, 3)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.export(os.path.join(out_dir, f"frame_{i:05d}.obj"))

        if (i + 1) % 50 == 0 or i == nframes - 1:
            print(f"  {i + 1}/{nframes}")

    # Also save faces once (Blender script will need them)
    np.save(os.path.join(out_dir, "faces.npy"), faces)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to (T,22,3) joints .npy parent directory")
    parser.add_argument("--input_file", help="Input file name for .npy file")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--smpl_path", default="body_models/smpl", help="Path to SMPL .pkl files")
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    full_path = pjoin(args.root, args.input_file)
    joints = np.load(full_path, allow_pickle=True)
    print(f"Loaded joints: {joints.shape}")

    global_orient, body_pose, trans = joints_to_smpl_params(joints, args.device_id)

    # Move to device for SMPL forward pass
    global_orient = global_orient.to(device).float()
    body_pose = body_pose.to(device).float()
    trans = trans.to(device).float()

    ip_stem = Path(args.input_file).name.removesuffix('.npy')
    output_path = pjoin(args.root, ip_stem)

    export_obj_sequence(
        global_orient, body_pose, trans,
        out_dir=output_path,
        smpl_model_path=args.smpl_path,
        gender=args.gender,
    )


if __name__ == "__main__":
    main()