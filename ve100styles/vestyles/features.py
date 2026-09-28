"""HumanML3D 263-dim feature handling (MM_MARDM / 100STYLE-SMPL layout).

Per frame (Guo et al., 2022):
    [0]        root angular velocity (Y)
    [1:3]      root linear velocity (XZ, root frame)
    [3]        root height
    [4:67]     21 local joint positions (RIC)
    [67:193]   21 local joint rotations (6D)
    [193:259]  22 local joint velocities
    [259:263]  4 foot-contact labels

Joint recovery follows HumanML3D's `recover_from_ric`, which reads channels 0:67 of the
263-dim vector. The input contract is the full 263-dim representation: 67-dim VISTA features
are rejected, and normalisation always uses the 263-dim Mean.npy / Std.npy.
"""
import warnings

import numpy as np
import torch

FEATURE_DIM = 263
N_JOINTS = 22
ROOT_HEIGHT_CH = 3


def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q, v):
    """Rotate vectors v [*, 3] by quaternions q [*, 4]."""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    # Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    # add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    # add Y-axis rotation to local joints
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    # add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    # concatenate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def load_stats(mean_path, std_path):
    mean, std = np.load(mean_path), np.load(std_path)
    if mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,):
        raise ValueError(f"Mean/Std must be ({FEATURE_DIM},) HumanML3D statistics, got {mean.shape} / {std.shape}")
    return mean, std


def to_joints(motion, normalized=False, mean=None, std=None):
    """Return world-space joints [T, 22, 3] from 263-dim features or already-recovered joints.

    motion:     [T, 263] features (raw, as stored in 100STYLE-SMPL/new_joint_vecs), or
                [T, 22, 3] / [22, 3, T] joints (e.g. 100STYLE-SMPL/new_joints).
    normalized: set when the 263-dim features are z-normalised; `mean`/`std` are then required.
    """
    motion = np.asarray(motion)
    if motion.ndim == 3:
        if motion.shape[1] == 3 and motion.shape[0] == N_JOINTS:  # [22, 3, T]
            motion = np.transpose(motion, (2, 0, 1))
        if motion.shape[1:] != (N_JOINTS, 3):
            raise ValueError(f"Joint input must be [T, {N_JOINTS}, 3], got {motion.shape}")
        return motion.astype(np.float32)

    if motion.ndim != 2 or motion.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [T, {FEATURE_DIM}] HumanML3D features, got {motion.shape}. "
                         "67-dim VISTA features are not accepted; render from new_joint_vecs (263-dim).")

    feats = motion.astype(np.float32)
    root_h = float(np.mean(feats[:, ROOT_HEIGHT_CH]))
    if normalized:
        if mean is None or std is None:
            raise ValueError("normalized=True requires the 263-dim Mean.npy / Std.npy")
        if root_h > 0.5:
            warnings.warn(f"--normalized set but mean root height is {root_h:.2f} (looks like raw features)")
        feats = feats * std + mean
    elif abs(root_h) < 0.3:
        warnings.warn(f"Mean root height {root_h:.2f} m looks z-normalised; pass --normalized with Mean/Std if so")

    joints = recover_from_ric(torch.from_numpy(feats).unsqueeze(0), N_JOINTS)
    return joints.squeeze(0).numpy()
