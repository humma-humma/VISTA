"""
Foot contact detection and skating loss for MM-MARDM.

Provides runtime foot contact detection using the dual-threshold approach
(velocity AND height) from DuetGen, adapted for the HumanML3D 67-dim
representation.

Also provides a corrected skating loss that fixes the inverted masking bug
found in DuetGen's foot_skating_loss().

Joint index reference for HumanML3D 67-dim (channel-first [B, 67, T]):
    dims [0:4]   — root features (angular_vel_y, vel_x, vel_z, height)
    dims [4:67]  — 21 joints × 3 (local positions in root space)

    Foot joints (0-indexed within the 21 local joints, offset by 4 in full vector):
        joint 7  = left_ankle   → dims [4 + 7*3 : 4 + 7*3 + 3]  = [25:28] but using
                                                                      SMPL indexing:
        joint 6  = l_ankle      → y = 4 + 6*3 + 1 = 23
        joint 7  = r_ankle      → y = 4 + 7*3 + 1 = 26
        joint 9  = l_foot       → y = 4 + 9*3 + 1 = 32
        joint 10 = r_foot       → y = 4 + 10*3 + 1 = 35
"""

import torch
import torch.nn.functional as F


# =============================================================================
# Joint indices in the 67-dim HumanML3D representation (channel-first)
# =============================================================================
ROOT_HEIGHT_IDX = 3

# Y (height) indices for each foot joint
FOOT_Y_INDICES = {
    'l_ankle': 4 + 6 * 3 + 1,   # dim 23
    'r_ankle': 4 + 7 * 3 + 1,   # dim 26
    'l_foot':  4 + 9 * 3 + 1,   # dim 32
    'r_foot':  4 + 10 * 3 + 1,  # dim 35
}

# XYZ slice start indices for each foot joint (3 consecutive dims each)
FOOT_XYZ_SLICES = {
    'l_ankle': 4 + 6 * 3,   # dims 22, 23, 24
    'r_ankle': 4 + 7 * 3,   # dims 25, 26, 27
    'l_foot':  4 + 9 * 3,   # dims 31, 32, 33
    'r_foot':  4 + 10 * 3,  # dims 34, 35, 36
}

# XZ (horizontal) indices for each foot joint
FOOT_XZ_INDICES = {
    'l_ankle': (4 + 6 * 3, 4 + 6 * 3 + 2),   # (22, 24)
    'r_ankle': (4 + 7 * 3, 4 + 7 * 3 + 2),   # (25, 27)
    'l_foot':  (4 + 9 * 3, 4 + 9 * 3 + 2),   # (31, 33)
    'r_foot':  (4 + 10 * 3, 4 + 10 * 3 + 2),  # (34, 36)
}

# Height thresholds per joint pair (from DuetGen, may need tuning for
# normalized HumanML3D coordinates)
FOOT_HEIGHT_THRESHOLDS = {
    'l_ankle': 0.12,
    'r_ankle': 0.12,
    'l_foot':  0.05,
    'r_foot':  0.05,
}

FOOT_JOINT_NAMES = ['l_ankle', 'r_ankle', 'l_foot', 'r_foot']


# =============================================================================
# Velocity computation
# =============================================================================
def compute_velocities(positions):
    """
    Compute per-frame velocities via finite differencing.

    Args:
        positions: [B, C, T] — joint positions (channel-first)

    Returns:
        velocities: [B, C, T] — frame-to-frame differences,
            with frame 0 velocity set to zero (same as DuetGen convention).
    """
    vel = positions[:, :, 1:] - positions[:, :, :-1]   # [B, C, T-1]
    # Pad frame 0 with zeros to maintain temporal dimension
    zeros = torch.zeros_like(positions[:, :, :1])       # [B, C, 1]
    return torch.cat([zeros, vel], dim=-1)              # [B, C, T]


# =============================================================================
# Foot contact detection
# =============================================================================
@torch.no_grad()
def detect_foot_contact(motion, vel_threshold=0.002, height_thresholds=None):
    """
    Detect foot-ground contact using dual threshold: velocity AND height.

    A foot joint is marked as in-contact if BOTH conditions hold:
        1. Its 3D squared velocity (sum over XYZ) < vel_threshold
        2. Its global height (root_height + local_y) < height_threshold

    This matches DuetGen's foot_detect() logic, adapted for HumanML3D coords.

    Args:
        motion: [B, 67, T] — decoded motion (channel-first)
        vel_threshold: float — squared velocity threshold (default 0.002,
            slightly higher than DuetGen's 0.001 to account for noisier
            generated motions)
        height_thresholds: dict or None — per-joint height thresholds.
            Defaults to FOOT_HEIGHT_THRESHOLDS.

    Returns:
        contact: [B, 4, T] — binary contact mask (float), ordered as
            [l_ankle, r_ankle, l_foot, r_foot]
    """
    if height_thresholds is None:
        height_thresholds = FOOT_HEIGHT_THRESHOLDS

    B, _, T = motion.shape
    root_h = motion[:, ROOT_HEIGHT_IDX, :]  # [B, T]

    contacts = []
    for name in FOOT_JOINT_NAMES:
        start = FOOT_XYZ_SLICES[name]
        foot_xyz = motion[:, start:start + 3, :]         # [B, 3, T]

        # Condition 1: Low velocity
        vel = foot_xyz[:, :, 1:] - foot_xyz[:, :, :-1]   # [B, 3, T-1]
        vel_sq = (vel ** 2).sum(dim=1)                    # [B, T-1]
        low_vel = vel_sq < vel_threshold                  # [B, T-1]
        # Pad frame 0 (assume contact to match DuetGen convention)
        low_vel = torch.cat([torch.ones(B, 1, device=motion.device, dtype=torch.bool), low_vel], dim=1)  # [B, T]

        # Condition 2: Low height (global = root_h + local_y)
        local_y = motion[:, FOOT_Y_INDICES[name], :]      # [B, T]
        global_y = root_h + local_y                        # [B, T]
        h_thresh = height_thresholds[name]
        low_height = global_y < h_thresh                   # [B, T]

        # Both conditions must hold
        contact = (low_vel & low_height).float()           # [B, T]
        contacts.append(contact)

    return torch.stack(contacts, dim=1)                    # [B, 4, T]


# =============================================================================
# Skating loss (corrected)
# =============================================================================
def compute_foot_skating_loss(motion, contact_mask, lengths=None):
    """
    Penalize horizontal (XZ) velocity of foot joints that are in ground contact.

    This is the CORRECTED version of DuetGen's foot_skating_loss(), which had
    an inverted masking bug (it zeroed velocities WHERE mask was True, meaning
    it penalized airborne feet instead of grounded ones).

    Correct logic: penalize horizontal velocity ONLY where contact_mask = 1.

    Args:
        motion: [B, 67, T] — decoded motion (channel-first)
        contact_mask: [B, 4, T] — foot contact mask from detect_foot_contact()
        lengths: [B] tensor or None — sequence lengths for padding mask

    Returns:
        loss: scalar — mean squared horizontal velocity of grounded feet
    """
    B, _, T = motion.shape

    skating_losses = []
    for i, name in enumerate(FOOT_JOINT_NAMES):
        dim_x, dim_z = FOOT_XZ_INDICES[name]

        # Horizontal velocity
        vel_x = motion[:, dim_x, 1:] - motion[:, dim_x, :-1]  # [B, T-1]
        vel_z = motion[:, dim_z, 1:] - motion[:, dim_z, :-1]  # [B, T-1]
        horiz_speed_sq = vel_x ** 2 + vel_z ** 2               # [B, T-1]

        # Contact at both frames of the velocity computation
        contact_t0 = contact_mask[:, i, :-1]                    # [B, T-1]
        contact_t1 = contact_mask[:, i, 1:]                     # [B, T-1]
        contact = contact_t0 * contact_t1                       # [B, T-1]

        # Penalize ONLY grounded feet (correct masking)
        skating_losses.append(horiz_speed_sq * contact)

    all_skating = torch.stack(skating_losses, dim=0).mean(dim=0)  # [B, T-1]

    if lengths is not None:
        from utils.train_utils import lengths_to_mask
        pad_mask = lengths_to_mask(lengths, T - 1).float()
        loss = (all_skating * pad_mask).sum() / (pad_mask.sum() + 1e-8)
    else:
        loss = all_skating.mean()

    return loss


# =============================================================================
# Option 2: Post-hoc foot-lock contact solver (inference-time)
# =============================================================================
@torch.no_grad()
def apply_foot_lock_solver(motion, contact_mask=None, lengths=None):
    """
    Post-hoc foot locking in local (root-relative) space.

    For each frame where a foot is grounded, replaces the foot's local XYZ
    position with that of the previous frame (velocity zeroing). This prevents
    local sliding without requiring root integration or IK.

    Causal: only looks backward, no future frames needed.

    Args:
        motion: [B, 67, T] — channel-first decoded motion
        contact_mask: [B, 4, T] float or None — if None, detected automatically
        lengths: [B] tensor or None — sequence lengths for masking

    Returns:
        corrected: [B, 67, T] — motion with foot positions locked during contact
    """
    corrected = motion.clone()
    B, _, T = motion.shape

    if contact_mask is None:
        contact_mask = detect_foot_contact(motion)  # [B, 4, T]

    for b in range(B):
        T_b = int(lengths[b].item()) if lengths is not None else T
        for fi, name in enumerate(FOOT_JOINT_NAMES):
            start = FOOT_XYZ_SLICES[name]
            for t in range(1, T_b):
                if contact_mask[b, fi, t] > 0.5:
                    corrected[b, start:start + 3, t] = corrected[b, start:start + 3, t - 1]

    return corrected


# =============================================================================
# Foot sinking loss (retained from train_MARDM.py for completeness)
# =============================================================================
def compute_foot_sinking_loss(motion, lengths, floor_y=0.0):
    """
    Penalize foot joints below the ground plane.
    Global foot height = root_height + local_foot_y.

    Args:
        motion: [B, 67, T] — decoded motion (channel-first)
        lengths: [B] tensor — sequence lengths for padding mask
        floor_y: float — ground plane height (default 0.0)

    Returns:
        loss: scalar — mean floor penetration across all foot joints
    """
    from utils.train_utils import lengths_to_mask

    B, _, T = motion.shape
    root_h = motion[:, ROOT_HEIGHT_IDX, :]  # [B, T]

    foot_losses = []
    for name in FOOT_JOINT_NAMES:
        local_y = motion[:, FOOT_Y_INDICES[name], :]  # [B, T]
        global_y = root_h + local_y                     # [B, T]
        penetration = F.relu(floor_y - global_y)        # only penalize below floor
        foot_losses.append(penetration)

    all_penetration = torch.stack(foot_losses, dim=0).mean(dim=0)  # [B, T]

    pad_mask = lengths_to_mask(lengths, T).float()
    loss = (all_penetration * pad_mask).sum() / (pad_mask.sum() + 1e-8)

    return loss