"""
Post-hoc motion smoothing utilities for inference-time use.

These are applied AFTER the learned trajectory refinement net, as a final
cleanup pass. They are NOT used during training.

Pipeline order at inference:
    1. dae.decode(latents) → raw motion [B, 67, T]
    2. TrajectoryRefinementNet → corrected root trajectory
    3. smooth_root_ema() → jitter removal on root
    4. clamp_floor_penetration() → hard floor constraint
    5. correct_foot_sliding() → pin grounded feet

Reference: DuetGen smooth_root_joint_deltas() and adjust_foot_sliding(),
    adapted for HumanML3D 67-dim representation.
"""

import torch
import torch.nn.functional as F


ROOT_HEIGHT_IDX = 3


def smooth_root_ema(motion, alpha_horizontal=0.5, alpha_vertical=0.1):
    """
    Exponential moving average smoothing on root trajectory deltas.

    Applies EMA to frame-to-frame position increments, then reconstructs
    positions via cumulative sum from the original starting position.

    Different smoothing strengths for horizontal (XZ velocity) and vertical
    (height): vertical gets much stronger smoothing because height bobbing
    is a common artifact and legitimate height variation is slower.

    Adapted from DuetGen's smooth_root_joint_deltas().

    NOTE: This operates on the root features in HumanML3D format:
        dim 0 = root angular velocity (Y-axis) — smoothed with alpha_horizontal
        dim 1 = root X velocity — smoothed with alpha_horizontal
        dim 2 = root Z velocity — smoothed with alpha_horizontal
        dim 3 = root height — smoothed with alpha_vertical (as deltas)

    Since dims 0-2 are already velocities (not positions), we apply EMA
    directly to them rather than to their deltas. Dim 3 (height) is a
    position, so we EMA its deltas and reconstruct.

    Args:
        motion: [B, 67, T] — decoded motion (channel-first)
        alpha_horizontal: float — EMA weight for velocity features (0-1).
            Higher = less smoothing (more responsive). Default 0.5.
        alpha_vertical: float — EMA weight for height deltas (0-1).
            Higher = less smoothing. Default 0.1 (aggressive).

    Returns:
        smoothed: [B, 67, T] — motion with smoothed root trajectory
    """
    smoothed = motion.clone()
    B, _, T = motion.shape

    if T < 3:
        return smoothed

    # --- Velocity features (dims 0, 1, 2): direct EMA ---
    vel_feats = motion[:, :3, :].clone()   # [B, 3, T]
    for t in range(1, T):
        vel_feats[:, :, t] = (
            alpha_horizontal * vel_feats[:, :, t]
            + (1 - alpha_horizontal) * vel_feats[:, :, t - 1]
        )
    smoothed[:, :3, :] = vel_feats

    # --- Height (dim 3): EMA on deltas, then reconstruct ---
    height = motion[:, ROOT_HEIGHT_IDX, :]           # [B, T]
    height_deltas = height[:, 1:] - height[:, :-1]   # [B, T-1]

    smoothed_deltas = height_deltas.clone()
    for t in range(1, T - 1):
        smoothed_deltas[:, t] = (
            alpha_vertical * smoothed_deltas[:, t]
            + (1 - alpha_vertical) * smoothed_deltas[:, t - 1]
        )

    # Reconstruct height from smoothed deltas + original start
    smoothed_height = torch.zeros_like(height)
    smoothed_height[:, 0] = height[:, 0]             # preserve start frame
    smoothed_height[:, 1:] = height[:, 0:1] + torch.cumsum(smoothed_deltas, dim=1)
    smoothed[:, ROOT_HEIGHT_IDX, :] = smoothed_height

    return smoothed


def clamp_floor_penetration(motion, floor_y=0.0):
    """
    Hard-clamp foot joints to be at or above the ground plane.

    If a foot joint's global height (root_h + local_y) is below floor_y,
    adjust the local_y upward so global_y = floor_y.

    This is a simple, non-learned correction that guarantees no foot sinking
    in the final output.

    Args:
        motion: [B, 67, T] — decoded motion (channel-first)
        floor_y: float — ground plane height (default 0.0)

    Returns:
        clamped: [B, 67, T] — motion with no sub-floor foot positions
    """
    from utils.foot_contact import FOOT_Y_INDICES, FOOT_JOINT_NAMES

    clamped = motion.clone()
    root_h = motion[:, ROOT_HEIGHT_IDX, :]  # [B, T]

    for name in FOOT_JOINT_NAMES:
        y_idx = FOOT_Y_INDICES[name]
        local_y = motion[:, y_idx, :]       # [B, T]
        global_y = root_h + local_y         # [B, T]

        # Where penetrating, adjust local_y so global_y = floor_y
        penetrating = global_y < floor_y
        corrected_local_y = floor_y - root_h   # what local_y needs to be
        clamped[:, y_idx, :] = torch.where(penetrating, corrected_local_y, local_y)

    return clamped


def correct_foot_sliding(motion, contact_mask, pin_threshold=0.01):
    """
    Pin grounded feet to their position at contact onset.

    Within each contiguous ground-contact segment for each foot joint,
    the foot's XZ position is locked to the position at the first frame
    of the segment. This eliminates sliding for feet that should be planted.

    This is a corrected version of DuetGen's adjust_foot_sliding(), which
    used an unreasonable 1.2 m/frame threshold and only lifted feet
    vertically. Our version directly pins the horizontal position.

    NOTE: This modifies the local joint positions, not the root. This means
    the kinematic chain is not strictly preserved — for visualization via
    SMPL, the joint positions are used directly, so this is fine. For
    skeleton-based rendering, the root would need adjustment instead.

    Args:
        motion: [B, 67, T] — decoded motion (channel-first)
        contact_mask: [B, 4, T] — foot contact mask from detect_foot_contact()
        pin_threshold: float — minimum contact segment length (frames) to
            trigger pinning. Very short contacts are likely noise.

    Returns:
        corrected: [B, 67, T] — motion with pinned foot positions
    """
    from utils.foot_contact import FOOT_XZ_INDICES, FOOT_JOINT_NAMES

    corrected = motion.clone()
    B, _, T = motion.shape

    for b in range(B):
        for i, name in enumerate(FOOT_JOINT_NAMES):
            dim_x, dim_z = FOOT_XZ_INDICES[name]
            contact = contact_mask[b, i, :]  # [T]

            # Find contiguous contact segments
            segments = _find_contact_segments(contact)

            for start, end in segments:
                seg_len = end - start
                if seg_len < 2:
                    continue

                # Pin XZ to position at contact onset
                pin_x = corrected[b, dim_x, start].clone()
                pin_z = corrected[b, dim_z, start].clone()
                corrected[b, dim_x, start:end] = pin_x
                corrected[b, dim_z, start:end] = pin_z

    return corrected


def _find_contact_segments(contact_1d):
    """
    Find contiguous segments where contact = 1.

    Args:
        contact_1d: [T] — binary contact signal for one foot joint, one batch

    Returns:
        segments: list of (start, end) tuples (end is exclusive)
    """
    segments = []
    T = contact_1d.shape[0]
    in_segment = False
    start = 0

    for t in range(T):
        if contact_1d[t] > 0.5 and not in_segment:
            start = t
            in_segment = True
        elif contact_1d[t] <= 0.5 and in_segment:
            segments.append((start, t))
            in_segment = False

    if in_segment:
        segments.append((start, T))

    return segments


def full_refinement_pipeline(
    motion,
    refine_net=None,
    smooth=True,
    clamp_floor=True,
    fix_sliding=True,
    alpha_horizontal=0.5,
    alpha_vertical=0.1,
    floor_y=0.0,
):
    """
    Complete inference-time refinement pipeline.

    Applies all corrections in the correct order:
        1. Learned trajectory refinement (if refine_net provided)
        2. EMA root smoothing (if smooth=True)
        3. Floor penetration clamping (if clamp_floor=True)
        4. Foot sliding correction (if fix_sliding=True)

    Args:
        motion: [B, 67, T] — raw decoded motion
        refine_net: TrajectoryRefinementNet or None
        smooth: bool — apply EMA root smoothing
        clamp_floor: bool — apply floor penetration clamping
        fix_sliding: bool — apply foot sliding correction
        alpha_horizontal: float — EMA alpha for horizontal root features
        alpha_vertical: float — EMA alpha for height
        floor_y: float — ground plane height

    Returns:
        refined: [B, 67, T] — fully refined motion
    """
    from utils.foot_contact import detect_foot_contact
    from models.refinement import build_refinement_input

    refined = motion

    # Step 1: Learned trajectory refinement
    if refine_net is not None:
        with torch.no_grad():
            refine_input = build_refinement_input(refined)  # [B, 130, T]
            root_pred = refine_net(refine_input)             # [B, 4, T]
            refined = refined.clone()
            refined[:, :4, :] = root_pred

    # Step 2: EMA root smoothing
    if smooth:
        refined = smooth_root_ema(
            refined,
            alpha_horizontal=alpha_horizontal,
            alpha_vertical=alpha_vertical,
        )

    # Step 3: Floor penetration clamping
    if clamp_floor:
        refined = clamp_floor_penetration(refined, floor_y=floor_y)

    # Step 4: Foot sliding correction
    if fix_sliding:
        contact = detect_foot_contact(refined)
        refined = correct_foot_sliding(refined, contact)

    return refined