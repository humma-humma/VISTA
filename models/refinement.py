"""
Trajectory Refinement Module for MM-MARDM.

Learns the mapping: body-local features → root trajectory.
Trained on GT motions (Phase 1) and fine-tuned on generated motions (Phase 2).

At inference, replaces the noisy generated root trajectory with the predicted
one based on the (higher-quality) generated body-local features.

Architecture: Dilated 1D convolutional residual stack.
    - No encoder-decoder bottleneck, no downsampling
    - Exponential dilation growth → large receptive field for temporal coherence
    - Full temporal resolution preserved throughout

Reference: DuetGen (SIGGRAPH 2025) Global_Trajectory_Pred, adapted for
    single-person HumanML3D 67-dim representation.
"""

import torch
import torch.nn as nn


class DilatedResBlock(nn.Module):
    """
    Residual block with dilated 1D convolutions.

    Structure: LayerNorm → Conv1d (dilated) → SiLU → Conv1d (dilated) → + skip

    Matches the residual block pattern used in the DAE encoder
    (Conv1d → LayerNorm → SiLU → Conv1d → + skip) with dilation for
    expanded receptive fields.
    """

    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        padding = dilation  # same-padding for kernel_size=3

        self.net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=padding),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=padding),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, T] — channel-first temporal signal

        Note: LayerNorm expects [B, T, C], so we permute in/out.
        """
        residual = x
        # LayerNorm on channel dim: permute to [B, T, C], norm, permute back
        h = x.permute(0, 2, 1)               # [B, T, C]
        h = self.net[0](h)                    # LayerNorm
        h = h.permute(0, 2, 1)               # [B, C, T]
        h = self.net[1:](h)                  # Conv → SiLU → Dropout → Conv → Dropout
        return residual + h


class TrajectoryRefinementNet(nn.Module):
    """
    Predicts refined root trajectory (and optionally foot corrections) from
    body-local motion features.

    refine_mode='root_only'  (default, Option 1):
        Output: [B, 4, T] — root features only
        output_feats=4

    refine_mode='root_foot'  (Option 3b):
        Output: [B, 16, T]
            [:, 0:4,  :] — root trajectory features
            [:, 4:16, :] — foot XYZ position deltas (4 joints × 3)
        output_feats=16

    Input (130 dims = 63 joint pos + 63 joint vel + 4 foot contact):
        GT branch  : GT velocities + GT contacts from 263-dim
        Gen branch : finite-diff velocities + zeros (Option 1) or detected contacts
    """

    # XYZ slice starts for 4 foot joints within the 67-dim motion (FOOT_XYZ_SLICES)
    FOOT_DIMS = [22, 25, 31, 34]  # l_ankle, r_ankle, l_foot, r_foot start indices

    def __init__(
        self,
        input_feats=130,
        output_feats=4,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        dropout=0.1,
        refine_mode='root_only',
    ):
        super().__init__()

        self.input_feats = input_feats
        self.refine_mode = refine_mode

        # Force correct output_feats based on mode
        if refine_mode == 'root_foot':
            output_feats = 16   # 4 root + 4 joints × 3
        else:
            output_feats = 4
        self.output_feats = output_feats

        self.input_proj = nn.Conv1d(input_feats, width, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList([
            DilatedResBlock(
                channels=width,
                dilation=dilation_growth_rate ** i,
                dropout=dropout,
            )
            for i in range(depth)
        ])

        # Shared output projection head
        self.output_proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, output_feats),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init so predictions start as identity (no correction)
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x):
        """
        Args:
            x: [B, input_feats, T]

        Returns:
            If refine_mode='root_only': root_pred [B, 4, T]
            If refine_mode='root_foot': (root_pred [B, 4, T], foot_delta [B, 12, T])
        """
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)

        h = h.permute(0, 2, 1)           # [B, T, width]
        out = self.output_proj(h)         # [B, T, output_feats]
        out = out.permute(0, 2, 1)        # [B, output_feats, T]

        if self.refine_mode == 'root_foot':
            return out[:, :4, :], out[:, 4:, :]   # root [B,4,T], foot_delta [B,12,T]
        return out  # [B, 4, T]

    def apply_to_motion(self, motion, root_pred, foot_delta=None):
        """
        Splice predictions back into a [B, 67, T] motion tensor.

        Args:
            motion: [B, 67, T]
            root_pred: [B, 4, T]
            foot_delta: [B, 12, T] or None (only for root_foot mode)

        Returns:
            refined: [B, 67, T]
        """
        refined = motion.clone()
        refined[:, :4, :] = root_pred

        if foot_delta is not None:
            for i, start in enumerate(self.FOOT_DIMS):
                refined[:, start:start + 3, :] = (
                    motion[:, start:start + 3, :] + foot_delta[:, i * 3:(i + 1) * 3, :]
                )
        return refined


def apply_refinement(motion, refine_net, use_contact_solver=False):
    """
    Convenience inference wrapper: build input features, run refinement net,
    splice predictions back, and optionally apply foot-lock solver (Option 2).

    Args:
        motion: [B, 67, T] — decoded motion (channel-first)
        refine_net: TrajectoryRefinementNet instance
        use_contact_solver: bool — apply post-hoc foot locking after refinement

    Returns:
        refined_motion: [B, 67, T]
    """
    from utils.foot_contact import compute_velocities, apply_foot_lock_solver

    body_local = motion[:, 4:, :]
    velocities = compute_velocities(body_local)
    contacts = torch.zeros(motion.shape[0], 4, motion.shape[2], device=motion.device)
    input_feats = torch.cat([body_local, velocities, contacts], dim=1)

    out = refine_net(input_feats)
    if refine_net.refine_mode == 'root_foot':
        root_pred, foot_delta = out
    else:
        root_pred, foot_delta = out, None

    refined = refine_net.apply_to_motion(motion, root_pred, foot_delta)

    if use_contact_solver:
        refined = apply_foot_lock_solver(refined)

    return refined


def build_refinement_input(motion, gt_velocities=None, gt_contacts=None):
    """
    Build the 130-dim input tensor for the refinement net.

    During Phase 1 (GT training): pass gt_velocities and gt_contacts from
        the 263-dim representation.
    During Phase 2 (generated fine-tuning) and inference: pass None for both,
        and velocities/contacts will be computed from the motion.

    Args:
        motion: [B, 67, T] — decoded motion (channel-first)
        gt_velocities: [B, 63, T] or None — GT joint velocities from 263-dim
        gt_contacts: [B, 4, T] or None — GT foot contacts from 263-dim

    Returns:
        input_feats: [B, 130, T] — concatenated features for refinement net
    """
    from utils.foot_contact import compute_velocities, detect_foot_contact

    body_local = motion[:, 4:, :]  # [B, 63, T] — joint positions

    # Velocities
    if gt_velocities is not None:
        velocities = gt_velocities  # [B, 63, T]
    else:
        velocities = compute_velocities(body_local)  # [B, 63, T]

    # Contacts
    if gt_contacts is not None:
        contacts = gt_contacts  # [B, 4, T]
    else:
        contacts = detect_foot_contact(motion)  # [B, 4, T]

    return torch.cat([body_local, velocities, contacts], dim=1)  # [B, 130, T]