"""
Standalone training script for the Trajectory Refinement Module.

Single-stage unified training. Each iteration, with probability p_gt (schedule
below), the batch is drawn from the GT distribution (clean body-local features
+ GT velocities + GT contacts). Otherwise, the batch is drawn from the generated
distribution (MARDM+DAE output + finite-diff velocities + detected contacts).
The target is always the GT root trajectory.

Mixing schedule (batch-level Bernoulli):
    iteration <  warmup_iters : p_gt = 1.0            (pure GT warmup)
    iteration >= warmup_iters : p_gt = mix_gt_ratio   (constant mix)

Loss design (adapted from DuetGen's GlobalTrajectoryTrainer):
    - Root reconstruction: SmoothL1(pred_root, gt_root)
    - Velocity matching: SmoothL1 on temporal differences at curriculum strides
      (stride-20 for first 20k iters, then stride-5)
    - Contact-aware skating: penalize horizontal velocity of grounded feet

Usage:
    python train_refinement.py --dataset_dir ./datasets \\
        --model_dir ./checkpoints/100styles/MARDM_.../model \\
        --warmup_iters 20000 --mix_gt_ratio 0.3 --epoch 300

    The model_dir must contain `final.tar` (MARDM) and `dae_final.tar` (DAE).
"""

import os
from os.path import join as pjoin
import argparse
import time
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import wandb

from models.refinement import TrajectoryRefinementNet, build_refinement_input
from utils.foot_contact import (
    detect_foot_contact,
    compute_foot_skating_loss,
    compute_velocities,
)
from utils.train_utils import (
    lengths_to_mask,
    update_lr_warm_up,
    def_value,
    print_current_loss,
    build_training_schedule,
)
from transformers import VivitModel, VivitImageProcessor


# =============================================================================
# HumanML3D 263-dim feature slicing
# =============================================================================
# Full 263-dim layout (from HumanML3D codebase, 22 joints SMPL skeleton):
#   [0]         root angular velocity (Y-axis)           1
#   [1:3]       root linear velocity (X, Z)              2
#   [3]         root height                              1
#   [4:67]      local joint positions (21 × 3)           63    ← your 67-dim ends here
#   [67:130]    local joint velocities (21 × 3)          63
#   [130:256]   local joint rotations (21 × 6)           126
#   [256:260]   foot contact labels (4 binary)           4
#                                                        ---
#                                                        260  (NOT 263?)
#
# WARNING: The exact foot contact offset depends on your HumanML3D version.
# Some versions use 22×3=66 for positions (including root), shifting everything.
# RUN THIS TO VERIFY before training:
#
#   m = np.load('./datasets/HumanML3D/new_joint_vecs/000000.npy')
#   print(f"Shape: {m.shape}")
#   for s in [256, 259]:
#       print(f"  dims [{s}:{s+4}] @ frame 10: {m[10, s:s+4]}")
#       print(f"    unique values: {np.unique(m[:, s:s+4].round(1))}")
#
# The foot contacts should have values clustered near 0.0 and 1.0.

SLICE_ROOT = slice(0, 4)               # root features (target)
SLICE_JOINT_POS = slice(4, 67)         # local joint positions (21×3)
SLICE_JOINT_VEL = slice(67, 130)       # local joint velocities (21×3)
SLICE_JOINT_ROT = slice(130, 259)      # local joint rotations (unused)
SLICE_FOOT_CONTACT = slice(259, 263)   # foot contacts — VERIFY THIS


# =============================================================================
# Dataset for Phase 1: GT motions with 263-dim features
# =============================================================================
class RefinementGTDataset(Dataset):
    """
    Loads GT motions in the full 263-dim HumanML3D representation.

    Uses the SLICED split files (e.g., splits_sliced/train.txt) which contain
    entries like "000000_0", "000000_1". Parses these to recover the original
    filename and text annotation index, then loads the full 263-dim file from
    new_joint_vecs/ and applies the same temporal slicing as hml3d_encoder.py.

    Extracts:
        - Joint positions [63] (input)
        - Joint velocities [63] (input)
        - Foot contacts [4] (input)
        - Root features [4] (target)
    """

    def __init__(self, motion_dir, text_dir, split_file, mean_path, std_path,
                 max_motion_length=196, unit_length=4, fps=20.0):
        self.motion_dir = motion_dir    # new_joint_vecs/ (263-dim originals)
        self.text_dir = text_dir        # texts/ (original text annotations)
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.fps = fps

        # Load normalization stats (full 263-dim)
        self.mean = np.load(mean_path)           # [263]
        self.std = np.load(std_path)             # [263]

        # Load sliced split file (entries like "000000_0")
        with open(split_file, 'r') as f:
            raw_entries = [line.strip() for line in f if line.strip()]

        # Parse each entry and verify the source file exists
        self.entries = []
        for entry in raw_entries:
            # Parse "000000_0" → base_name="000000", annotation_idx=0
            parts = entry.rsplit('_', 1)
            if len(parts) != 2:
                continue
            base_name, idx_str = parts
            try:
                ann_idx = int(idx_str)
            except ValueError:
                continue

            motion_path = pjoin(motion_dir, base_name + '.npy')
            text_path = pjoin(text_dir, base_name + '.txt')
            if os.path.exists(motion_path) and os.path.exists(text_path):
                self.entries.append((base_name, ann_idx, motion_path, text_path))

        print(f"RefinementGTDataset: {len(self.entries)} sliced motions from {split_file}")

    def __len__(self):
        return len(self.entries)

    def _parse_slice_bounds(self, text_path, ann_idx, full_length):
        """
        Read the text file and extract temporal bounds for the given annotation.
        Matches the slicing logic in hml3d_encoder.py lines 57-80.
        """
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if ann_idx >= len(lines):
            return 0, full_length

        line = lines[ann_idx].strip()
        parts = line.split('#')

        start_frame = 0
        end_frame = full_length

        if len(parts) >= 4:
            try:
                f_tag = float(parts[2])
                to_tag = float(parts[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                if f_tag != 0.0 or to_tag != 0.0:
                    start_frame = int(f_tag * self.fps) if f_tag < 100 else int(f_tag)
                    end_frame = int(to_tag * self.fps) if to_tag < 100 else int(to_tag)
            except ValueError:
                pass

        safe_start = max(0, min(start_frame, full_length - 1))
        safe_end = max(safe_start + 1, min(end_frame, full_length))

        return safe_start, safe_end

    def __getitem__(self, idx):
        base_name, ann_idx, motion_path, text_path = self.entries[idx]

        # Load full 263-dim motion
        motion_full = np.load(motion_path)  # [T_full, 263]
        full_length = motion_full.shape[0]

        # Apply the same temporal slicing as hml3d_encoder.py
        start, end = self._parse_slice_bounds(text_path, ann_idx, full_length)
        motion_263 = motion_full[start:end]  # [T_slice, 263]

        T = motion_263.shape[0]

        # Crop if too long
        if T > self.max_motion_length:
            motion_263 = motion_263[:self.max_motion_length]
            T = self.max_motion_length

        # Align to unit length
        T = T - (T % self.unit_length)
        if T < self.unit_length:
            T = self.unit_length
        motion_263 = motion_263[:T]

        # Extract raw binary foot contacts BEFORE normalization (for skating loss)
        raw_contacts = torch.from_numpy(motion_263[:, SLICE_FOOT_CONTACT].copy()).float()  # [T, 4] — binary 0/1

        # Normalize (using full 263-dim stats)
        motion_norm = (motion_263 - self.mean[:263]) / (self.std[:263] + 1e-8)

        # Extract features (normalized — for network input)
        motion_t = torch.from_numpy(motion_norm).float()  # [T, 263]

        root_feats = motion_t[:, SLICE_ROOT]            # [T, 4]
        joint_pos = motion_t[:, SLICE_JOINT_POS]        # [T, 63]
        joint_vel = motion_t[:, SLICE_JOINT_VEL]        # [T, 63]
        foot_contact = motion_t[:, SLICE_FOOT_CONTACT]  # [T, 4] — normalized, for net input

        # Also extract the 67-dim subset (for skating loss computation)
        motion_67 = motion_t[:, :67]                    # [T, 67]

        return {
            'root_feats': root_feats,           # [T, 4] — target
            'joint_pos': joint_pos,             # [T, 63] — input
            'joint_vel': joint_vel,             # [T, 63] — input (GT)
            'foot_contact': foot_contact,       # [T, 4] — input (normalized, for network)
            'raw_contacts': raw_contacts,       # [T, 4] — binary 0/1 (for skating loss)
            'motion_67': motion_67,             # [T, 67] — for skating loss
            'length': T,
        }

class Refinement100StylesDataset(Dataset):
    """
    Loads 100STYLES GT motions (67-dim only).
    
    Velocities computed via finite-differencing (no GT available).
    No GT foot contacts available — skating loss skipped for these samples.
    """

    def __init__(self, motion_dir, split_file, mean_path, std_path,
                 max_motion_length=196, unit_length=4, window_size=64):
        self.motion_dir = motion_dir
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.window_size = window_size

        # Load normalization stats (100STYLES, 67-dim)
        self.mean = np.load(mean_path)[:67]
        self.std = np.load(std_path)[:67]

        # Load split
        with open(split_file, 'r') as f:
            self.file_list = [line.strip().split()[0] for line in f if line.strip()]

        self.file_list = [
            f for f in self.file_list
            if os.path.exists(pjoin(motion_dir, f + '.npy'))
        ]
        print(f"Refinement100StylesDataset: {len(self.file_list)} motions from {split_file}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        motion_raw = np.load(pjoin(self.motion_dir, fname + '.npy'))[:, :67]  # [T_full, 67]

        T_full = motion_raw.shape[0]

        # Random window (same as AEVideoDataset_100styles_v3)
        if T_full > self.window_size:
            start = random.randint(0, T_full - self.window_size)
            motion_raw = motion_raw[start:start + self.window_size]
        T = motion_raw.shape[0]

        # Align to unit length
        T = T - (T % self.unit_length)
        if T < self.unit_length:
            T = self.unit_length
        motion_raw = motion_raw[:T]

        # Normalize
        motion_norm = (motion_raw - self.mean) / (self.std + 1e-8)
        motion_t = torch.from_numpy(motion_norm).float()  # [T, 67]

        # Extract features
        root_feats = motion_t[:, :4]        # [T, 4] — target
        joint_pos = motion_t[:, 4:]         # [T, 63] — input

        # Compute velocities via finite-differencing (no GT available)
        vel = joint_pos[1:] - joint_pos[:-1]                    # [T-1, 63]
        vel = torch.cat([torch.zeros(1, 63), vel], dim=0)       # [T, 63]

        return {
            'root_feats': root_feats,       # [T, 4]
            'joint_pos': joint_pos,         # [T, 63]
            'joint_vel': vel,               # [T, 63] — finite-diff
            'motion_67': motion_t,          # [T, 67]
            'length': T,
        }


def collate_refinement(batch):
    """Collate with padding to max length in batch."""
    max_len = max(item['length'] for item in batch)
    B = len(batch)

    root_feats = torch.zeros(B, max_len, 4)
    joint_pos = torch.zeros(B, max_len, 63)
    joint_vel = torch.zeros(B, max_len, 63)
    foot_contact = torch.zeros(B, max_len, 4)
    raw_contacts = torch.zeros(B, max_len, 4)
    motion_67 = torch.zeros(B, max_len, 67)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        T = item['length']
        root_feats[i, :T] = item['root_feats']
        joint_pos[i, :T] = item['joint_pos']
        joint_vel[i, :T] = item['joint_vel']
        foot_contact[i, :T] = item['foot_contact']
        raw_contacts[i, :T] = item['raw_contacts']
        motion_67[i, :T] = item['motion_67']
        lengths[i] = T

    return {
        'root_feats': root_feats,       # [B, T, 4]
        'joint_pos': joint_pos,         # [B, T, 63]
        'joint_vel': joint_vel,         # [B, T, 63]
        'foot_contact': foot_contact,   # [B, T, 4] — normalized (network input)
        'raw_contacts': raw_contacts,   # [B, T, 4] — binary 0/1 (skating loss)
        'motion_67': motion_67,         # [B, T, 67]
        'lengths': lengths,             # [B]
    }

def collate_100styles(batch):
    """Collate for 100STYLES (no GT contacts)."""
    max_len = max(item['length'] for item in batch)
    B = len(batch)

    root_feats = torch.zeros(B, max_len, 4)
    joint_pos = torch.zeros(B, max_len, 63)
    joint_vel = torch.zeros(B, max_len, 63)
    motion_67 = torch.zeros(B, max_len, 67)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        T = item['length']
        root_feats[i, :T] = item['root_feats']
        joint_pos[i, :T] = item['joint_pos']
        joint_vel[i, :T] = item['joint_vel']
        motion_67[i, :T] = item['motion_67']
        lengths[i] = T

    return {
        'root_feats': root_feats,
        'joint_pos': joint_pos,
        'joint_vel': joint_vel,
        'motion_67': motion_67,
        'lengths': lengths,
    }


# =============================================================================
# Loss computation
# =============================================================================
def compute_refinement_losses(
    pred_root, gt_root, motion_pred_67, lengths, iteration,
    recon_weight=1.0, vel_weight=10.0, skate_weight=0.5,
    gt_contacts=None,
    skate_mode='gt_only',
):
    """
    Compute root-trajectory refinement losses.

    skate_mode controls skating loss behaviour:
        'none'    — Option 1: skating loss always 0 (no gradient, no noise)
        'gt_only' — skating loss only when gt_contacts provided (GT branch only)
        'full'    — skating loss always; uses detect_foot_contact for gen branch

    NOTE: in 'root_only' refine_mode the skating loss gradient does NOT flow
    through pred_root (foot dims are fixed). It is only useful in 'root_foot'
    mode or as a diagnostic. Prefer 'none' for 'root_only' training.
    """
    B, _, T = pred_root.shape
    pad_mask = lengths_to_mask(lengths, T).float()

    # --- Reconstruction ---
    recon_per_frame = F.smooth_l1_loss(pred_root, gt_root, reduction='none').mean(dim=1)
    recon_loss = (recon_per_frame * pad_mask).sum() / (pad_mask.sum() + 1e-8)

    # --- Velocity (curriculum stride) ---
    stride = 20 if iteration < 20000 else 5
    if T > stride:
        vel_pred = pred_root[:, :, stride:] - pred_root[:, :, :-stride]
        vel_gt   = gt_root[:, :, stride:]   - gt_root[:, :, :-stride]
        vel_pad  = lengths_to_mask(lengths, T - stride).float()
        vel_per_frame = F.smooth_l1_loss(vel_pred, vel_gt, reduction='none').mean(dim=1)
        vel_loss = (vel_per_frame * vel_pad).sum() / (vel_pad.sum() + 1e-8)
    else:
        vel_loss = torch.tensor(0.0, device=pred_root.device)

    # --- Skating ---
    zero = torch.tensor(0.0, device=pred_root.device)
    if skate_mode == 'none' or skate_weight <= 0:
        skate_loss = zero
    elif skate_mode == 'gt_only':
        skate_loss = (
            compute_foot_skating_loss(motion_pred_67, gt_contacts, lengths)
            if gt_contacts is not None else zero
        )
    else:  # 'full'
        contacts = gt_contacts if gt_contacts is not None else detect_foot_contact(motion_pred_67)
        skate_loss = compute_foot_skating_loss(motion_pred_67, contacts, lengths)

    total_loss = recon_weight * recon_loss + vel_weight * vel_loss + skate_weight * skate_loss

    loss_dict = {
        'recon': recon_loss.item(),
        'vel':   vel_loss.item(),
        'skate': skate_loss.item(),
        'total': total_loss.item(),
    }
    return total_loss, loss_dict


# foot-correction loss used only in refine_mode='root_foot' (Option 3b)
FOOT_XYZ_STARTS = [22, 25, 31, 34]  # l_ankle, r_ankle, l_foot, r_foot

def compute_foot_correction_losses(
    motion_pred_67, motion_gt_67, lengths,
    gt_contacts=None,
    foot_recon_weight=1.0, skate_weight=0.5,
):
    """
    Additional losses for refine_mode='root_foot' (Option 3b).

    motion_pred_67 already has foot deltas applied by apply_to_motion().
    Gradient flows through the corrected foot positions → skating loss is meaningful.
    """
    _, _, T = motion_pred_67.shape
    pad_mask = lengths_to_mask(lengths, T).float()

    # Foot reconstruction: penalize corrected positions vs GT foot positions
    foot_recon_loss = torch.tensor(0.0, device=motion_pred_67.device)
    for start in FOOT_XYZ_STARTS:
        pred_foot = motion_pred_67[:, start:start + 3, :]
        gt_foot   = motion_gt_67[:, start:start + 3, :]
        per_frame = F.smooth_l1_loss(pred_foot, gt_foot, reduction='none').mean(dim=1)
        foot_recon_loss = foot_recon_loss + (per_frame * pad_mask).sum() / (pad_mask.sum() + 1e-8)
    foot_recon_loss = foot_recon_loss / len(FOOT_XYZ_STARTS)

    # Skating loss on corrected positions — gradient flows through foot_delta
    contacts = gt_contacts if gt_contacts is not None else detect_foot_contact(motion_pred_67)
    skate_loss = compute_foot_skating_loss(motion_pred_67, contacts, lengths)

    total = foot_recon_weight * foot_recon_loss + skate_weight * skate_loss
    loss_dict = {
        'foot_recon': foot_recon_loss.item(),
        'foot_skate': skate_loss.item(),
        'foot_total': total.item(),
    }
    return total, loss_dict


# =============================================================================
# Training loop
# =============================================================================
def get_p_gt(iteration, warmup_iters, mix_gt_ratio):
    """Probability of drawing a GT-input batch at the given iteration."""
    if iteration < warmup_iters:
        return 1.0
    return mix_gt_ratio


def train_unified(args):
    """
    Single-stage unified training for TrajectoryRefinementNet.

    Per iteration, with probability p_gt the batch is drawn from the GT
    distribution (clean body-local features, GT velocities, GT contacts).
    Otherwise it is drawn from the generated distribution (MARDM+DAE output,
    finite-diff velocities, detected contacts). Target is always the GT root.

    Schedule (batch-level Bernoulli, no intra-batch mixing):
        iteration <  warmup_iters : p_gt = 1.0            (pure GT warmup)
        iteration >= warmup_iters : p_gt = mix_gt_ratio   (constant mix)
    """
    wandb.init(
        project='Multimodal MARDM Refinement',
        name=args.exp_name,
        config=vars(args),
    )

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # GT-branch datasets -------------------------------------------------
    data_root = pjoin(args.dataset_dir, 'HumanML3D')
    gt_hml3d_train = RefinementGTDataset(
        motion_dir=pjoin(data_root, 'new_joint_vecs'),
        text_dir=pjoin(data_root, 'texts'),
        split_file=pjoin(data_root, 'splits_sliced/train.txt'),
        mean_path=pjoin(data_root, 'Mean.npy'),
        std_path=pjoin(data_root, 'Std.npy'),
        max_motion_length=args.max_motion_length,
    )
    gt_hml3d_val = RefinementGTDataset(
        motion_dir=pjoin(data_root, 'new_joint_vecs'),
        text_dir=pjoin(data_root, 'texts'),
        split_file=pjoin(data_root, 'splits_sliced/val.txt'),
        mean_path=pjoin(data_root, 'Mean.npy'),
        std_path=pjoin(data_root, 'Std.npy'),
        max_motion_length=args.max_motion_length,
    )
    style_root = pjoin(args.dataset_dir, '100STYLE-SMPL')
    gt_styles_train = Refinement100StylesDataset(
        motion_dir=pjoin(style_root, 'new_joint_vecs'),
        split_file=pjoin(style_root, 'train_100STYLE_Full.txt'),
        mean_path=pjoin(style_root, 'Mean.npy'),
        std_path=pjoin(style_root, 'Std.npy'),
        max_motion_length=args.max_motion_length,
        window_size=args.window_size,
    )
    gt_hml3d_loader = DataLoader(
        gt_hml3d_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=collate_refinement, pin_memory=True,
    )
    gt_hml3d_val_loader = DataLoader(
        gt_hml3d_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
        collate_fn=collate_refinement, pin_memory=True,
    )
    gt_styles_loader = DataLoader(
        gt_styles_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=collate_100styles, pin_memory=True,
    )

    # Generated-branch dataset -------------------------------------------
    from utils.datasets import Text2MotionDatasetCombined_v4, mld_collate_paired
    from models.AE import DAE_models
    from models.MARDM import MARDM_models

    dim_pose = 67
    style_mean = np.load(pjoin(style_root, 'Mean.npy'))
    style_std = np.load(pjoin(style_root, 'Std.npy'))
    prior_mean = np.load(pjoin(data_root, 'Mean.npy'))
    prior_std = np.load(pjoin(data_root, 'Std.npy'))

    gen_train = Text2MotionDatasetCombined_v4(
        style_mean=style_mean, style_std=style_std,
        style_split_file=pjoin(style_root, 'train_100STYLE_Full.txt'),
        style_motion_dir=pjoin(style_root, 'new_joint_vecs'),
        style_text_dir=pjoin(style_root, 'texts'),
        style_video_dir=pjoin(style_root, 'videos'),
        style_dict_file=pjoin(style_root, '100STYLE_name_dict_length.txt'),
        humanml_mean=prior_mean, humanml_std=prior_std,
        humanml_split_file=pjoin(data_root, 'splits_sliced/train.txt'),
        humnaml_motion_dir=pjoin(data_root, 'sliced_joint_vecs'),
        humanml_latent_dir=pjoin(data_root, 'latent_vecs'),
        humanml_text_dir=pjoin(data_root, 'splits_sliced/texts_sliced'),
        humanml_dict_file=pjoin(data_root, 'splits_sliced/all_lengths.txt'),
        dim_pose=dim_pose, unit_length=4, max_motion_length=args.max_motion_length,
        epoch_mode='100styles', tiny=True,
    )
    gen_val_full = Text2MotionDatasetCombined_v4(
        style_mean=style_mean, style_std=style_std,
        style_split_file=pjoin(style_root, 'test_100STYLE_Full.txt'),
        style_motion_dir=pjoin(style_root, 'new_joint_vecs'),
        style_text_dir=pjoin(style_root, 'texts'),
        style_video_dir=pjoin(style_root, 'videos'),
        style_dict_file=pjoin(style_root, '100STYLE_name_dict_length.txt'),
        humanml_mean=prior_mean, humanml_std=prior_std,
        humanml_split_file=pjoin(data_root, 'splits_sliced/val.txt'),
        humnaml_motion_dir=pjoin(data_root, 'sliced_joint_vecs'),
        humanml_latent_dir=pjoin(data_root, 'latent_vecs'),
        humanml_text_dir=pjoin(data_root, 'splits_sliced/texts_sliced'),
        humanml_dict_file=pjoin(data_root, 'splits_sliced/all_lengths.txt'),
        dim_pose=dim_pose, unit_length=4, max_motion_length=args.max_motion_length,
        epoch_mode='100styles', tiny=True,
    )
    val_size = len(gen_val_full) * 2 // 3
    test_size = len(gen_val_full) - val_size
    gen_val, _ = torch.utils.data.random_split(
        gen_val_full, [val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    gen_train_loader = DataLoader(
        gen_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=mld_collate_paired, pin_memory=True,
    )
    gen_val_loader = DataLoader(
        gen_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
        collate_fn=mld_collate_paired, pin_memory=True,
    )
    print(f"GT-HML3D train/val: {len(gt_hml3d_train)}/{len(gt_hml3d_val)}")
    print(f"GT-Styles train: {len(gt_styles_train)}")
    print(f"Gen combined train/val: {len(gen_train)}/{len(gen_val)}")

    # Training schedule (LR warmup + milestone LR drop) ------------------
    steps_per_epoch = len(gen_train_loader)
    schedule = build_training_schedule(args.epoch, steps_per_epoch)
    total_iters = schedule['total_iters']
    print(f"[schedule] total_iters={total_iters} | "
          f"lr_warmup={schedule['warm_up_iter']} iters "
          f"({schedule['warm_up_iter'] // max(steps_per_epoch,1)} ep) | "
          f"lr_drop at iter {schedule['lr_milestones'][0]} "
          f"(ep {schedule['lr_milestones'][0] // max(steps_per_epoch,1)})")

    # Frozen MARDM + DAE + ViViT (used only on gen-branch iterations) ----
    assert args.model_dir is not None, "unified training requires --model_dir"
    # dae_ckpt_path = pjoin(args.checkpoints_dir, '100styles', args.model_dir, 'model', 'dae_final.tar')
    # mardm_ckpt_path = pjoin(args.checkpoints_dir, '100styles', args.model_dir, 'model', 'final.tar')

    dae_ckpt_path = pjoin(args.checkpoints_dir, '100styles', 'DAE', 'final_finetune_diffmlp_decoder_fixed_hybrid_ema_fix.tar')
    # mardm_ckpt_path = pjoin(args.checkpoints_dir, 't2m', 'MARDM-DDPM-XL', 'model', 'final_diffmlp_diff_v4_cfg_cross_batch_hybrid_ema_fix_best_fid.tar')
    mardm_ckpt_path = pjoin(args.checkpoints_dir, 't2m', 'MARDM-DDPM-XL', 'model', 'final_diffmlp_diff_v4_cfg_cross_batch_hybrid_ema_fix.tar')

    num_classes = len(args.styles) if hasattr(args, 'styles') and args.styles else 5
    dae = DAE_models[args.ae_model](
        window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose,
    )
    # dae_ckpt = torch.load(dae_ckpt_path, map_location=device, weights_only=False)
    dae_ckpt = torch.load(dae_ckpt_path, map_location=device, weights_only=False)
    dae.load_state_dict(dae_ckpt['ae'])
    for p in dae.parameters():
        p.requires_grad = False
    dae.to(device).eval()
    print(f"DAE loaded from {dae_ckpt_path} (frozen)")

    mardm = MARDM_models[args.mardm_model](
        ae_dim=dae.output_emb_width,
        cond_mode='text',
        style_routing=args.style_routing,
        style_dim=512,
    )
    mardm_ckpt = torch.load(mardm_ckpt_path, map_location=device, weights_only=False)
    mardm.load_state_dict(mardm_ckpt['mardm'], strict=False)
    for p in mardm.parameters():
        p.requires_grad = False
    mardm.to(device).eval()
    print(f"MARDM loaded from {mardm_ckpt_path} (frozen)")

    print("Loading ViViT for video style extraction...")
    processor = VivitImageProcessor.from_pretrained('google/vivit-b-16x2-kinetics400')
    vmodel = VivitModel.from_pretrained('google/vivit-b-16x2-kinetics400').to(device)
    vmodel.eval()
    for p in vmodel.parameters():
        p.requires_grad = False
    print("ViViT loaded (frozen)")

    if args.style_routing == 'diffmlp':
        num_blocks = mardm.DiffMLPs.get_total_blocks()
        w_schedule = np.linspace(0.0, 1.0, num_blocks).tolist() if args.use_weight_schedule else [1.0] * num_blocks
    else:
        num_blocks = len(mardm.MARTransformer)
        w_schedule = [1.0] * num_blocks

    # Refinement net -----------------------------------------------------
    model = TrajectoryRefinementNet(
        input_feats=130,
        width=args.width, depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate, dropout=args.dropout,
        refine_mode=args.refine_mode,
    ).to(device)

    start_epoch = 0
    it = 0
    if args.refine_ckpt:
        ckpt = torch.load(args.refine_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)
        it = ckpt.get('iteration', 0)
        print(f"Resumed refinement ckpt: epoch {start_epoch}, iter {it}")

    pc = sum(p.numel() for p in model.parameters())
    print(f"TrajectoryRefinementNet: {pc:,} parameters ({pc/1e6:.2f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=schedule['lr_milestones'], gamma=args.lr_decay
    )

    model_dir = pjoin(args.checkpoints_dir, 'refinement', args.exp_name)
    os.makedirs(model_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=model_dir)

    # Per-branch forward helpers ----------------------------------------
    def _run_model(input_feats, motion_base):
        """Run model and splice output back into motion. Returns (motion_pred, pred_root, foot_delta)."""
        out = model(input_feats)
        if args.refine_mode == 'root_foot':
            pred_root, foot_delta = out
        else:
            pred_root, foot_delta = out, None
        motion_pred = model.apply_to_motion(motion_base, pred_root, foot_delta)
        return motion_pred, pred_root, foot_delta

    def _root_foot_extra_loss(motion_pred, motion_gt_67, lengths, gt_contacts=None):
        """Option 3b extra losses — only called when refine_mode='root_foot'."""
        if args.refine_mode != 'root_foot':
            return torch.tensor(0.0, device=device), {}
        extra, eld = compute_foot_correction_losses(
            motion_pred, motion_gt_67, lengths,
            gt_contacts=gt_contacts,
            foot_recon_weight=args.foot_recon_weight,
            skate_weight=args.skate_weight,
        )
        return extra, eld

    def forward_gt_branch(batch_hml3d, batch_styles, it_now):
        lengths = batch_hml3d['lengths'].to(device)
        joint_pos   = batch_hml3d['joint_pos'].to(device)
        joint_vel   = batch_hml3d['joint_vel'].to(device)
        foot_contact = batch_hml3d['foot_contact'].to(device)
        gt_root     = batch_hml3d['root_feats'].to(device).permute(0, 2, 1)
        motion_67   = batch_hml3d['motion_67'].to(device).permute(0, 2, 1)
        raw_contacts_cf = batch_hml3d['raw_contacts'].to(device).permute(0, 2, 1)

        input_feats = torch.cat([joint_pos, joint_vel, foot_contact], dim=-1).permute(0, 2, 1)
        motion_pred, pred_root, _ = _run_model(input_feats, motion_67)

        loss_h, ld_h = compute_refinement_losses(
            pred_root, gt_root, motion_pred, lengths, it_now,
            recon_weight=args.recon_weight, vel_weight=args.vel_weight,
            skate_weight=args.skate_weight, gt_contacts=raw_contacts_cf,
            skate_mode=args.skate_mode,
        )
        extra_h, _ = _root_foot_extra_loss(motion_pred, motion_67, lengths, raw_contacts_cf)

        lengths_s  = batch_styles['lengths'].to(device)
        joint_pos_s = batch_styles['joint_pos'].to(device)
        joint_vel_s = batch_styles['joint_vel'].to(device)
        gt_root_s  = batch_styles['root_feats'].to(device).permute(0, 2, 1)
        motion_67_s = batch_styles['motion_67'].to(device).permute(0, 2, 1)

        fake_contacts = torch.zeros(joint_pos_s.shape[0], joint_pos_s.shape[1], 4, device=device)
        input_feats_s = torch.cat([joint_pos_s, joint_vel_s, fake_contacts], dim=-1).permute(0, 2, 1)
        motion_pred_s, pred_root_s, _ = _run_model(input_feats_s, motion_67_s)

        loss_s, ld_s = compute_refinement_losses(
            pred_root_s, gt_root_s, motion_pred_s, lengths_s, it_now,
            recon_weight=args.recon_weight, vel_weight=args.vel_weight,
            skate_weight=0.0, gt_contacts=None, skate_mode='none',
        )
        extra_s, _ = _root_foot_extra_loss(motion_pred_s, motion_67_s, lengths_s)

        return loss_h + loss_s + extra_h + extra_s, ld_h, ld_s

    def forward_gen_branch(batch_data, it_now, train_mode=True):
        z_hml3d   = batch_data['latent_humanml'].to(device)
        motion_gt = batch_data['motion_humanml'].float().to(device)
        len_hml3d = batch_data['length_humanml'].to(device)
        text_hml3d = batch_data['text_humanml']

        with torch.no_grad():
            _, full_pred, _ = mardm.forward_loss(
                z_hml3d, text_hml3d, len_hml3d, raw_style_latents=None,
            )
            motion_gen = dae.decode(full_pred)

        motion_gen = motion_gen.permute(0, 2, 1)
        motion_gt_cf = motion_gt.permute(0, 2, 1)

        body_local = motion_gen[:, 4:, :]
        velocities = compute_velocities(body_local)
        # Option 1: zeros instead of noisy detect_foot_contact on normalized data
        contacts   = torch.zeros(motion_gen.shape[0], 4, motion_gen.shape[2], device=device)
        input_feats = torch.cat([body_local, velocities, contacts], dim=1)

        gt_root = motion_gt_cf[:, :4, :]
        motion_base = motion_gen.clone().detach() if train_mode else motion_gen.clone()
        motion_pred, pred_root, _ = _run_model(input_feats, motion_base)

        loss_h, ld_h = compute_refinement_losses(
            pred_root, gt_root, motion_pred, len_hml3d * 4, it_now,
            recon_weight=args.recon_weight, vel_weight=args.vel_weight,
            skate_weight=args.skate_weight, skate_mode=args.skate_mode,
        )
        extra_h, _ = _root_foot_extra_loss(motion_pred, motion_gt_cf, len_hml3d * 4)

        motion_style = batch_data['motion_styled'].float().to(device)
        len_style = batch_data['length_styled'].to(device) // 4
        text_style = batch_data['text_styled']

        with torch.no_grad():
            vid_inputs = processor(batch_data['video_styled'], return_tensors="pt").to(device)
            vid_tensors = vmodel(**vid_inputs).last_hidden_state
            z_style, raw_video_latents = dae.encode(motion_style, vid_tensors)
            _, full_pred_style, _ = mardm.forward_loss(
                z_style, text_style, len_style,
                raw_style_latents=raw_video_latents,
                style_weight_schedule=w_schedule,
            )
            motion_gen_style = dae.decode(full_pred_style)

        motion_gen_style = motion_gen_style.permute(0, 2, 1)
        motion_style_cf  = motion_style.permute(0, 2, 1)

        body_local_s = motion_gen_style[:, 4:, :]
        velocities_s = compute_velocities(body_local_s)
        contacts_s   = torch.zeros(motion_gen_style.shape[0], 4, motion_gen_style.shape[2], device=device)
        input_feats_s = torch.cat([body_local_s, velocities_s, contacts_s], dim=1)

        gt_root_s = motion_style_cf[:, :4, :]
        motion_base_s = motion_gen_style.clone().detach() if train_mode else motion_gen_style.clone()
        motion_pred_s, pred_root_s, _ = _run_model(input_feats_s, motion_base_s)

        loss_s, ld_s = compute_refinement_losses(
            pred_root_s, gt_root_s, motion_pred_s, len_style * 4, it_now,
            recon_weight=args.recon_weight, vel_weight=args.vel_weight,
            skate_weight=args.skate_weight, skate_mode=args.skate_mode,
        )
        extra_s, _ = _root_foot_extra_loss(motion_pred_s, motion_style_cf, len_style * 4)

        return loss_h + loss_s + extra_h + extra_s, ld_h, ld_s

    # Training loop ------------------------------------------------------
    best_val_loss = float('inf')
    start_time = time.time()
    logs = defaultdict(def_value)

    for epoch in range(start_epoch, start_epoch + args.epoch):
        model.train()
        epoch_losses = defaultdict(float)
        n_gt_iters = 0
        n_gen_iters = 0

        gt_hml3d_iter = iter(gt_hml3d_loader)
        gt_styles_iter = iter(gt_styles_loader)
        gen_train_iter = iter(gen_train_loader)

        num_steps = len(gen_train_loader)

        with tqdm(range(num_steps), desc=f"Epoch {epoch} | Train") as pbar:
            for step in pbar:
                optimizer.zero_grad()
                it += 1

                # LR warmup (overrides scheduler during first warm_up_iter iters)
                if it < schedule['warm_up_iter']:
                    update_lr_warm_up(it, schedule['warm_up_iter'], optimizer, args.lr)

                p_gt = get_p_gt(it, args.warmup_iters, args.mix_gt_ratio)
                use_gt = random.random() < p_gt

                if use_gt:
                    try:
                        b_hml3d = next(gt_hml3d_iter)
                    except StopIteration:
                        gt_hml3d_iter = iter(gt_hml3d_loader)
                        b_hml3d = next(gt_hml3d_iter)
                    try:
                        b_styles = next(gt_styles_iter)
                    except StopIteration:
                        gt_styles_iter = iter(gt_styles_loader)
                        b_styles = next(gt_styles_iter)

                    loss_total, ld_h, ld_s = forward_gt_branch(b_hml3d, b_styles, it)
                    branch = 'gt'
                    n_gt_iters += 1
                else:
                    try:
                        b_gen = next(gen_train_iter)
                    except StopIteration:
                        gen_train_iter = iter(gen_train_loader)
                        b_gen = next(gen_train_iter)

                    loss_total, ld_h, ld_s = forward_gen_branch(b_gen, it, train_mode=True)
                    branch = 'gen'
                    n_gen_iters += 1

                loss_total.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                scheduler.step()

                for k, v in ld_h.items():
                    epoch_losses[f'{branch}_hml3d_{k}'] += v
                for k, v in ld_s.items():
                    epoch_losses[f'{branch}_styles_{k}'] += v
                epoch_losses[f'{branch}_total'] += loss_total.item()

                logs['loss'] += loss_total.item()
                logs['recon'] += ld_h.get('recon', 0)
                logs['vel'] += ld_h.get('vel', 0)
                logs['lr'] += optimizer.param_groups[0]['lr']

                pbar.set_postfix({
                    'br': branch,
                    'p_gt': f"{p_gt:.2f}",
                    'total': f"{loss_total.item():.4f}",
                    'hml3d': f"{ld_h['total']:.4f}",
                    'styles': f"{ld_s['total']:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                })

                if it % args.log_every == 0:
                    mean_logs = {k: v / args.log_every for k, v in logs.items()}
                    print_current_loss(start_time, it, total_iters, mean_logs,
                                       epoch=epoch, inner_iter=step)
                    for k, v in mean_logs.items():
                        logger.add_scalar(f'Train_iter/{k}', v, it)
                    logs = defaultdict(def_value)

        wandb_log = {
            'train/lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
            'train/n_gt_iters': n_gt_iters,
            'train/n_gen_iters': n_gen_iters,
            'train/p_gt': get_p_gt(it, args.warmup_iters, args.mix_gt_ratio),
        }
        for k, v in epoch_losses.items():
            denom = n_gt_iters if k.startswith('gt_') else n_gen_iters
            if denom > 0:
                wandb_log[f'train/{k}'] = v / denom
                logger.add_scalar(f'Train/{k}', v / denom, epoch)
        wandb.log(wandb_log, step=it)

        print(f"Epoch {epoch} | GT iters: {n_gt_iters} | Gen iters: {n_gen_iters}")

        # Validation: both GT-path and Gen-path --------------------------
        model.eval()
        val_gt = defaultdict(float)
        val_gen = defaultdict(float)
        n_gt_val = 0
        n_gen_val = 0

        with torch.no_grad():
            for batch in tqdm(gt_hml3d_val_loader, desc=f"Epoch {epoch} | Val(GT)"):
                lengths = batch['lengths'].to(device)
                joint_pos    = batch['joint_pos'].to(device)
                joint_vel    = batch['joint_vel'].to(device)
                foot_contact = batch['foot_contact'].to(device)
                gt_root      = batch['root_feats'].to(device).permute(0, 2, 1)
                motion_67    = batch['motion_67'].to(device).permute(0, 2, 1)
                raw_contacts_cf = batch['raw_contacts'].to(device).permute(0, 2, 1)

                input_feats = torch.cat([joint_pos, joint_vel, foot_contact], dim=-1).permute(0, 2, 1)
                motion_pred, pred_root, _ = _run_model(input_feats, motion_67)

                _, ld = compute_refinement_losses(
                    pred_root, gt_root, motion_pred, lengths, it,
                    recon_weight=args.recon_weight,
                    vel_weight=args.vel_weight,
                    skate_weight=args.skate_weight,
                    gt_contacts=raw_contacts_cf,
                    skate_mode=args.skate_mode,
                )
                for k, v in ld.items():
                    val_gt[k] += v
                n_gt_val += 1

            for batch_data in tqdm(gen_val_loader, desc=f"Epoch {epoch} | Val(Gen)"):
                _, ld_h, ld_s = forward_gen_branch(batch_data, it, train_mode=False)
                for k in ld_h:
                    val_gen[k] += (ld_h[k] + ld_s[k]) / 2
                n_gen_val += 1

        val_gt_total = val_gt['total'] / max(n_gt_val, 1)
        val_gen_total = val_gen['total'] / max(n_gen_val, 1)
        val_combined = (val_gt_total + val_gen_total) / 2

        vlog = {
            'val/gt_total': val_gt_total,
            'val/gen_total': val_gen_total,
            'val/combined': val_combined,
            'val/best': best_val_loss,
            'epoch': epoch,
        }
        for k, v in val_gt.items():
            vlog[f'val/gt_{k}'] = v / max(n_gt_val, 1)
            logger.add_scalar(f'Val_GT/{k}', v / max(n_gt_val, 1), epoch)
        for k, v in val_gen.items():
            vlog[f'val/gen_{k}'] = v / max(n_gen_val, 1)
            logger.add_scalar(f'Val_Gen/{k}', v / max(n_gen_val, 1), epoch)
        wandb.log(vlog, step=it)

        print(f"Epoch {epoch} | Val GT: {val_gt_total:.5f} | Val Gen: {val_gen_total:.5f} | Combined: {val_combined:.5f}")

        if val_combined < best_val_loss:
            best_val_loss = val_combined
            save_path = pjoin(model_dir, 'best.tar')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'iteration': it,
                'best_val_loss': best_val_loss,
            }, save_path)
            print(f"New best combined val: {val_combined:.5f} -- saved to {save_path}")

        if epoch % args.save_every_epochs == 0 or epoch == start_epoch + args.epoch - 1:
            save_path = pjoin(model_dir, f'epoch_{epoch}.tar')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'iteration': it,
            }, save_path)
            print(f"Checkpoint saved: {save_path}")

    logger.close()
    elapsed = time.time() - start_time
    print(f"Unified training complete. {elapsed/60:.1f} minutes.")
    wandb.finish()


# =============================================================================
# Entry point
# =============================================================================
def main(args):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    train_unified(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Trajectory Refinement Module')

    # Schedule (single-stage, batch-level GT/gen mixing)
    parser.add_argument('--warmup_iters', type=int, default=20000,
                        help='Iterations of pure-GT input before mixing kicks in')
    parser.add_argument('--mix_gt_ratio', type=float, default=0.3,
                        help='After warmup, probability of drawing a GT batch (vs generated)')

    # Data
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--max_motion_length', type=int, default=196)
    parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 4,
                        help='DataLoader workers (default 0 on Windows, where worker processes can deadlock)')

    # Model
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of dilated residual blocks')
    parser.add_argument('--dilation_growth_rate', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='LR multiplier at milestone (80%% of training)')
    parser.add_argument('--save_every_epochs', type=int, default=50,
                        help='Save epoch_N.tar every N epochs')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Print loss summary every N iterations')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)

    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--vel_weight', type=float, default=10.0,
                        help='Weight for velocity loss (high = smoother trajectories)')
    parser.add_argument('--skate_weight', type=float, default=0.5)
    parser.add_argument('--foot_recon_weight', type=float, default=1.0,
                        help='Weight for foot-position reconstruction loss (root_foot mode only)')

    # Foot sliding control
    parser.add_argument('--skate_mode', type=str, default='none',
                        choices=['none', 'gt_only', 'full'],
                        help=(
                            'none    — Option 1: no skating loss (recommended for root_only mode); '
                            'gt_only — skating loss only on GT-HumanML3D batches where contacts are reliable; '
                            'full    — skating loss on all batches (uses detect_foot_contact for gen branch)'
                        ))
    parser.add_argument('--refine_mode', type=str, default='root_only',
                        choices=['root_only', 'root_foot'],
                        help=(
                            'root_only — predict root trajectory only (4 dims); '
                            'root_foot — predict root + foot XYZ corrections (16 dims, Option 3b)'
                        ))
    parser.add_argument('--use_contact_solver', action='store_true',
                        help='Apply post-hoc foot-lock solver at inference (Option 2). '
                             'No effect during training — pass to apply_refinement() at inference.')

    # Checkpoints
    # parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/100styles/MARDM_diffmlp_diff_v4_cross_batch_h')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--exp_name', type=str, default='refinment_v1'),
    parser.add_argument('--refine_ckpt', type=str, default=None,
                        help='Path to refinement checkpoint (for resume)')

    # Frozen pretrained models (used on gen-branch iterations).
    # Expects `--model_dir <dir>` containing both `final.tar` (MARDM) and
    # `dae_final.tar` (DAE).
    parser.add_argument('--model_dir', type=str, default='MARDM_diffmlp_diff_v4_cross_batch_h',
                        help='Directory holding final.tar (MARDM) and dae_final.tar (DAE)')
    parser.add_argument('--mardm_model', type=str, default='MARDM-DDPM-XL',
                        choices=['MARDM-DDPM-XL', 'MARDM-SiT-XL'])
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--style_routing', type=str, default='diffmlp',
                        choices=['diffmlp', 'mart'])
    parser.add_argument('--use_weight_schedule', action='store_true')
    parser.add_argument('--styles', type=str, nargs='+',
                        default=["Aeroplane", "Chicken", "Robot", "Superman", "ArmsFolded"])

    args = parser.parse_args()
    main(args)