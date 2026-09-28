"""
Phase 6: Unified Comparative Evaluator

Loads pre-exported feat67/joints .npy files from all models + GT,
and computes all metrics through one frozen evaluation pipeline.

Run from the repository root:
    python evaluate_comparative.py
    python evaluate_comparative.py --models mardm_2way smoodi --eval_modes styled base
    python evaluate_comparative.py --device 0 --seed 3407

Input: comparative_eval/ directory with manifest.json and per-model subdirectories.
Output: comparison table (stdout) + comparative_eval/results.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure MM_MARDM root is on sys.path so we can import utils/models
# ---------------------------------------------------------------------------
MM_MARDM_ROOT = os.path.dirname(os.path.abspath(__file__))
if MM_MARDM_ROOT not in sys.path:
    sys.path.insert(0, MM_MARDM_ROOT)

from utils.evaluators import Evaluators
from utils.eval_utils import (
    calculate_activation_statistics,
    calculate_frechet_distance,
    calculate_rr_mpjpe,
    calculate_global_mpjpe,
)
from train_style_classification import StyleClassification


# ###########################################################################
#                          Skating Ratio                                     #
# ###########################################################################

def calculate_skating_ratio(joints):
    """
    Calculate foot skating ratio from joint positions.

    Args:
        joints: numpy array [B, T, 22, 3] -- recovered joint positions

    Returns:
        skating_ratio: numpy array [B] -- fraction of frames with skating per sample
    """
    thresh_height = 0.05
    fps = 20.0
    thresh_vel = 0.50  # 50 cm/s
    avg_window = 5

    # foot joints: 10 (l_foot), 11 (r_foot)
    verts_feet = joints[:, :, [10, 11], :]          # [B, T, 2, 3]
    verts_feet = verts_feet.transpose(0, 2, 1, 3)   # [B, 2, T, 3]

    # XZ plane velocity
    verts_feet_plane_vel = np.linalg.norm(
        verts_feet[:, :, 1:, [0, 2]] - verts_feet[:, :, :-1, [0, 2]], axis=-1
    ) * fps  # [B, 2, T-1]

    vel_avg = uniform_filter1d(
        verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0
    )

    # Foot height (y axis)
    verts_feet_height = verts_feet[:, :, :, 1]  # [B, 2, T]

    # Contact: foot near ground in adjacent frames
    feet_contact = np.logical_and(
        verts_feet_height[:, :, :-1] < thresh_height,
        verts_feet_height[:, :, 1:] < thresh_height,
    )  # [B, 2, T-1]

    # Skating: contact + high velocity
    skating = np.logical_and(feet_contact, verts_feet_plane_vel > thresh_vel)
    skating = np.logical_and(skating, vel_avg > thresh_vel)

    # Either foot sliding
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :])  # [B, T-1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio


# ###########################################################################
#                         R-Precision (CLIP-based)                           #
# ###########################################################################

def compute_r_precision(text_embs, motion_embs, top_k=3, R_size=32, seed=3407):
    """
    Compute R-Precision Top-1/2/3 from CLIP text and motion embeddings.

    Args:
        text_embs:   Tensor [N, D] -- L2-normalized CLIP text embeddings
        motion_embs: Tensor [N, D] -- L2-normalized CLIP motion embeddings
        top_k: max k for top-k accuracy
        R_size: group size (number of candidates per query)
        seed: random seed for shuffle

    Returns:
        dict with R_precision_top_1, R_precision_top_2, R_precision_top_3
    """
    N = text_embs.shape[0]
    if N < R_size:
        print(f"  Warning: only {N} samples, need >= {R_size} for R-Precision. Skipping.")
        return None

    # Shuffle deterministically
    rng = torch.Generator()
    rng.manual_seed(seed)
    shuffle_idx = torch.randperm(N, generator=rng)
    text_embs = text_embs[shuffle_idx]
    motion_embs = motion_embs[shuffle_idx]

    top_k_hits = torch.zeros(top_k)
    num_groups = N // R_size

    for i in range(num_groups):
        s, e = i * R_size, (i + 1) * R_size
        gt = text_embs[s:e]
        gm = motion_embs[s:e]

        # Euclidean distance matrix [R_size, R_size]
        dist_mat = torch.cdist(gt, gm, p=2).nan_to_num()

        # Sort each row (ascending distance)
        argsort = torch.argsort(dist_mat, dim=1)
        for k in range(top_k):
            # row i's correct match is column i; check if i appears in first k+1 columns
            top_k_hits[k] += (
                argsort[:, :k + 1] == torch.arange(R_size).unsqueeze(1)
            ).any(dim=1).sum().item()

    R_count = num_groups * R_size
    results = {}
    for k in range(top_k):
        results[f'R_precision_top_{k + 1}'] = float(top_k_hits[k] / R_count)
    return results


# ###########################################################################
#                           SRA (Style Recognition Accuracy)                 #
# ###########################################################################

def compute_sra(all_logits, all_labels, topk=(1, 3, 5)):
    """
    Compute Style Recognition Accuracy at top-1, top-3, top-5.

    Args:
        all_logits: Tensor [N, C] -- classifier logits
        all_labels: Tensor [N]    -- ground-truth class indices
        topk: tuple of k values

    Returns:
        dict  e.g. {'SRA_top_1': 72.5, 'SRA_top_3': 91.0, 'SRA_top_5': 98.0}
    """
    N = all_labels.shape[0]
    maxk = max(topk)
    _, pred = all_logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # [maxk, N]
    correct = pred.eq(all_labels.view(1, -1).expand_as(pred))

    results = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        results[f'SRA_top_{k}'] = float(correct_k * 100.0 / N)
    return results


def compute_sra_per_style(all_logits, all_labels, idx_to_style, topk=(1,)):
    """
    Compute per-style SRA at top-1.

    Args:
        all_logits: Tensor [N, C]
        all_labels: Tensor [N]
        idx_to_style: dict {int: str} mapping class index to style name
        topk: tuple of k values (only top-1 used for per-style breakdown)

    Returns:
        dict  e.g. {'SRA_T1_Aeroplane': 85.7, 'SRA_T1_Chicken': 50.0, ...}
    """
    k = topk[0]
    _, pred = all_logits.topk(k, dim=1, largest=True, sorted=True)
    results = {}
    for idx, style_name in sorted(idx_to_style.items()):
        mask = (all_labels == idx)
        n_style = mask.sum().item()
        if n_style == 0:
            continue
        correct = pred[mask].eq(idx).any(dim=1).sum().item()
        results[f'SRA_T1_{style_name}'] = float(correct * 100.0 / n_style)
    return results


# ###########################################################################
#                        Diversity                                           #
# ###########################################################################

def compute_diversity(embeddings, num_pairs=300, seed=3407):
    """
    Compute diversity as mean pairwise L2 distance.

    Args:
        embeddings: np.ndarray [N, D]
        num_pairs: number of random pairs
        seed: random seed

    Returns:
        float -- mean L2 distance between pairs, or None if not enough samples
    """
    N = embeddings.shape[0]
    if N < 2:
        return None

    actual_pairs = min(num_pairs, N // 2)
    if actual_pairs < 2:
        return None

    rng = np.random.RandomState(seed)
    idx1 = rng.choice(N, actual_pairs, replace=False)
    idx2 = rng.choice(N, actual_pairs, replace=False)
    dist = np.linalg.norm(embeddings[idx1] - embeddings[idx2], axis=1)
    return float(dist.mean())


# ###########################################################################
#                       Data Loading Helpers                                 #
# ###########################################################################

def load_npy_files(directory, prefix, ids, desc="Loading"):
    """
    Load .npy files from a directory, keyed by manifest id.

    Args:
        directory: path to directory containing .npy files
        prefix: 'feat67' or 'joints'
        ids: list of manifest_id integers

    Returns:
        dict {manifest_id: np.ndarray}, count_found, count_missing
    """
    data = {}
    missing = 0
    for mid in ids:
        fname = f"{prefix}_{mid:05d}.npy"
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            data[mid] = np.load(fpath)
        else:
            missing += 1
    return data, len(data), missing


def get_model_display_name(model_key):
    """Map internal model keys to display names."""
    mapping = {
        'gt': 'GT',
        'mardm_2way': 'MARDM-2way',
        'mardm_3way_additive': 'MARDM-3way',
        'smoodi': 'SMooDi',
        'loramdm': 'LoRA-MDM',
    }
    if model_key in mapping:
        return mapping[model_key]
    if model_key.startswith('mardm_3way_t'):
        return model_key.replace('mardm_3way_', '3way-')
    if model_key.startswith('mardm_2way_c'):
        return model_key.replace('mardm_2way_', '2way-')
    return model_key


# ###########################################################################
#                        Per-mode Evaluation                                 #
# ###########################################################################

def evaluate_styled(manifest, comp_root, model_key, eval_wrapper, style_classifier,
                    style_to_idx, device, R_size, seed, gt_dir_override=None):
    """
    Evaluate STYLED mode for one model (or GT).

    Returns dict of metric_name -> value.
    """
    samples = manifest['styled_samples']
    ids = [s['manifest_id'] for s in samples]

    gt_base = gt_dir_override if gt_dir_override else os.path.join(comp_root, 'gt')
    if model_key == 'gt':
        model_dir = os.path.join(gt_base, 'styled')
    else:
        model_dir = os.path.join(comp_root, model_key, 'styled')
    gt_dir = os.path.join(gt_base, 'styled')

    # Load model data
    feat67_data, n_feat, n_miss_feat = load_npy_files(model_dir, 'feat67', ids)
    joints_data, n_joints, n_miss_joints = load_npy_files(model_dir, 'joints', ids)

    # Load GT data (for FID reference distribution + MPJPE)
    gt_feat67, _, _ = load_npy_files(gt_dir, 'feat67', ids)
    gt_joints, _, _ = load_npy_files(gt_dir, 'joints', ids)

    print(f"  [{get_model_display_name(model_key)}] styled: {n_feat}/{len(ids)} feat67, "
          f"{n_joints}/{len(ids)} joints loaded")

    if n_feat == 0:
        print(f"  WARNING: No feat67 files found for {model_key}/styled. Skipping.")
        return None

    results = OrderedDict()
    results['num_samples'] = n_feat

    # Build sample list ordered by manifest_id (only samples with feat67)
    sample_lookup = {s['manifest_id']: s for s in samples}
    valid_ids = sorted(feat67_data.keys())

    # --- Compute embeddings ---
    motion_embs_list = []
    clip_motion_embs_list = []
    clip_text_embs_list = []
    gt_motion_embs_list = []
    sra_logits_list = []
    sra_labels_list = []
    mpjpe_rr_list = []
    mpjpe_global_list = []
    all_gen_joints = []

    with torch.no_grad():
      for i, mid in enumerate(tqdm(valid_ids, desc=f"  {get_model_display_name(model_key)} styled embed", leave=False)):
        sample = sample_lookup[mid]
        feat67 = feat67_data[mid]  # [T, 67]
        T = feat67.shape[0]

        feat_t = torch.from_numpy(feat67).unsqueeze(0).float().to(device)  # [1, T, 67]
        m_len = torch.tensor([T], dtype=torch.long, device=device)

        # Motion embeddings (for FID + diversity)
        mot_emb, clip_mot_emb = eval_wrapper.get_motion_embeddings(feat_t, m_len)
        motion_embs_list.append(mot_emb.cpu().numpy())

        # CLIP embeddings for R-Precision
        clip_mot_emb_norm = clip_mot_emb / clip_mot_emb.norm(dim=1, keepdim=True)
        clip_motion_embs_list.append(clip_mot_emb_norm.cpu())

        # Use first caption for R-Precision
        captions = sample['captions']
        caption = captions[0] if isinstance(captions, list) else captions
        clip_text_emb = eval_wrapper.contrast_model.encode_text([caption])
        clip_text_emb_norm = clip_text_emb / clip_text_emb.norm(dim=1, keepdim=True)
        clip_text_embs_list.append(clip_text_emb_norm.cpu())

        # SRA: style classifier
        if style_classifier is not None and style_to_idx is not None:
            logits = style_classifier(feat_t, stage="Classification")  # [1, C]
            gt_idx = sample['style_idx']
            sra_logits_list.append(logits.cpu())
            sra_labels_list.append(torch.tensor([gt_idx], dtype=torch.long))

        # GT embeddings for FID
        if mid in gt_feat67:
            gt_f = gt_feat67[mid]
            gt_f_t = torch.from_numpy(gt_f).unsqueeze(0).float().to(device)
            gt_m_len = torch.tensor([gt_f.shape[0]], dtype=torch.long, device=device)
            gt_emb, _ = eval_wrapper.get_motion_embeddings(gt_f_t, gt_m_len)
            gt_motion_embs_list.append(gt_emb.cpu().numpy())

        # MPJPE (if both gen and GT joints available)
        if mid in joints_data and mid in gt_joints:
            gen_j = joints_data[mid]  # [T, 22, 3]
            gt_j = gt_joints[mid]
            min_len = min(gen_j.shape[0], gt_j.shape[0])
            gen_j_t = torch.from_numpy(gen_j[:min_len]).float()
            gt_j_t = torch.from_numpy(gt_j[:min_len]).float()
            rr = calculate_rr_mpjpe(gt_j_t, gen_j_t).mean().item()
            glob = calculate_global_mpjpe(gt_j_t, gen_j_t).mean().item()
            mpjpe_rr_list.append(rr)
            mpjpe_global_list.append(glob)

        # Skating (gen joints)
        if mid in joints_data:
            all_gen_joints.append(joints_data[mid])

        if (i + 1) % 200 == 0:
            torch.cuda.empty_cache()

    # --- Aggregate metrics ---

    # FID
    if gt_motion_embs_list and motion_embs_list:
        gt_all = np.concatenate(gt_motion_embs_list, axis=0)
        gen_all = np.concatenate(motion_embs_list, axis=0)
        # FID needs same-size distributions; use min
        min_n = min(len(gt_all), len(gen_all))
        if min_n >= 2:
            mu_gt, sigma_gt = calculate_activation_statistics(gt_all[:min_n])
            mu_gen, sigma_gen = calculate_activation_statistics(gen_all[:min_n])
            results['FID'] = float(calculate_frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen))

    # R-Precision
    if clip_text_embs_list and clip_motion_embs_list:
        all_text = torch.cat(clip_text_embs_list, dim=0)
        all_mot = torch.cat(clip_motion_embs_list, dim=0)
        rp = compute_r_precision(all_text, all_mot, top_k=3, R_size=R_size, seed=seed)
        if rp is not None:
            results.update(rp)

    # SRA
    if sra_logits_list and sra_labels_list:
        all_logits = torch.cat(sra_logits_list, dim=0)
        all_labels = torch.cat(sra_labels_list, dim=0)
        sra = compute_sra(all_logits, all_labels, topk=(1, 3, 5))
        results.update(sra)

        if style_to_idx:
            idx_to_style = {v: k for k, v in style_to_idx.items()}
            per_style = compute_sra_per_style(all_logits, all_labels, idx_to_style)
            results.update(per_style)

    # MPJPE
    if mpjpe_rr_list:
        results['RR_MPJPE'] = float(np.mean(mpjpe_rr_list))
    if mpjpe_global_list:
        results['Global_MPJPE'] = float(np.mean(mpjpe_global_list))

    # Skating
    if all_gen_joints:
        # Pad to same length and batch
        max_t = max(j.shape[0] for j in all_gen_joints)
        padded = [np.pad(j, ((0, max_t - j.shape[0]), (0, 0), (0, 0))) for j in all_gen_joints]
        stacked = np.stack(padded, axis=0)  # [B, T, 22, 3]
        skating = calculate_skating_ratio(stacked)
        results['Skating'] = float(skating.mean())

    # Diversity
    if motion_embs_list:
        gen_all_div = np.concatenate(motion_embs_list, axis=0)
        div = compute_diversity(gen_all_div, num_pairs=300, seed=seed)
        if div is not None:
            results['Diversity'] = div

    return results


def evaluate_base(manifest, comp_root, model_key, eval_wrapper, device, R_size, seed, gt_dir_override=None):
    """
    Evaluate BASE mode for one model (or GT).
    No SRA for base samples.

    Returns dict of metric_name -> value.
    """
    samples = manifest['base_samples']
    ids = [s['manifest_id'] for s in samples]

    gt_base = gt_dir_override if gt_dir_override else os.path.join(comp_root, 'gt')
    if model_key == 'gt':
        model_dir = os.path.join(gt_base, 'base')
    else:
        model_dir = os.path.join(comp_root, model_key, 'base')
    gt_dir = os.path.join(gt_base, 'base')

    feat67_data, n_feat, n_miss_feat = load_npy_files(model_dir, 'feat67', ids)
    joints_data, n_joints, n_miss_joints = load_npy_files(model_dir, 'joints', ids)
    gt_feat67, _, _ = load_npy_files(gt_dir, 'feat67', ids)
    gt_joints, _, _ = load_npy_files(gt_dir, 'joints', ids)

    print(f"  [{get_model_display_name(model_key)}] base: {n_feat}/{len(ids)} feat67, "
          f"{n_joints}/{len(ids)} joints loaded")

    if n_feat == 0:
        print(f"  WARNING: No feat67 files found for {model_key}/base. Skipping.")
        return None

    results = OrderedDict()
    results['num_samples'] = n_feat

    sample_lookup = {s['manifest_id']: s for s in samples}
    valid_ids = sorted(feat67_data.keys())

    motion_embs_list = []
    clip_motion_embs_list = []
    clip_text_embs_list = []
    gt_motion_embs_list = []
    mpjpe_rr_list = []
    mpjpe_global_list = []
    all_gen_joints = []

    with torch.no_grad():
      for i, mid in enumerate(tqdm(valid_ids, desc=f"  {get_model_display_name(model_key)} base embed", leave=False)):
        sample = sample_lookup[mid]
        feat67 = feat67_data[mid]
        T = feat67.shape[0]

        feat_t = torch.from_numpy(feat67).unsqueeze(0).float().to(device)
        m_len = torch.tensor([T], dtype=torch.long, device=device)

        mot_emb, clip_mot_emb = eval_wrapper.get_motion_embeddings(feat_t, m_len)
        motion_embs_list.append(mot_emb.cpu().numpy())

        clip_mot_emb_norm = clip_mot_emb / clip_mot_emb.norm(dim=1, keepdim=True)
        clip_motion_embs_list.append(clip_mot_emb_norm.cpu())

        caption = sample['caption']
        clip_text_emb = eval_wrapper.contrast_model.encode_text([caption])
        clip_text_emb_norm = clip_text_emb / clip_text_emb.norm(dim=1, keepdim=True)
        clip_text_embs_list.append(clip_text_emb_norm.cpu())

        if mid in gt_feat67:
            gt_f = gt_feat67[mid]
            gt_f_t = torch.from_numpy(gt_f).unsqueeze(0).float().to(device)
            gt_m_len = torch.tensor([gt_f.shape[0]], dtype=torch.long, device=device)
            gt_emb, _ = eval_wrapper.get_motion_embeddings(gt_f_t, gt_m_len)
            gt_motion_embs_list.append(gt_emb.cpu().numpy())

        if mid in joints_data and mid in gt_joints:
            gen_j = joints_data[mid]
            gt_j = gt_joints[mid]
            min_len = min(gen_j.shape[0], gt_j.shape[0])
            gen_j_t = torch.from_numpy(gen_j[:min_len]).float()
            gt_j_t = torch.from_numpy(gt_j[:min_len]).float()
            rr = calculate_rr_mpjpe(gt_j_t, gen_j_t).mean().item()
            glob = calculate_global_mpjpe(gt_j_t, gen_j_t).mean().item()
            mpjpe_rr_list.append(rr)
            mpjpe_global_list.append(glob)

        if mid in joints_data:
            all_gen_joints.append(joints_data[mid])

        if (i + 1) % 200 == 0:
            torch.cuda.empty_cache()

    # Aggregate
    if gt_motion_embs_list and motion_embs_list:
        gt_all = np.concatenate(gt_motion_embs_list, axis=0)
        gen_all = np.concatenate(motion_embs_list, axis=0)
        min_n = min(len(gt_all), len(gen_all))
        if min_n >= 2:
            mu_gt, sigma_gt = calculate_activation_statistics(gt_all[:min_n])
            mu_gen, sigma_gen = calculate_activation_statistics(gen_all[:min_n])
            results['FID'] = float(calculate_frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen))

    if clip_text_embs_list and clip_motion_embs_list:
        all_text = torch.cat(clip_text_embs_list, dim=0)
        all_mot = torch.cat(clip_motion_embs_list, dim=0)
        rp = compute_r_precision(all_text, all_mot, top_k=3, R_size=R_size, seed=seed)
        if rp is not None:
            results.update(rp)

    if mpjpe_rr_list:
        results['RR_MPJPE'] = float(np.mean(mpjpe_rr_list))
    if mpjpe_global_list:
        results['Global_MPJPE'] = float(np.mean(mpjpe_global_list))

    if all_gen_joints:
        chunk_size = 256
        skating_vals = []
        for ci in range(0, len(all_gen_joints), chunk_size):
            chunk = all_gen_joints[ci:ci + chunk_size]
            max_t = max(j.shape[0] for j in chunk)
            padded = [np.pad(j, ((0, max_t - j.shape[0]), (0, 0), (0, 0))) for j in chunk]
            stacked = np.stack(padded, axis=0)
            skating_vals.append(calculate_skating_ratio(stacked))
        skating = np.concatenate(skating_vals)
        results['Skating'] = float(skating.mean())

    if motion_embs_list:
        gen_all_div = np.concatenate(motion_embs_list, axis=0)
        div = compute_diversity(gen_all_div, num_pairs=300, seed=seed)
        if div is not None:
            results['Diversity'] = div

    return results


def evaluate_transfer(manifest, comp_root, model_key, eval_wrapper, style_classifier,
                      style_to_idx, device, R_size, seed):
    """
    Evaluate TRANSFER mode for one model.
    No FID or MPJPE (no paired GT). Only SRA, R-Precision, Skating, Diversity.

    Returns dict of metric_name -> value, or None if no transfer_samples.
    """
    samples = manifest.get('transfer_samples', [])
    if not samples:
        print(f"  [{get_model_display_name(model_key)}] transfer: no transfer_samples in manifest. Skipping.")
        return None

    ids = [s['manifest_id'] for s in samples]
    model_dir = os.path.join(comp_root, model_key, 'styled')  # transfer uses styled dir convention

    # Try 'transfer' subdirectory first, fall back to 'styled'
    transfer_dir = os.path.join(comp_root, model_key, 'transfer')
    if os.path.isdir(transfer_dir):
        model_dir = transfer_dir

    feat67_data, n_feat, _ = load_npy_files(model_dir, 'feat67', ids)
    joints_data, n_joints, _ = load_npy_files(model_dir, 'joints', ids)

    print(f"  [{get_model_display_name(model_key)}] transfer: {n_feat}/{len(ids)} feat67, "
          f"{n_joints}/{len(ids)} joints loaded")

    if n_feat == 0:
        print(f"  WARNING: No feat67 files found for {model_key}/transfer. Skipping.")
        return None

    results = OrderedDict()
    results['num_samples'] = n_feat

    sample_lookup = {s['manifest_id']: s for s in samples}
    valid_ids = sorted(feat67_data.keys())

    motion_embs_list = []
    clip_motion_embs_list = []
    clip_text_embs_list = []
    sra_logits_list = []
    sra_labels_list = []
    all_gen_joints = []

    with torch.no_grad():
      for mid in tqdm(valid_ids, desc=f"  {get_model_display_name(model_key)} transfer embed", leave=False):
        sample = sample_lookup[mid]
        feat67 = feat67_data[mid]
        T = feat67.shape[0]

        feat_t = torch.from_numpy(feat67).unsqueeze(0).float().to(device)
        m_len = torch.tensor([T], dtype=torch.long, device=device)

        mot_emb, clip_mot_emb = eval_wrapper.get_motion_embeddings(feat_t, m_len)
        motion_embs_list.append(mot_emb.cpu().numpy())

        clip_mot_emb_norm = clip_mot_emb / clip_mot_emb.norm(dim=1, keepdim=True)
        clip_motion_embs_list.append(clip_mot_emb_norm.cpu())

        caption = sample.get('text', sample.get('caption', ''))
        if caption:
            clip_text_emb = eval_wrapper.contrast_model.encode_text([caption])
            clip_text_emb_norm = clip_text_emb / clip_text_emb.norm(dim=1, keepdim=True)
            clip_text_embs_list.append(clip_text_emb_norm.cpu())

        if style_classifier is not None and style_to_idx is not None:
            logits = style_classifier(feat_t, stage="Classification")
            gt_idx = sample.get('style_idx', 0)
            sra_logits_list.append(logits.cpu())
            sra_labels_list.append(torch.tensor([gt_idx], dtype=torch.long))

        if mid in joints_data:
            all_gen_joints.append(joints_data[mid])

    # SRA
    if sra_logits_list and sra_labels_list:
        all_logits = torch.cat(sra_logits_list, dim=0)
        all_labels = torch.cat(sra_labels_list, dim=0)
        sra = compute_sra(all_logits, all_labels, topk=(1, 3, 5))
        results.update(sra)

        if style_to_idx:
            idx_to_style = {v: k for k, v in style_to_idx.items()}
            per_style = compute_sra_per_style(all_logits, all_labels, idx_to_style)
            results.update(per_style)

    # R-Precision
    if clip_text_embs_list and clip_motion_embs_list:
        all_text = torch.cat(clip_text_embs_list, dim=0)
        all_mot = torch.cat(clip_motion_embs_list, dim=0)
        rp = compute_r_precision(all_text, all_mot, top_k=3, R_size=R_size, seed=seed)
        if rp is not None:
            results.update(rp)

    # Skating
    if all_gen_joints:
        max_t = max(j.shape[0] for j in all_gen_joints)
        padded = [np.pad(j, ((0, max_t - j.shape[0]), (0, 0), (0, 0))) for j in all_gen_joints]
        stacked = np.stack(padded, axis=0)
        skating = calculate_skating_ratio(stacked)
        results['Skating'] = float(skating.mean())

    # Diversity
    if motion_embs_list:
        gen_all_div = np.concatenate(motion_embs_list, axis=0)
        div = compute_diversity(gen_all_div, num_pairs=min(300, len(gen_all_div) // 2), seed=seed)
        if div is not None:
            results['Diversity'] = div

    return results


# ###########################################################################
#                    GT Metrics (sanity check column)                         #
# ###########################################################################

def evaluate_gt_styled(manifest, comp_root, eval_wrapper, style_classifier,
                       style_to_idx, device, R_size, seed, gt_dir_override=None):
    """Compute GT-on-GT metrics for the styled split (sanity check)."""
    return evaluate_styled(
        manifest, comp_root, 'gt', eval_wrapper, style_classifier,
        style_to_idx, device, R_size, seed, gt_dir_override=gt_dir_override
    )


def evaluate_gt_base(manifest, comp_root, eval_wrapper, device, R_size, seed, gt_dir_override=None):
    """Compute GT-on-GT metrics for the base split (sanity check)."""
    return evaluate_base(
        manifest, comp_root, 'gt', eval_wrapper, device, R_size, seed,
        gt_dir_override=gt_dir_override
    )


# ###########################################################################
#                       Pretty-Print Tables                                  #
# ###########################################################################

def build_styled_metrics(styles):
    """Build STYLED_METRICS list with per-style SRA rows."""
    metrics = [
        ('FID',                 'FID',                  '(down)',  '{:.3f}',  True),
        ('R_precision_top_1',   'R-Prec T1',            '(up)',    '{:.4f}',  False),
        ('R_precision_top_2',   'R-Prec T2',            '(up)',    '{:.4f}',  False),
        ('R_precision_top_3',   'R-Prec T3',            '(up)',    '{:.4f}',  False),
        ('SRA_top_1',           'SRA T1',               '(up)',    '{:.2f}%', False),
        ('SRA_top_3',           'SRA T3',               '(up)',    '{:.2f}%', False),
        ('SRA_top_5',           'SRA T5',               '(up)',    '{:.2f}%', False),
    ]
    for style in styles:
        metrics.append(
            (f'SRA_T1_{style}', f'  SRA T1 {style}', '(up)', '{:.2f}%', False),
        )
    metrics += [
        ('RR_MPJPE',            'RR-MPJPE',             '(down)',  '{:.4f}',  True),
        ('Global_MPJPE',        'Global MPJPE',         '(down)',  '{:.4f}',  True),
        ('Skating',             'Skating',              '(down)',  '{:.4f}',  False),
        ('Diversity',           'Diversity',             '(mid)',   '{:.4f}',  False),
    ]
    return metrics

BASE_METRICS = [
    ('FID',                 'FID',                  '(down)',  '{:.3f}',  True),
    ('R_precision_top_1',   'R-Prec T1',            '(up)',    '{:.4f}',  False),
    ('R_precision_top_2',   'R-Prec T2',            '(up)',    '{:.4f}',  False),
    ('R_precision_top_3',   'R-Prec T3',            '(up)',    '{:.4f}',  False),
    ('RR_MPJPE',            'RR-MPJPE',             '(down)',  '{:.4f}',  True),
    ('Global_MPJPE',        'Global MPJPE',         '(down)',  '{:.4f}',  True),
    ('Skating',             'Skating',              '(down)',  '{:.4f}',  False),
    ('Diversity',           'Diversity',             '(mid)',   '{:.4f}',  False),
]

def build_transfer_metrics(styles):
    """Build TRANSFER_METRICS list with per-style SRA rows."""
    metrics = [
        ('SRA_top_1',           'SRA T1',               '(up)',    '{:.2f}%', False),
        ('SRA_top_3',           'SRA T3',               '(up)',    '{:.2f}%', False),
        ('SRA_top_5',           'SRA T5',               '(up)',    '{:.2f}%', False),
    ]
    for style in styles:
        metrics.append(
            (f'SRA_T1_{style}', f'  SRA T1 {style}', '(up)', '{:.2f}%', False),
        )
    metrics += [
        ('R_precision_top_1',   'R-Prec T1',            '(up)',    '{:.4f}',  False),
        ('R_precision_top_2',   'R-Prec T2',            '(up)',    '{:.4f}',  False),
        ('R_precision_top_3',   'R-Prec T3',            '(up)',    '{:.4f}',  False),
        ('Skating',             'Skating',              '(down)',  '{:.4f}',  False),
        ('Diversity',           'Diversity',             '(mid)',   '{:.4f}',  False),
    ]
    return metrics


def format_value(val, fmt, is_gt_dash):
    """Format a metric value, or return '---' if missing / GT-only."""
    if val is None:
        return '---'
    if is_gt_dash and val is None:
        return '---'
    # Handle percent format
    if fmt.endswith('%'):
        return fmt[:-1].format(val) + '%'
    return fmt.format(val)


def print_table(title, metrics_spec, all_results, models_with_gt):
    """
    Print a formatted comparison table.

    Args:
        title: table title
        metrics_spec: list of (key, display_name, direction, fmt, gt_is_dash)
        all_results: dict { model_key: dict-of-metrics }  (includes 'gt' key)
        models_with_gt: list of model keys in display order, 'gt' first
    """
    # Determine column widths
    col_names = [get_model_display_name(m) for m in models_with_gt]
    metric_col_w = max(len(display) + len(direction) + 2 for _, display, direction, _, _ in metrics_spec)
    metric_col_w = max(metric_col_w, 20)
    data_col_w = max(max(len(n) for n in col_names), 12)

    # Determine how many samples each model evaluated
    sample_counts = []
    for m in models_with_gt:
        r = all_results.get(m)
        if r is not None:
            sample_counts.append(str(r.get('num_samples', '?')))
        else:
            sample_counts.append('---')

    # Print header
    print()
    print(f"=== {title} ===")
    header = f"{'Metric':<{metric_col_w}}"
    for name in col_names:
        header += f"  {name:>{data_col_w}}"
    print(header)
    print("-" * len(header))

    # Sample count row
    row = f"{'# samples':<{metric_col_w}}"
    for sc in sample_counts:
        row += f"  {sc:>{data_col_w}}"
    print(row)
    print("-" * len(header))

    # Metric rows
    for key, display, direction, fmt, gt_is_dash in metrics_spec:
        label = f"{display} {direction}"
        row = f"{label:<{metric_col_w}}"
        for m in models_with_gt:
            r = all_results.get(m)
            if r is None:
                row += f"  {'---':>{data_col_w}}"
            else:
                val = r.get(key)
                # For GT column, MPJPE/FID should be '---' (comparing GT vs GT is meaningless)
                if m == 'gt' and gt_is_dash:
                    row += f"  {'---':>{data_col_w}}"
                else:
                    formatted = format_value(val, fmt, False)
                    row += f"  {formatted:>{data_col_w}}"
        print(row)

    print()


# ###########################################################################
#                              Main                                          #
# ###########################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Unified Comparative Evaluator for all motion generation models."
    )
    # parser.add_argument('--manifest', type=str, default='../comparative_eval/manifest.json',
    parser.add_argument('--manifest', type=str, default='comparative_eval/manifest.json',
                        help='Path to manifest.json')
    # parser.add_argument('--comp_root', type=str, default='../comparative_eval',
    parser.add_argument('--comp_root', type=str, default='comparative_eval',
                        help='Root directory of comparative_eval/')
    parser.add_argument('--models', nargs='+',
                        default=['mardm_2way', 'mardm_3way_additive', 'smoodi', 'loramdm'],
                        help='Model keys to evaluate')
    parser.add_argument('--eval_modes', nargs='+', default=['styled', 'base', 'transfer'],
                        help='Evaluation modes to run')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Override GT directory (default: <comp_root>/gt)')
    parser.add_argument('--R_size', type=int, default=16,
                        help='Group size for R-Precision computation (16 matches MM-MARDM native eval)')
    parser.add_argument('--device', type=int, default=0, help='GPU device index')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Path for results JSON. Default: <comp_root>/results.json')
    args = parser.parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(args.device)}")

    # Load manifest
    manifest_path = os.path.join(MM_MARDM_ROOT, args.manifest) if not os.path.isabs(args.manifest) else args.manifest
    comp_root = os.path.join(MM_MARDM_ROOT, args.comp_root) if not os.path.isabs(args.comp_root) else args.comp_root

    print(f"Loading manifest from: {manifest_path}")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"  styled_samples: {len(manifest.get('styled_samples', []))}")
    print(f"  base_samples:   {len(manifest.get('base_samples', []))}")
    print(f"  transfer_samples: {len(manifest.get('transfer_samples', []))}")

    # -------------------------------------------------------------------
    # Load evaluation models (once, shared across all model evaluations)
    # -------------------------------------------------------------------
    print("\nLoading Evaluators (text_mot_match + CLIP)...")
    eval_wrapper = Evaluators('t2m', device=device)

    # Load style classifier
    style_classifier = None
    style_to_idx = None
    sc_ckpt_path = os.path.join(MM_MARDM_ROOT, 'checkpoints', 'style_classifier', 'style_classifier_final.pt')
    if os.path.exists(sc_ckpt_path):
        print(f"Loading style classifier from: {sc_ckpt_path}")
        sc_ckpt = torch.load(sc_ckpt_path, map_location=device, weights_only=False)

        # Recover style_to_idx from checkpoint (training order, NOT alphabetical!)
        if 'style_to_idx' in sc_ckpt:
            style_to_idx = sc_ckpt['style_to_idx']
            print(f"  style_to_idx from checkpoint: {style_to_idx}")
        elif isinstance(sc_ckpt.get('args'), dict) and sc_ckpt['args'].get('styles'):
            train_styles = sc_ckpt['args']['styles']
            style_to_idx = {s: i for i, s in enumerate(train_styles)}
            print(f"  style_to_idx rebuilt from checkpoint args.styles ({len(train_styles)} classes)")
        else:
            # Fallback to manifest metadata (which stores training order)
            meta_s2i = manifest.get('metadata', {}).get('style_to_idx')
            if meta_s2i:
                style_to_idx = meta_s2i
                print(f"  style_to_idx from manifest metadata: {style_to_idx}")
            else:
                print("  WARNING: Cannot determine style_to_idx. SRA will be skipped.")

        if style_to_idx is not None:
            nclasses = len(style_to_idx)
            # Determine architecture params from checkpoint args if available
            ckpt_args = sc_ckpt.get('args', {})
            latent_dim = ckpt_args.get('latent_dim', 512)
            ff_size = ckpt_args.get('ff_size', 1024)
            num_layers = ckpt_args.get('num_layers', 6)
            num_heads = ckpt_args.get('num_heads', 4)
            dropout = ckpt_args.get('dropout', 0.1)

            style_classifier = StyleClassification(
                nclasses=nclasses, input_dim=67,
                latent_dim=[1, latent_dim], ff_size=ff_size,
                num_layers=num_layers, num_heads=num_heads, dropout=dropout,
            )
            state_key = 'model_state_dict' if 'model_state_dict' in sc_ckpt else 'state_dict'
            style_classifier.load_state_dict(sc_ckpt[state_key])
            style_classifier.to(device).eval()
            for p in style_classifier.parameters():
                p.requires_grad = False
            print(f"  Style classifier loaded: {nclasses} classes")
    else:
        print(f"Style classifier not found at {sc_ckpt_path}. SRA metrics will be skipped.")

    # -------------------------------------------------------------------
    # Verify directory structure
    # -------------------------------------------------------------------
    print("\nChecking directory structure...")
    all_model_keys = ['gt'] + list(args.models)
    for mode in args.eval_modes:
        for mk in all_model_keys:
            d = os.path.join(comp_root, mk, mode)
            if os.path.isdir(d):
                n_files = len([f for f in os.listdir(d) if f.endswith('.npy')])
                print(f"  {mk}/{mode}: {n_files} .npy files")
            else:
                if mk == 'gt' or mode == 'transfer':
                    # GT may not have transfer; transfer may not exist yet
                    pass
                else:
                    print(f"  WARNING: {d} does not exist")

    # -------------------------------------------------------------------
    # Run evaluations
    # -------------------------------------------------------------------
    all_results = {}
    start_time = time.time()

    # ---- STYLED ----
    if 'styled' in args.eval_modes and manifest.get('styled_samples'):
        print("\n" + "=" * 70)
        n_styled = len(manifest['styled_samples'])
        print(f"STYLED EVALUATION ({n_styled} samples)")
        print("=" * 70)

        styled_results = {}

        # GT sanity check
        print("\n  Evaluating GT (sanity check)...")
        gt_res = evaluate_gt_styled(
            manifest, comp_root, eval_wrapper, style_classifier,
            style_to_idx, device, args.R_size, args.seed,
            gt_dir_override=args.gt_dir
        )
        if gt_res is not None:
            styled_results['gt'] = gt_res

        # Each model
        for mk in args.models:
            print(f"\n  Evaluating {get_model_display_name(mk)}...")
            res = evaluate_styled(
                manifest, comp_root, mk, eval_wrapper, style_classifier,
                style_to_idx, device, args.R_size, args.seed,
                gt_dir_override=args.gt_dir
            )
            if res is not None:
                styled_results[mk] = res

        all_results['styled'] = styled_results

        # Print table
        display_order = ['gt'] + [m for m in args.models if m in styled_results]
        styles = manifest.get('metadata', {}).get('styles', [])
        print_table(
            f"STYLED EVALUATION ({n_styled} samples)",
            build_styled_metrics(styles), styled_results, display_order,
        )

    # ---- BASE ----
    if 'base' in args.eval_modes and manifest.get('base_samples'):
        print("\n" + "=" * 70)
        n_base = len(manifest['base_samples'])
        print(f"BASE EVALUATION ({n_base} samples)")
        print("=" * 70)

        base_results = {}

        print("\n  Evaluating GT (sanity check)...")
        gt_res = evaluate_gt_base(
            manifest, comp_root, eval_wrapper, device, args.R_size, args.seed,
            gt_dir_override=args.gt_dir
        )
        if gt_res is not None:
            base_results['gt'] = gt_res

        for mk in args.models:
            print(f"\n  Evaluating {get_model_display_name(mk)}...")
            res = evaluate_base(
                manifest, comp_root, mk, eval_wrapper, device, args.R_size, args.seed,
                gt_dir_override=args.gt_dir
            )
            if res is not None:
                base_results[mk] = res

        all_results['base'] = base_results

        display_order = ['gt'] + [m for m in args.models if m in base_results]
        print_table(
            f"BASE EVALUATION ({n_base} samples)",
            BASE_METRICS, base_results, display_order,
        )

    # ---- TRANSFER ----
    if 'transfer' in args.eval_modes and manifest.get('transfer_samples'):
        print("\n" + "=" * 70)
        n_transfer = len(manifest['transfer_samples'])
        print(f"TRANSFER EVALUATION ({n_transfer} samples)")
        print("=" * 70)

        transfer_results = {}

        for mk in args.models:
            print(f"\n  Evaluating {get_model_display_name(mk)}...")
            res = evaluate_transfer(
                manifest, comp_root, mk, eval_wrapper, style_classifier,
                style_to_idx, device, args.R_size, args.seed
            )
            if res is not None:
                transfer_results[mk] = res

        all_results['transfer'] = transfer_results

        display_order = [m for m in args.models if m in transfer_results]
        if display_order:
            styles = manifest.get('metadata', {}).get('styles', [])
            print_table(
                f"TRANSFER EVALUATION ({n_transfer} samples)",
                build_transfer_metrics(styles), transfer_results, display_order,
            )
    elif 'transfer' in args.eval_modes:
        print("\n  No transfer_samples in manifest yet. Skipping transfer evaluation.")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    elapsed = time.time() - start_time
    all_results['metadata'] = {
        'seed': args.seed,
        'R_size': args.R_size,
        'models': args.models,
        'eval_modes': args.eval_modes,
        'elapsed_seconds': elapsed,
    }

    output_path = args.output or os.path.join(comp_root, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == '__main__':
    main()
