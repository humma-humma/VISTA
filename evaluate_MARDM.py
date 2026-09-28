"""
Comprehensive Evaluation Script for MARDM with Dual-AdaLN Style Conditioning

Uses the same data loading strategy as train_MARDM.py:
- Text2MotionDatasetCombined_v4 for combined 100STYLES + HumanML3D data
- mld_collate_paired for batch collation
- Same val/test split strategy

Evaluates:
1. Base content generation (HumanML3D text → motion, no style)
2. Style-conditioned generation (100STYLES text+video → stylized motion)
3. Cross-modal style transfer (HumanML3D text + 100STYLES video → novel stylized motion)

Metrics computed:
- FID (Fréchet Inception Distance) for distribution quality
- Diversity for generation variety
- MPJPE (Mean Per-Joint Position Error) for reconstruction accuracy
- Latent MSE for latent space accuracy

Usage:
    # Full evaluation
    python evaluate_MARDM.py --checkpoint path/to/checkpoint.tar --eval_mode full
    
    # Only styled evaluation
    python evaluate_MARDM.py --checkpoint path/to/checkpoint.tar --eval_mode styled
"""



import os
from os.path import join as pjoin
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import argparse
import json
import copy
from pathlib import Path

# Model imports
from models.AE import DAE_models, AE_models
from models.MARDM import MARDM_models
from models.LengthEstimator import LengthEstimator
from models.refinement import TrajectoryRefinementNet

# Dataset imports - SAME AS train_MARDM.py
from utils.datasets import (
    Text2MotionDatasetCombined_v4,
    Text2MotionDatasetCombined_v5,
    mld_collate_paired,
    mld_collate_async
)

# Evaluation utilities
from utils.evaluators import Evaluators
from utils.eval_utils import (
    calculate_activation_statistics,
    calculate_frechet_distance,
    calculate_rr_mpjpe,
    calculate_global_mpjpe
)

# Visualization utilities
from utils.motion_process import (
    recover_from_ric,
    kit_kinematic_chain,
    t2m_kinematic_chain,
    plot_3d_motion_gif,
    plot_3d_motion_side_by_side,
    plot_3d_motion_three_way2
)

# Video encoder imports
from transformers import VivitModel, TimesformerModel, VivitImageProcessor, AutoProcessor

# Style classifier import
from train_style_classification import StyleClassification

from scipy.ndimage import uniform_filter1d


#################################################################################
#                              Helper Functions                                  #
#################################################################################

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, checkpoint_path, device, key='ema_mardm'):
    """
    Load model checkpoint with flexible key handling and weight schedule extraction.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        key: Key in checkpoint dict for model weights
    
    Returns:
        epoch: Epoch number from checkpoint (or 0 if not found)
        weight_schedule: The saved style weight schedule (or None if not found)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Load on CPU: the file also holds optimizer state and a second model copy (~7.8 GB); only the
    # selected weights should reach the GPU (load_state_dict copies onto the model's device).
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Try different keys in order of preference
    keys_to_try = [key, 'ema_mardm', 'mardm', 'model', 'state_dict']
    state_dict = None
    
    for k in keys_to_try:
        if k in checkpoint:
            state_dict = checkpoint[k]
            print(f"   Using weights from key: '{k}'")
            break
    
    if state_dict is None:
        state_dict = checkpoint
        print("   Using checkpoint directly as state_dict")
    
    # Load with strict=False to handle CLIP model keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Filter out expected missing keys (CLIP model is loaded separately)
    missing_filtered = [k for k in missing if not k.startswith('clip_model.')]
    
    if missing_filtered:
        print(f"   Warning: Missing keys: {missing_filtered[:5]}..." if len(missing_filtered) > 5 else f"   Warning: Missing keys: {missing_filtered}")
    if unexpected:
        print(f"   Warning: Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"   Warning: Unexpected keys: {unexpected}")
    
    # Extract metadata
    epoch = checkpoint.get('ep', 0)
    weight_schedule = checkpoint.get('weight_schedule', None)
    style_router = checkpoint.get('style_routing', None)
    
    if weight_schedule is not None:
        print(f"   Successfully loaded weight_schedule (length: {len(weight_schedule)})")
    else:
        print("   Warning: No weight_schedule found in checkpoint.")
    
    return epoch, weight_schedule, style_router


def get_style_weight_schedule(model, args):
    """
    Generate the style weight schedule based on routing mode.
    
    Args:
        model: MARDM model
        args: Arguments containing style_routing and use_weight_schedule
    
    Returns:
        List of weights for each block
    """
    if args.style_routing == 'diffmlp':
        num_blocks = model.DiffMLPs.get_total_blocks()
        if args.use_weight_schedule:
            return np.linspace(0.0, 1.0, num_blocks).tolist()
        else:
            return [1.0] * num_blocks
    else:
        # MART mode
        num_blocks = len(model.MARTransformer)
        return [args.mart_style_weight] * num_blocks


#################################################################################
#                           Evaluation Metrics                                   #
#################################################################################

def calculate_skating_ratio(joints):
    """
    Calculate foot skating ratio from joint positions.

    Args:
        joints: numpy array [B, T, 22, 3] — recovered joint positions

    Returns:
        skating_ratio: numpy array [B] — fraction of frames with foot skating per sample
    """
    thresh_height = 0.05
    fps = 20.0
    thresh_vel = 0.50  # 50 cm/s
    avg_window = 5

    # joints: [B, T, 22, 3] -> foot joints 10 (l_foot), 11 (r_foot)
    verts_feet = joints[:, :, [10, 11], :]  # [B, T, 2, 3]
    verts_feet = verts_feet.transpose(0, 2, 1, 3)  # [B, 2, T, 3]

    # XZ plane velocity
    verts_feet_plane_vel = np.linalg.norm(
        verts_feet[:, :, 1:, [0, 2]] - verts_feet[:, :, :-1, [0, 2]], axis=-1
    ) * fps  # [B, 2, T-1]

    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    # Foot height (y axis)
    verts_feet_height = verts_feet[:, :, :, 1]  # [B, 2, T]

    # Contact: foot near ground in adjacent frames
    feet_contact = np.logical_and(
        verts_feet_height[:, :, :-1] < thresh_height,
        verts_feet_height[:, :, 1:] < thresh_height
    )  # [B, 2, T-1]

    # Skating: contact + high velocity
    skating = np.logical_and(feet_contact, verts_feet_plane_vel > thresh_vel)
    skating = np.logical_and(skating, vel_avg > thresh_vel)

    # Either foot sliding
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :])  # [B, T-1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio


class MetricsAccumulator:
    """Accumulates and computes evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.motion_embeddings_gt = []
        self.motion_embeddings_gen = []
        self.mpjpe_rr = []
        self.mpjpe_global = []
        self.latent_mse = []
        self.recon_mse = []
        self.per_sample_results = []
        # R-Precision: CLIP-based text and motion embeddings
        self.text_embeddings_clip = []
        self.motion_embeddings_clip = []
        # SRA: style classifier logits and ground-truth labels
        self.style_predictions = []
        self.style_labels = []
        # Foot skating
        self.skating_ratios = []

    def add_embeddings(self, gt_emb, gen_emb):
        """Add motion embeddings for FID computation."""
        if gt_emb is not None:
            self.motion_embeddings_gt.append(gt_emb.cpu().numpy())
        if gen_emb is not None:
            self.motion_embeddings_gen.append(gen_emb.cpu().numpy())

    def add_clip_embeddings(self, text_emb, motion_emb):
        """Add CLIP-based text and motion embeddings for R-Precision."""
        if text_emb is not None:
            self.text_embeddings_clip.append(text_emb.cpu())
        if motion_emb is not None:
            self.motion_embeddings_clip.append(motion_emb.cpu())

    def add_style_predictions(self, logits, labels):
        """Add style classifier logits and ground-truth label indices for SRA."""
        self.style_predictions.append(logits.cpu())
        self.style_labels.append(labels.cpu())

    def add_skating_ratio(self, joints):
        """
        Compute and store skating ratio for a batch of joint positions.

        Args:
            joints: numpy array [B, T, 22, 3]
        """
        if joints.shape[0] == 0:
            return
        ratios = calculate_skating_ratio(joints)
        self.skating_ratios.extend(ratios.tolist())
    
    def add_mpjpe(self, gt_joints, gen_joints, sample_info=None):
        """Compute and store MPJPE metrics."""
        gt_tensor = torch.from_numpy(gt_joints).float()
        gen_tensor = torch.from_numpy(gen_joints).float()
        
        rr = calculate_rr_mpjpe(gt_tensor, gen_tensor).mean().item()
        glob = calculate_global_mpjpe(gt_tensor, gen_tensor).mean().item()
        
        self.mpjpe_rr.append(rr)
        self.mpjpe_global.append(glob)
        
        if sample_info:
            sample_info['rr_mpjpe'] = rr
            sample_info['global_mpjpe'] = glob
            self.per_sample_results.append(sample_info)
    
    def add_losses(self, latent_loss=None, recon_loss=None):
        """Add loss values."""
        if latent_loss is not None:
            self.latent_mse.append(latent_loss)
        if recon_loss is not None:
            self.recon_mse.append(recon_loss)
    
    def compute_fid(self):
        """Compute FID score from accumulated embeddings."""
        if not self.motion_embeddings_gt or not self.motion_embeddings_gen:
            return None
        
        gt_all = np.concatenate(self.motion_embeddings_gt, axis=0)
        gen_all = np.concatenate(self.motion_embeddings_gen, axis=0)
        
        gt_mu, gt_cov = calculate_activation_statistics(gt_all)
        gen_mu, gen_cov = calculate_activation_statistics(gen_all)
        
        return calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
    
    def compute_diversity(self, num_samples=300):
        """Compute diversity score from generated embeddings."""
        if not self.motion_embeddings_gen:
            return None

        gen_all = np.concatenate(self.motion_embeddings_gen, axis=0)

        if len(gen_all) < num_samples * 2:
            num_samples = len(gen_all) // 2

        if num_samples < 2:
            return None

        # Random sampling for diversity
        idx1 = np.random.choice(len(gen_all), num_samples, replace=False)
        idx2 = np.random.choice(len(gen_all), num_samples, replace=False)

        dist = np.linalg.norm(gen_all[idx1] - gen_all[idx2], axis=1)
        return dist.mean()

    def compute_r_precision(self, top_k=3, R_size=16):
        """
        Compute R-Precision (Top-1, Top-2, Top-3) from CLIP text/motion embeddings.

        Groups samples into batches of R_size, computes euclidean distance matrix
        between text and motion embeddings, and checks if the correct match ranks
        within top-k.
        """
        if not self.text_embeddings_clip or not self.motion_embeddings_clip:
            return None

        all_texts = torch.cat(self.text_embeddings_clip, dim=0)
        all_motions = torch.cat(self.motion_embeddings_clip, dim=0)

        # Normalize for stable distance computation
        all_texts = all_texts / all_texts.norm(dim=1, keepdim=True)
        all_motions = all_motions / all_motions.norm(dim=1, keepdim=True)

        count_seq = all_texts.shape[0]
        if count_seq < R_size:
            print(f"  Warning: Not enough samples ({count_seq}) for R-Precision (need {R_size})")
            return None

        # Shuffle
        shuffle_idx = torch.randperm(count_seq)
        all_texts = all_texts[shuffle_idx]
        all_motions = all_motions[shuffle_idx]

        top_k_mat = torch.zeros(top_k)
        matching_score = 0.0
        num_groups = count_seq // R_size

        for i in range(num_groups):
            group_texts = all_texts[i * R_size:(i + 1) * R_size]
            group_motions = all_motions[i * R_size:(i + 1) * R_size]

            # Euclidean distance matrix [R_size, R_size]
            dist_mat = torch.cdist(group_texts, group_motions, p=2).nan_to_num()

            # Matching score (trace = sum of diagonal = correct pairs)
            matching_score += dist_mat.trace().item()

            # Sort each row and check if correct index is in top-k
            argsmax = torch.argsort(dist_mat, dim=1)
            for k in range(top_k):
                # For each row i, check if i appears in argsmax[i, :k+1]
                top_k_mat[k] += (argsmax[:, :k+1] == torch.arange(R_size).unsqueeze(1)).any(dim=1).sum().item()

        R_count = num_groups * R_size
        results = {}
        for k in range(top_k):
            results[f'R_precision_top_{k+1}'] = float(top_k_mat[k] / R_count)
        results['matching_score'] = float(matching_score / R_count)

        return results

    def compute_sra(self, topk=(1, 3, 5)):
        """
        Compute Style Recognition Accuracy (Top-1, Top-3, Top-5).

        Uses accumulated style classifier logits and ground-truth labels.
        """
        if not self.style_predictions or not self.style_labels:
            return None

        all_predictions = torch.cat(self.style_predictions, dim=0)
        all_labels = torch.cat(self.style_labels, dim=0)

        maxk = max(topk)
        batch_size = all_labels.size(0)

        _, pred = all_predictions.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(all_labels.view(1, -1).expand_as(pred))

        results = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0).item()
            results[f'SRA_top_{k}'] = float(correct_k * 100.0 / batch_size)

        return results

    def get_summary(self):
        """Get summary statistics."""
        results = {}
        
        # FID
        fid = self.compute_fid()
        if fid is not None:
            results['fid'] = float(fid)
        
        # Diversity
        div = self.compute_diversity()
        if div is not None:
            results['diversity'] = float(div)
        
        # MPJPE
        if self.mpjpe_rr:
            results['rr_mpjpe_mean'] = float(np.mean(self.mpjpe_rr))
            results['rr_mpjpe_std'] = float(np.std(self.mpjpe_rr))
        if self.mpjpe_global:
            results['global_mpjpe_mean'] = float(np.mean(self.mpjpe_global))
            results['global_mpjpe_std'] = float(np.std(self.mpjpe_global))
        
        # Losses
        if self.latent_mse:
            results['latent_mse_mean'] = float(np.mean(self.latent_mse))
            results['latent_mse_std'] = float(np.std(self.latent_mse))
        if self.recon_mse:
            results['recon_mse_mean'] = float(np.mean(self.recon_mse))
            results['recon_mse_std'] = float(np.std(self.recon_mse))
        
        # Foot skating ratio
        if self.skating_ratios:
            results['skating_ratio_mean'] = float(np.mean(self.skating_ratios))
            results['skating_ratio_std'] = float(np.std(self.skating_ratios))

        # R-Precision
        r_prec = self.compute_r_precision()
        if r_prec is not None:
            results.update(r_prec)

        # SRA
        sra = self.compute_sra()
        if sra is not None:
            results.update(sra)

        results['num_samples'] = len(self.per_sample_results) if self.per_sample_results else max(len(self.mpjpe_rr), len(self.motion_embeddings_gen)) if (self.mpjpe_rr or self.motion_embeddings_gen) else 0

        return results


#################################################################################
#                           Evaluation Functions                                 #
#################################################################################

def evaluate_base_generation(model, ae, eval_loader, eval_wrapper, device, mean, std, w_schedule, args, output_dir=None, max_vis=20):
    """
    Evaluate base content generation (HumanML3D text → motion, no style).
    Uses the 'latent_humanml' from the combined dataset batch.
    """
    print("\n" + "=" * 70)
    print("BASE GENERATION EVALUATION (HumanML3D, No Style Conditioning)")
    print("=" * 70)
    
    model.eval()
    
    metrics = MetricsAccumulator()
    joints_num = 22
    kinematic_chain = t2m_kinematic_chain
    vis_count = 0
    
    if output_dir:
        vis_dir = pjoin(output_dir, 'base_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        with tqdm(total=len(eval_loader), desc="Base Evaluation") as pbar:
            for batch_idx, batch_data in enumerate(eval_loader):
                # Extract HumanML3D data from combined batch (same as train_MARDM.py)
                z_hml3d = batch_data['latent_humanml'].to(device)
                len_hml3d = batch_data['length_humanml'].to(device)
                text_hml3d = batch_data['text_humanml']
                
                # Generate without style (same as training PASS 1)
                generated_latents = model.generate(
                    conds=text_hml3d,
                    m_lens=len_hml3d,
                    timesteps=args.timesteps,
                    cond_scale=args.cfg_scale,
                    raw_style_latents=None,  # No style
                    style_weight_schedule=None,
                    cfg_mode=args.cfg_mode,
                    cfg_text=args.cfg_text,
                    cfg_style=args.cfg_style,
                )
                
                # Compute latent MSE
                latent_mse = F.mse_loss(generated_latents, z_hml3d).item()
                metrics.add_losses(latent_loss=latent_mse)

                # Decode both for comparison
                generated_motion = ae.decode(generated_latents)
                gt_motion = ae.decode(z_hml3d)

                # Refine generated root trajectory (no-op if refine_net is None)
                generated_motion = eval_wrapper.refine_motion(generated_motion)

                # Compute embeddings for FID (using latent space)
                m_lens_full = len_hml3d * 4

                # Get embeddings for FID
                gt_emb, _ = eval_wrapper.get_motion_embeddings(generated_motion, m_lens_full)
                gen_emb, _ = eval_wrapper.get_motion_embeddings(gt_motion, m_lens_full)
                metrics.add_embeddings(gt_emb, gen_emb)

                # CLIP embeddings for R-Precision (text-motion matching)
                clip_text_emb = eval_wrapper.contrast_model.encode_text(text_hml3d)
                clip_motion_emb = eval_wrapper.contrast_model.encode_motion(generated_motion, m_lens_full)
                metrics.add_clip_embeddings(clip_text_emb, clip_motion_emb)

                # Compute latent MSE
                latent_mse = F.mse_loss(generated_latents, z_hml3d).item()
                recon_mse = F.mse_loss(generated_motion, gt_motion).item()
                metrics.add_losses(latent_loss=latent_mse, recon_loss=recon_mse)

                # Denormalize and compute MPJPE
                gen_np = generated_motion.cpu().numpy() * std + mean
                gt_np = gt_motion.cpu().numpy() * std + mean
                
                # Per-sample metrics
                batch_size = z_hml3d.shape[0]
                batch_gen_joints = []
                for k in range(batch_size):
                    actual_len = min(m_lens_full[k].item(), gen_np.shape[2])

                    # Transpose from [D, T] to [T, D] before recover_from_ric
                    gen_joints = recover_from_ric(torch.from_numpy(gen_np[k]).float(), joints_num).numpy()
                    gt_joints = recover_from_ric(torch.from_numpy(gt_np[k]).float(), joints_num).numpy()

                    # Save the derived joints
                    np.save(pjoin(vis_dir, f'styled_batch{batch_idx}_sample{k}_gen_joints.npy'), gen_joints)
                    np.save(pjoin(vis_dir, f'styled_batch{batch_idx}_sample{k}_gt_joints.npy'), gt_joints)

                    min_len = min(gen_joints.shape[0], gt_joints.shape[0])
                    batch_gen_joints.append(gen_joints[:min_len])

                    sample_info = {
                        'batch_idx': batch_idx,
                        'sample_idx': k,
                        'text': text_hml3d[k] if isinstance(text_hml3d, list) else str(text_hml3d),
                        'length': m_lens_full[k].item(),
                        'latent_mse': F.mse_loss(generated_latents[k], z_hml3d[k]).item()
                    }

                    metrics.add_mpjpe(gt_joints[:min_len], gen_joints[:min_len], sample_info)

                    # Save visualizations
                    if output_dir and vis_count < max_vis:
                        save_path = pjoin(vis_dir, f'styled_batch{batch_idx}_sample{k}.gif')
                        text_str = text_hml3d[k] if isinstance(text_hml3d, list) else "Generated"
                        style_str = "NA"
                        
                        plot_3d_motion_side_by_side(
                            save_path, kinematic_chain,
                            gt_joints[:min_len], gen_joints[:min_len],
                            f"GT ({style_str})", f"Gen ({style_str})",
                            fps=20, text_prompt=text_str[:50], style_label=style_str
                        )
                        vis_count += 1

                # Foot skating ratio for this batch
                if batch_gen_joints:
                    # Pad to same length and stack: [B, T, 22, 3]
                    max_t = max(j.shape[0] for j in batch_gen_joints)
                    padded = [np.pad(j, ((0, max_t - j.shape[0]), (0, 0), (0, 0))) for j in batch_gen_joints]
                    metrics.add_skating_ratio(np.stack(padded, axis=0))

                pbar.update(1)

    results = metrics.get_summary()
    print("\nBase Generation Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return results, metrics.per_sample_results


def evaluate_styled_generation(model, dae, eval_loader, eval_wrapper, device, mean, std, vmodel, processor, w_schedule, args, style_classifier=None, style_to_idx=None, output_dir=None, max_vis=20):
    """
    Evaluate style-conditioned generation (100STYLES text + video → stylized motion).
    Uses 'motion_styled', 'video_styled', etc. from the combined dataset batch.
    
    This matches the training PASS 2 in train_MARDM.py.
    """
    print("\n" + "=" * 70)
    print("STYLED GENERATION EVALUATION (100STYLES Text + Video → Stylized Motion)")
    print("=" * 70)
    
    model.eval()
    dae.eval()
    
    metrics = MetricsAccumulator()
    joints_num = 22
    kinematic_chain = t2m_kinematic_chain
    vis_count = 0
    
    if output_dir:
        vis_dir = pjoin(output_dir, 'styled_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        with tqdm(total=len(eval_loader), desc="Styled Evaluation") as pbar:
            for batch_idx, batch_data in enumerate(eval_loader):
                # Extract 100STYLES data from combined batch (same as train_MARDM.py PASS 2)
                motion_style = batch_data['motion_styled'].float().to(device)
                len_style = batch_data['length_styled'].to(device) // 4
                text_style = batch_data['text_styled']
                video_style = batch_data['video_styled']
                style_names = batch_data.get('style_name', ['unknown'] * motion_style.shape[0])
                
                # Process video through video encoder (same as train_MARDM.py)
                inputs = processor(video_style, return_tensors="pt").to(device)
                vid_tensors = vmodel(**inputs).last_hidden_state
                
                # Encode motion and get style latents (same as train_MARDM.py)
                z_style, raw_video_latents = dae.encode(motion_style, vid_tensors)
                
                # Generate with style (same as training)
                generated_latents = model.generate(
                    conds=text_style,
                    m_lens=len_style,
                    timesteps=args.timesteps,
                    cond_scale=args.cfg_scale,
                    raw_style_latents=raw_video_latents,
                    style_weight_schedule=w_schedule,
                    cfg_mode=args.cfg_mode,
                    cfg_text=args.cfg_text,
                    cfg_style=args.cfg_style,
                )
                
                # Decode both for comparison
                generated_motion = dae.decode(generated_latents)

                # Refine generated root trajectory (no-op if refine_net is None)
                generated_motion = eval_wrapper.refine_motion(generated_motion)

                # Compute metrics
                m_lens_full = len_style * 4
                
                # Get embeddings for FID
                gt_emb, _ = eval_wrapper.get_motion_embeddings(motion_style, m_lens_full)
                gen_emb, _ = eval_wrapper.get_motion_embeddings(generated_motion, m_lens_full)
                metrics.add_embeddings(gt_emb, gen_emb)

                # CLIP embeddings for R-Precision (text-motion matching)
                clip_text_emb = eval_wrapper.contrast_model.encode_text(text_style)
                clip_motion_emb = eval_wrapper.contrast_model.encode_motion(generated_motion, m_lens_full)
                metrics.add_clip_embeddings(clip_text_emb, clip_motion_emb)

                # SRA: run style classifier on generated motions
                if style_classifier is not None and style_to_idx is not None:
                    # dae.decode() already returns [B, T, D] which is what the classifier expects
                    gen_for_cls = generated_motion
                    style_logits = style_classifier(gen_for_cls, stage="Classification")
                    gt_style_indices = torch.tensor(
                        [style_to_idx.get(s, 0) for s in style_names], device=device
                    )
                    metrics.add_style_predictions(style_logits, gt_style_indices)

                # Compute losses
                latent_mse = F.mse_loss(generated_latents, z_style).item()
                recon_mse = F.mse_loss(generated_motion, motion_style).item()
                metrics.add_losses(latent_mse, recon_mse)
                
                # Denormalize and compute MPJPE
                gen_np = generated_motion.cpu().numpy() * std + mean
                gt_np = motion_style.cpu().numpy() * std + mean

                
                batch_size = motion_style.shape[0]
                batch_gen_joints = []
                for k in range(batch_size):
                    actual_len = min(m_lens_full[k].item(), gen_np.shape[2])

                    # Transpose from [D, T] to [T, D] before recover_from_ric
                    gen_joints = recover_from_ric(torch.from_numpy(gen_np[k]).float(), joints_num).numpy()
                    gt_joints = recover_from_ric(torch.from_numpy(gt_np[k]).float(), joints_num).numpy()

                    # Save the derived joints
                    np.save(pjoin(vis_dir, f'styled_batch{batch_idx}_sample{k}_gen_joints.npy'), gen_joints)
                    np.save(pjoin(vis_dir, f'styled_batch{batch_idx}_sample{k}_gt_joints.npy'), gt_joints)

                    min_len = min(gen_joints.shape[0], gt_joints.shape[0])
                    batch_gen_joints.append(gen_joints[:min_len])

                    sample_info = {
                        'batch_idx': batch_idx,
                        'sample_idx': k,
                        'text': text_style[k] if isinstance(text_style, list) else str(text_style),
                        'style': style_names[k] if k < len(style_names) else 'unknown',
                        'length': actual_len
                    }
                    metrics.add_mpjpe(gt_joints[:min_len], gen_joints[:min_len], sample_info)

                    # Save visualizations
                    if output_dir and vis_count < max_vis:
                        save_path = pjoin(vis_dir, f'styled_batch{batch_idx}_sample{k}.gif')
                        text_str = text_style[k] if isinstance(text_style, list) else "Generated"
                        style_str = style_names[k] if k < len(style_names) else "unknown"

                        plot_3d_motion_side_by_side(
                            save_path, kinematic_chain,
                            gt_joints[:min_len], gen_joints[:min_len],
                            f"GT ({style_str})", f"Gen ({style_str})",
                            fps=20, text_prompt=text_str[:50], style_label=style_str
                        )
                        vis_count += 1

                # Foot skating ratio for this batch
                if batch_gen_joints:
                    max_t = max(j.shape[0] for j in batch_gen_joints)
                    padded = [np.pad(j, ((0, max_t - j.shape[0]), (0, 0), (0, 0))) for j in batch_gen_joints]
                    metrics.add_skating_ratio(np.stack(padded, axis=0))

                pbar.update(1)

    results = metrics.get_summary()
    print("\nStyled Generation Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return results, metrics.per_sample_results


def evaluate_style_transfer(model, dae, ae, eval_loader, eval_wrapper, device, mean, std, vmodel, processor, w_schedule, args, style_classifier=None, style_to_idx=None, output_dir=None, max_vis=30):
    """
    Evaluate cross-modal style transfer:
    - Text prompts from HumanML3D (content)
    - Video styles from 100STYLES (style)
    
    This tests generalization: can the model apply a style to unseen content?
    Since there's no ground truth, we evaluate:
    1. Generation quality (no NaN, reasonable motion)
    2. Diversity of outputs
    3. Qualitative visualizations
    """
    print("\n" + "=" * 70)
    print("STYLE TRANSFER EVALUATION (HumanML3D Text + 100STYLES Video)")
    print("=" * 70)
    
    model.eval()
    dae.eval()
    
    metrics = MetricsAccumulator()
    joints_num = 22
    kinematic_chain = t2m_kinematic_chain
    vis_count = 0
    transfer_samples = []
    
    if output_dir:
        vis_dir = pjoin(output_dir, 'transfer_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        with tqdm(total=len(eval_loader), desc="Style Transfer") as pbar:
            for batch_idx, batch_data in enumerate(tqdm(eval_loader, desc="Style Transfer")):
                if batch_idx >= args.max_transfer_batches:
                    break
                
                # Extract HumanML3D text (content)
                text_hml3d = batch_data['text_humanml']
                len_hml3d = batch_data['length_humanml'].to(device) 
                
                # Extract 100STYLES video (style)
                motion_style = batch_data['motion_styled'].float().to(device)
                video_style = batch_data['video_styled']
                style_names = batch_data.get('style_name', ['unknown'] * motion_style.shape[0])
                
                # Match batch sizes (they should already match from mld_collate_paired)
                min_batch = min(len(text_hml3d), motion_style.shape[0])
                text_hml3d = text_hml3d[:min_batch]
                len_hml3d = len_hml3d[:min_batch]
                motion_style = motion_style[:min_batch]
                video_style = video_style[:min_batch]
                style_names = style_names[:min_batch]
                
                # Process video to get style latents
                inputs = processor(video_style, return_tensors="pt").to(device)
                vid_tensors = vmodel(**inputs).last_hidden_state
                raw_video_latents = dae.encode_video(vid_tensors)
                
                # Generate with cross-modal conditioning
                # (HumanML3D text + 100STYLES video style)
                generated_latents = model.generate(
                    conds=text_hml3d,  # Text from HumanML3D
                    m_lens=len_hml3d,
                    timesteps=args.timesteps,
                    cond_scale=args.cfg_scale,
                    raw_style_latents=raw_video_latents,  # Style from 100STYLES
                    style_weight_schedule=w_schedule,
                    cfg_mode=args.cfg_mode,
                    cfg_text=args.cfg_text,
                    cfg_style=args.cfg_style,
                )
                
                # Decode
                generated_motion1 = dae.decode(generated_latents)
                generated_motion2 = ae.decode(generated_latents)

                # Refine generated root trajectories (no-op if refine_net is None)
                generated_motion1 = eval_wrapper.refine_motion(generated_motion1)
                generated_motion2 = eval_wrapper.refine_motion(generated_motion2)
                
                # Compute embeddings for diversity
                m_lens_full = len_hml3d * 4
                gen_emb1, _ = eval_wrapper.get_motion_embeddings(generated_motion1, m_lens_full)
                gen_emb2, _ = eval_wrapper.get_motion_embeddings(generated_motion2, m_lens_full)
                metrics.add_embeddings(gen_emb1, gen_emb2)  # No GT for transfer

                # SRA: does the transferred motion reflect the intended style?
                if style_classifier is not None and style_to_idx is not None:
                    # Generated motion is already normalized; permute [B, D, T] -> [B, T, D]
                    gen_for_cls = generated_motion1
                    style_logits = style_classifier(gen_for_cls, stage="Classification")
                    gt_style_indices = torch.tensor(
                        [style_to_idx.get(s, 0) for s in style_names], device=device
                    )
                    metrics.add_style_predictions(style_logits, gt_style_indices)

                # Denormalize
                gen_np1 = generated_motion1.cpu().numpy() * std + mean
                gen_np2 = generated_motion2.cpu().numpy() * std + mean
                gt_style_np = motion_style.cpu().numpy() * std + mean
                
                batch_size = generated_motion1.shape[0]
                batch_gen_joints = []
                for k in range(batch_size):
                    actual_len = min(m_lens_full[k].item(), gen_np1.shape[2])

                    gen_joints1 = recover_from_ric(torch.from_numpy(gen_np1[k]).float(), joints_num).numpy()
                    gen_joints2 = recover_from_ric(torch.from_numpy(gen_np2[k]).float(), joints_num).numpy()
                    gt_style_joints = recover_from_ric(torch.from_numpy(gt_style_np[k]).float(), joints_num).numpy()

                    np.save(pjoin(vis_dir, f'transfer_batch{batch_idx}_sample{k}_gen_joints_dae.npy'), gen_joints1)
                    np.save(pjoin(vis_dir, f'transfer_batch{batch_idx}_sample{k}_gen_joints_ae.npy'), gen_joints2)
                    np.save(pjoin(vis_dir, f'transfer_batch{batch_idx}_sample{k}_gt_style_joints.npy'), gt_style_joints)

                    min_len = min(gen_joints1.shape[0], gen_joints2.shape[0], gt_style_joints.shape[0])
                    batch_gen_joints.append(gen_joints1[:min_len])

                    sample_info = {
                        'batch_idx': batch_idx,
                        'sample_idx': k,
                        'text': text_hml3d[k] if isinstance(text_hml3d, list) else str(text_hml3d),
                        'style': style_names[k] if k < len(style_names) else 'unknown',
                        'length': actual_len
                    }
                    transfer_samples.append(sample_info)

                    # Save visualizations
                    if output_dir and vis_count < max_vis:
                        save_path = pjoin(vis_dir, f'transfer_batch{batch_idx}_sample{k}.gif')
                        text_str = text_hml3d[k] if isinstance(text_hml3d, list) else "Generated"
                        style_str = style_names[k] if k < len(style_names) else "unknown"
                        
                        # plot_3d_motion_gif(
                        #     save_path, kinematic_chain, gen_joints,
                        #     title=f"Transfer: {style_str}",
                        #     fps=20, text_prompt=text_str[:50], style_label=style_str
                        # )

                        plot_3d_motion_three_way2(
                            save_path, kinematic_chain,
                            gt_style_joints[:min_len], gen_joints1[:min_len], gen_joints2[:min_len],
                            f"Style GT ({style_str})", f"Gen DAE ({style_str})", f"Gen AE ({style_str})",
                            fps=20, text_prompt=text_str[:50], style_label=style_str
                        )
                        
                        # Save motion data
                        # np.save(pjoin(vis_dir, f'transfer_batch{batch_idx}_sample{k}_joints.npy'), gen_joints)
                        vis_count += 1

                # Foot skating ratio for this batch
                if batch_gen_joints:
                    max_t = max(j.shape[0] for j in batch_gen_joints)
                    padded = [np.pad(j, ((0, max_t - j.shape[0]), (0, 0), (0, 0))) for j in batch_gen_joints]
                    metrics.add_skating_ratio(np.stack(padded, axis=0))

                pbar.update(1)

    # Calculate diversity and cast to standard python float if it exists
    div_score = metrics.compute_diversity()
    
    # SRA for transfer
    sra = metrics.compute_sra()

    # Compute transfer-specific metrics
    results = {
        'num_samples': len(transfer_samples),
        'nan_rate': sum(1 for s in transfer_samples if s.get('has_nan', False)) / len(transfer_samples) if transfer_samples else 0,
        'diversity': float(div_score) if div_score is not None else None
    }
    if sra is not None:
        results.update(sra)
    if metrics.skating_ratios:
        results['skating_ratio_mean'] = float(np.mean(metrics.skating_ratios))
        results['skating_ratio_std'] = float(np.std(metrics.skating_ratios))
    
    print("\nStyle Transfer Results:")
    for k, v in results.items():
        if v is not None:
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    return results, transfer_samples


#################################################################################
#                                    Main                                        #
#################################################################################

def main(args):
    # Set seed (same as train_MARDM.py)
    set_seed(args.seed)
    
    # Device setup
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(args.device)}")
    
    # Enable TF32 for faster computation (same as train_MARDM.py)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    #################################################################################
    #                    Data Setup                                                 #
    #################################################################################
    print("\n" + "=" * 70)
    print("LOADING DATA (Same strategy as train_MARDM.py)")
    print("=" * 70)
    
    # Paths - exactly as in train_MARDM.py
    data_root = f'{args.dataset_dir}/100STYLE-SMPL/'
    prior_data_root = f'{args.dataset_dir}/HumanML3D/'
    dim_pose = 67
    
    # 100STYLES paths
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    video_dir = pjoin(data_root, 'videos')
    text_dir = pjoin(data_root, 'texts')
    mean = np.load(pjoin(data_root, 'Mean.npy'))[:dim_pose]
    std = np.load(pjoin(data_root, 'Std.npy'))[:dim_pose]
    dict_file = pjoin(data_root, '100STYLE_name_dict_length.txt')
    val_split_file = pjoin(data_root, 'test_100STYLE_Filter.txt')
    
    # HumanML3D paths
    prior_motion_dir = pjoin(prior_data_root, 'sliced_joint_vecs')
    prior_latent_dir = pjoin(prior_data_root, 'latent_vecs')
    prior_text_dir = pjoin(prior_data_root, 'splits_sliced/texts_sliced')
    prior_mean = np.load(pjoin(prior_data_root, 'Mean.npy'))[:dim_pose]
    prior_std = np.load(pjoin(prior_data_root, 'Std.npy'))[:dim_pose]
    prior_val_split_file = pjoin(prior_data_root, 'splits_sliced/val.txt')
    
    # Create combined dataset (same as train_MARDM.py)
    print("Initializing Dataset...")
    val_dataset_full = Text2MotionDatasetCombined_v4(
        style_mean=mean, style_std=std, style_split_file=val_split_file, 
        style_motion_dir=motion_dir, style_text_dir=text_dir, style_video_dir=video_dir, style_dict_file=dict_file,
        humanml_mean=prior_mean, humanml_std=prior_std, humanml_split_file=prior_val_split_file, humnaml_motion_dir=prior_motion_dir,
        humanml_latent_dir=prior_latent_dir, humanml_text_dir=prior_text_dir, humanml_dict_file=pjoin(prior_data_root, 'splits_sliced/all_lengths.txt'),
        dim_pose=dim_pose, unit_length=args.unit_length, max_motion_length=args.max_motion_length, epoch_mode='100styles',
    )
    
    # Split into val/test (same as train_MARDM.py)
    val_size = len(val_dataset_full) * 2 // 3
    test_size = len(val_dataset_full) - val_size
    
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_dataset_full,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)  # Same seed for reproducibility
    )
    
    print(f"Dataset loaded - full val: {len(val_dataset_full)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    
    # Create DataLoaders (same as train_MARDM.py)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=mld_collate_paired
    )
    eval_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=mld_collate_paired
    )
    
    print(f"DataLoaders initialized. Val batches: {len(val_loader)}, Test batches: {len(eval_loader)}")
    
    #################################################################################
    #                                Model Setup                                    #
    #################################################################################
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)
    
    num_classes = len(args.styles) if args.styles else 100

    # Load AE for HumanML3D t2m evaluation
    print("\nLoading AE for HumanML3D...")
    ae = AE_models["AE_Model"](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, 't2m', 'AE', 'model', 'latest.tar'), map_location=device)
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])
    ae.to(device).eval()
    
    # Load DAE (same as train_MARDM.py)
    print("\nLoading DAE for 100STYLES...")
    dae = DAE_models[args.ae_model](window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose)
    dae_ckpt_path = args.dae_ckpt or pjoin(args.checkpoints_dir, '100styles', args.ae_name, 'final_finetune_diffmlp_decoder_fixed_hybrid_ema_fix.tar')
    print(f"  Loading from: {dae_ckpt_path}")
    dae_ckpt = torch.load(dae_ckpt_path, map_location=device, weights_only=False)
    dae.load_state_dict(dae_ckpt['ae'])
    dae.to(device).eval()
    
    # Load Video Encoder (same as train_MARDM.py)
    print(f"\nLoading video encoder ({args.video_encoder})...")
    MODEL_CONFIG = {
        'vivit': {"name": "google/vivit-b-16x2-kinetics400", "processor": "google/vivit-b-16x2-kinetics400"},
        'timesformer': {"name": "facebook/timesformer-base-finetuned-k400", "processor": "MCG-NJU/videomae-base"},
    }
    
    if args.video_encoder == 'vivit':
        processor = VivitImageProcessor.from_pretrained(MODEL_CONFIG['vivit']['processor'])
        vmodel = VivitModel.from_pretrained(MODEL_CONFIG['vivit']['name']).to(device)
    elif args.video_encoder == 'timesformer':
        processor = AutoProcessor.from_pretrained(MODEL_CONFIG['timesformer']['processor'])
        vmodel = TimesformerModel.from_pretrained(MODEL_CONFIG['timesformer']['name']).to(device)
    
    vmodel.eval()
    for p in vmodel.parameters():
        p.requires_grad = False
    
    # Load MARDM (same structure as train_MARDM.py)
    print(f"\nLoading MARDM ({args.model})...")
    mardm = MARDM_models[args.model](
        ae_dim=dae.output_emb_width, 
        cond_mode='text',
        style_routing=args.style_routing,
        style_dim=512
    )

    mardm_ckpt_path = args.mardm_ckpt or pjoin(args.checkpoints_dir, 't2m', args.model, 'model', 'final_diffmlp_diff_v4_cfg_cross_batch_hybrid_ema_fix.tar')
    ckpt_stem = Path(mardm_ckpt_path).name.removesuffix('.tar') 
    _, w_schedule1, style_router = load_checkpoint(mardm, mardm_ckpt_path, device, key=args.checkpoint_key)
    mardm.to(device).eval()
    
    # Create style weight schedule (same as train_MARDM.py)
    w_schedule = get_style_weight_schedule(mardm, args)
    print(f"Style weight schedule: {len(w_schedule)} blocks, range [{w_schedule[0]:.2f}, {w_schedule[-1]:.2f}]")
    
    # Optional trajectory refinement net (mirrors sample_new.py)
    refine_net = None
    if args.use_refinement:
        print(f"\nLoading Trajectory Refinement Net from: {args.refinement_ckpt}")
        refine_net = TrajectoryRefinementNet(
            input_feats=130, output_feats=4,
            width=args.refinement_width,
            depth=args.refinement_depth,
            dilation_growth_rate=args.refinement_dilation_growth_rate,
            dropout=0.0,
        )
        refine_ckpt = torch.load(args.refinement_ckpt, map_location=device, weights_only=False)
        refine_state = refine_ckpt.get('model', refine_ckpt)
        refine_net.load_state_dict(refine_state)
        refine_net.to(device).eval()
        print(f"  Refinement net loaded (epoch {refine_ckpt.get('epoch', '?')}).")

    # Load evaluation wrapper
    print("\nLoading evaluation wrapper...")
    eval_wrapper = Evaluators('t2m', device=device)
    eval_wrapper.set_refinement_net(refine_net)

    # Load style classifier for SRA
    style_classifier = None
    style_to_idx = None
    # Default checkpoint path matches train_MARDM.py
    sc_ckpt_path = args.style_classifier_ckpt or pjoin(args.checkpoints_dir, 'style_classifier', 'style_classifier_final.pt')
    if os.path.exists(sc_ckpt_path):
        print(f"\nLoading style classifier from: {sc_ckpt_path}")
        sc_ckpt = torch.load(sc_ckpt_path, map_location=device, weights_only=False)

        # Build style-to-index mapping, preserving the training-time order.
        # Priority: (1) explicit style_to_idx in checkpoint, (2) args.styles from checkpoint
        # (training order), (3) dataset style names, (4) sorted eval args.styles.
        # Using sorted() for the fallback causes index mismatches because the training
        # order in args.styles is non-alphabetical (e.g. Aeroplane,Chicken,Robot,Superman,
        # ArmsFolded,...) while sorted() gives Aeroplane,ArmsFolded,Chicken,...  -- only
        # Aeroplane (index 0) would ever match, producing ~11-19% SRA regardless of quality.
        if 'style_to_idx' in sc_ckpt:
            style_to_idx = sc_ckpt['style_to_idx']
            print("  style_to_idx loaded directly from checkpoint.")
        elif isinstance(sc_ckpt.get('args'), dict) and sc_ckpt['args'].get('styles'):
            # Reconstruct from the training-time styles list (preserves original order)
            train_styles = sc_ckpt['args']['styles']
            style_to_idx = {s: i for i, s in enumerate(train_styles)}
            print(f"  style_to_idx rebuilt from checkpoint args.styles ({len(train_styles)} classes, training order).")
        else:
            # Last resort: use dataset or eval args -- NOTE: sorted() breaks non-alpha training order
            style_names_all = list(val_dataset_full.get_style_names()) if hasattr(val_dataset_full, 'get_style_names') else list(args.styles) if args.styles else None
            if style_names_all:
                style_to_idx = {name: i for i, name in enumerate(style_names_all)}
                print(f"  Warning: style_to_idx built from eval dataset/args (order may not match training).")
            else:
                print("  Warning: Could not determine style-to-index mapping. SRA will be skipped.")

        if style_to_idx is not None:
            style_classifier = StyleClassification(
                nclasses=args.style_cls_nclasses, input_dim=dim_pose,
                latent_dim=[1, args.style_cls_latent_dim], ff_size=args.style_cls_ff_size,
                num_layers=args.style_cls_num_layers, num_heads=args.style_cls_num_heads,
                dropout=args.style_cls_dropout
            )
            state_key = 'model_state_dict' if 'model_state_dict' in sc_ckpt else 'state_dict'
            style_classifier.load_state_dict(sc_ckpt[state_key])
            style_classifier.to(device).eval()
            for p in style_classifier.parameters():
                p.requires_grad = False
            print(f"  Style classifier loaded: {args.style_cls_nclasses} classes")
    else:
        print(f"\nStyle classifier checkpoint not found at: {sc_ckpt_path}. SRA metrics will be skipped.")

    #################################################################################
    #                               Run Evaluation                                  #
    #################################################################################'=
    print("\n" + "=" * 70)
    print(f"RUNNING EVALUATION (mode: {args.eval_mode})")
    print("=" * 70)

    # start_time = time.time()
    
    # Create output directory
    if args.output_dir:
        output_dir = pjoin(args.output_dir, ckpt_stem)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = pjoin('./evaluation_results', f'{args.model}_{args.style_routing}_{timestamp}')
        # output_dir = pjoin('./evaluation_results', f'{args.model}_{style_router}_{timestamp}')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    all_results = {'config': vars(args)}
    start_time = time.time()
    
    # Choose which loader to use
    loader_to_use = eval_loader if args.use_test_set else val_loader
    print(f"Using {'test' if args.use_test_set else 'validation'} set ({len(loader_to_use)} batches)")
    
    # Base evaluation (HumanML3D, no style)
    if args.eval_mode in ['base', 'full']:
        base_results, base_samples = evaluate_base_generation(
            mardm, ae, loader_to_use, eval_wrapper, device,
            prior_mean, prior_std, w_schedule, args,
            output_dir=output_dir, max_vis=args.max_visualizations
        )
        all_results['base'] = base_results
    
    # Styled evaluation (100STYLES with style)
    if args.eval_mode in ['styled', 'full']:
        styled_results, styled_samples = evaluate_styled_generation(
            mardm, dae, loader_to_use, eval_wrapper, device,
            mean, std, vmodel, processor, w_schedule, args,
            style_classifier=style_classifier, style_to_idx=style_to_idx,
            output_dir=output_dir, max_vis=args.max_visualizations
        )
        all_results['styled'] = styled_results

    # Transfer evaluation (cross-modal)
    if args.eval_mode in ['transfer', 'full']:
        transfer_results, transfer_samples = evaluate_style_transfer(
            mardm, dae, ae, loader_to_use, eval_wrapper, device,
            mean, std, vmodel, processor, w_schedule, args,
            style_classifier=style_classifier, style_to_idx=style_to_idx,
            output_dir=output_dir, max_vis=args.max_visualizations
        )
        all_results['transfer'] = transfer_results
    
    # Save results
    total_time = time.time() - start_time
    all_results['total_time_seconds'] = total_time
    print(f"\nTotal evaluation time: {total_time/60:.2f} minutes")
    
    results_path = pjoin(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n--- SUMMARY ---")
    for mode, results in all_results.items():
        if mode in ['config', 'total_time_seconds']:
            continue
        print(f"\n{mode.upper()}:")
        if isinstance(results, dict):
            for k, v in results.items():
                if v is not None:
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MARDM with Dual-AdaLN Style Conditioning")
    
    # Model arguments
    # parser.add_argument('--model', type=str, default='MARDM-SiT-XL', choices=['MARDM-DDPM-XL', 'MARDM-SiT-XL'])
    parser.add_argument('--model', type=str, default='MARDM-DDPM-XL', choices=['MARDM-DDPM-XL', 'MARDM-SiT-XL'])
    parser.add_argument('--mardm_ckpt', type=str, default=None,
                        help='Stage-2 MARDM checkpoint. Default: <checkpoints_dir>/t2m/<model>/model/final_diffmlp_diff_v4_cfg_cross_batch_hybrid_ema_fix.tar')
    parser.add_argument('--dae_ckpt', type=str, default=None,
                        help='Stage-2 fine-tuned DualAE. Default: <checkpoints_dir>/100styles/<ae_name>/final_finetune_diffmlp_decoder_fixed_hybrid_ema_fix.tar')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Path to MARDM and DAE checkpoint')
    parser.add_argument('--checkpoint_key', type=str, default='ema_mardm', help='Key in checkpoint dict for model weights')
    
    # parser.add_argument('--ae_name', type=str, default='AE')
    # parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--ae_name', type=str, default='DAE')
    parser.add_argument('--ae_model', type=str, default='DAE_Model')
    parser.add_argument('--window_size', type=int, default=64)
    
    parser.add_argument('--style_routing', type=str, default='diffmlp', choices=['diffmlp', 'mart'])
    # parser.add_argument('--use_weight_schedule', action='store_true', help='Use gradual weight schedule for style injection')
    parser.add_argument('--use_weight_schedule', action=argparse.BooleanOptionalAction, default=True,
                        help='Block-wise linear style weight w in [0,1] across DiffMLP blocks (thesis Eq. 4.11-4.13). '
                             'ON by default; disable with --no-use_weight_schedule.')
    parser.add_argument('--mart_style_weight', type=float, default=1.0)
    
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--max_motion_length', type=int, default=196)
    parser.add_argument('--unit_length', type=int, default=4)
    parser.add_argument('--styles', type=str, nargs='+', default=["Aeroplane", "Chicken", "Robot", "Superman", "ArmsFolded"])
    parser.add_argument('--tiny', action='store_true', help='Use tiny dataset mode (same as train_MARDM.py)')
    
    # Style classifier arguments (for SRA metric) — defaults match train_MARDM.py
    parser.add_argument('--style_classifier_ckpt', type=str, default=None, help='Path to pre-trained style classifier checkpoint. Defaults to checkpoints/style_classifier/style_classifier_final.pt')
    parser.add_argument('--style_cls_latent_dim', type=int, default=512, help='Style classifier hidden dimension')
    parser.add_argument('--style_cls_ff_size', type=int, default=1024, help='Style classifier feedforward size')
    parser.add_argument('--style_cls_num_layers', type=int, default=6, help='Style classifier transformer layers')
    parser.add_argument('--style_cls_num_heads', type=int, default=4, help='Style classifier attention heads')
    parser.add_argument('--style_cls_dropout', type=float, default=0.1, help='Style classifier dropout')
    # parser.add_argument('--style_cls_nclasses', type=int, default=20, help='Number of style classes in the classifier')
    parser.add_argument('--style_cls_nclasses', type=int, default=21, help='Number of style classes in the classifier (released checkpoint: 21)')

    parser.add_argument('--video_encoder', type=str, default='vivit', choices=['vivit', 'timesformer'])
    parser.add_argument('--eval_mode', type=str, default='full', choices=['base', 'styled', 'transfer', 'full'], help='Evaluation mode')
    parser.add_argument('--use_test_set', action='store_true', help='Use test set instead of validation set')
    parser.add_argument('--timesteps', type=int, default=18, help='Number of generation timesteps')
    parser.add_argument('--cfg_scale', type=float, default=4.5, help='Classifier-free guidance scale (used in 2-way mode)')
    parser.add_argument('--cfg_mode', type=str, default='2way',
                        choices=['2way', '3way_additive', '3way_style_first'],
                        help='CFG mode: 2way (standard), 3way_additive (independent text+style), 3way_style_first (sequential)')
    parser.add_argument('--cfg_text', type=float, default=None,
                        help='Text CFG scale for 3-way modes (defaults to --cfg_scale when None)')
    parser.add_argument('--cfg_style', type=float, default=None,
                        help='Style CFG scale for 3-way modes (defaults to --cfg_scale when None)')
    parser.add_argument('--max_transfer_batches', type=int, default=50, help='Max batches for transfer evaluation')
    parser.add_argument('--max_visualizations', type=int, default=20, help='Max visualizations to save per evaluation mode')

    # Trajectory refinement (mirrors sample_new.py)
    parser.add_argument('--use_refinement', action='store_true',
                        help='Apply TrajectoryRefinementNet to decoded generated motions before metric computation.')
    parser.add_argument('--refinement_ckpt', type=str,
                        default='./checkpoints/refinement/refine_nofoot/best.tar',
                        help='Path to refinement net checkpoint (.tar with "model" state dict).')
    parser.add_argument('--refinement_width', type=int, default=512)
    parser.add_argument('--refinement_depth', type=int, default=3)
    parser.add_argument('--refinement_dilation_growth_rate', type=int, default=3)

    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')
    
    # Runtime arguments
    parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 4,
                        help='DataLoader workers (default 0 on Windows, where worker processes can deadlock)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    
    args = parser.parse_args()
    main(args)