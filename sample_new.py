"""
Sampling Script for MARDM with Dual-AdaLN Style Conditioning

STANDARD MODES:
1. Text-only generation: Base content, no style.
2. Single Style generation: Text + 1 Video style.
3. Quad Generation: 4 variations (Uncond, Text-only, Style-only, Both) using 1 Video.
4. Style Interpolation: Blends 2 videos using a specific weight.

EXPERIMENT MODES (8.1-8.7, excluding 8.4):
--exp_reverse_decoder    : 8.1 - Decode neutral latents with both DAE and AE
--exp_decoder_interp     : 8.2 - Blend decoder outputs for style strength control  
--exp_content_preserve   : 8.3 - Same style + different texts → verify content differs
--exp_per_style          : 8.5 - Generate for each style video in a directory
--exp_schedule_ablation  : 8.6 - Test different weight schedules
--exp_cfg_sweep          : 8.7 - Sweep CFG scales

Usage:
    # Standard modes
    python sample_new.py --text_prompt "a person walks" --generate_quad --style_video robot.mp4
    
    # Experiment modes
    python sample_new.py --exp_reverse_decoder --text_prompt "a person walks"
    python sample_new.py --exp_decoder_interp --style_video robot.mp4 --text_prompt "a person walks"
"""

import os
import sys
from os.path import join as pjoin
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import random
import argparse
import time
import json
from tqdm import tqdm
from pathlib import Path
from glob import glob

# Model imports
from models.AE import DAE_models, AE_models
from models.MARDM import MARDM_models
from models.LengthEstimator import LengthEstimator
from models.refinement import TrajectoryRefinementNet

# Utility imports
from utils.motion_process import (
    recover_from_ric,
    kit_kinematic_chain,
    t2m_kinematic_chain,
    plot_3d_motion_gif,
    plot_3d_motion_side_by_side
)
from utils.foot_contact import compute_velocities, detect_foot_contact

# Video encoder imports
from transformers import VivitModel, TimesformerModel, VivitImageProcessor, AutoProcessor

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


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
    """Load model checkpoint with flexible key handling."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Load on CPU: the file also holds optimizer state and a second model copy (~7.8 GB); only the
    # selected weights should reach the GPU (load_state_dict copies onto the model's device).
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    keys_to_try = [key, 'ema_mardm', 'mardm', 'model', 'state_dict']
    state_dict = None
    
    for k in keys_to_try:
        if k in checkpoint:
            state_dict = checkpoint[k]
            print(f"  Using weights from key: '{k}'")
            break
    
    if state_dict is None:
        state_dict = checkpoint
        print("  Using checkpoint directly as state_dict")
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_filtered = [k for k in missing if not k.startswith('clip_model.')]
    
    if missing_filtered:
        print(f"  Warning: {len(missing_filtered)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    
    return checkpoint.get('ep', 0)


def get_style_weight_schedule(model, args=None, schedule_type=None):
    """Generate style weight schedule based on routing mode or schedule type."""
    if args and args.style_routing == 'mart':
        num_blocks = len(model.MARTransformer)
        return [args.mart_style_weight] * num_blocks
    
    # DiffMLP mode
    num_blocks = model.DiffMLPs.get_total_blocks()
    
    # Use schedule_type if provided, otherwise use args
    if schedule_type is None and args:
        schedule_type = 'linear' if args.use_weight_schedule else 'uniform'
    elif schedule_type is None:
        schedule_type = 'uniform'
    
    if schedule_type == 'linear':
        return np.linspace(0.0, 1.0, num_blocks).tolist()
    elif schedule_type == 'linear_0.2_1':
        return np.linspace(0.2, 1.0, num_blocks).tolist()
    elif schedule_type == 'uniform':
        return [1.0] * num_blocks
    elif schedule_type == 'style_blocks_only':
        return [0.0] * 16 + [1.0] * 8
    elif schedule_type == 'two_phase':
        return list(np.linspace(0.2, 0.6, 16)) + [1.0] * 8
    elif schedule_type == 'cosine':
        return np.sin(np.linspace(0, np.pi/2, num_blocks)).tolist()
    elif schedule_type == 'none':
        return None
    else:
        return [1.0] * num_blocks


def load_video_frames(video_path, num_frames=32, target_size=(224, 224)):
    """Load video frames from a file."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) required for video loading.")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Could not read frames from video: {video_path}")
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))
    
    return frames


def extract_style_latents(video_path, dae, vmodel, processor, device, dim_pose=67):
    """Extract style latents from a video file."""
    video_frames = load_video_frames(video_path)
    inputs = processor([video_frames], return_tensors="pt").to(device)
    
    with torch.no_grad():
        vid_tensors = vmodel(**inputs).last_hidden_state
        raw_style_latents = dae.encode_video(vid_tensors)
    
    return raw_style_latents


def blend_style_latents(style_a, style_b, weight=0.5):
    """Blend two style latent tensors via linear interpolation."""
    return (1 - weight) * style_a + weight * style_b


def compute_motion_distance(motion1, motion2):
    """Compute L2 distance between two motion sequences."""
    min_len = min(len(motion1), len(motion2))
    return np.mean(np.linalg.norm(motion1[:min_len] - motion2[:min_len], axis=-1))


#################################################################################
#                              Core Generation Functions                         #
#################################################################################

def generate_latent(model, text_prompt, m_length, device,
                    raw_style_latents=None, w_schedule=None, args=None):
    """Generate latent only (without decoding)."""
    model.eval()
    m_lens = torch.tensor([m_length // 4], device=device)

    # Resolve 3-way CFG scales: fall back to cfg_scale if not explicitly set.
    cfg_mode = getattr(args, 'cfg_mode', '2way')
    cfg_text = getattr(args, 'cfg_text', None)
    cfg_style = getattr(args, 'cfg_style', None)
    if cfg_mode != '2way':
        if cfg_text is None:
            cfg_text = args.cfg_scale
        if cfg_style is None:
            cfg_style = args.cfg_scale

    t0 = time.perf_counter()
    with torch.no_grad():
        pred_latents = model.generate(
            conds=[text_prompt],
            m_lens=m_lens,
            timesteps=args.timesteps,
            cond_scale=args.cfg_scale,
            temperature=args.temperature,
            hard_pseudo_reorder=args.hard_pseudo_reorder,
            raw_style_latents=raw_style_latents,
            style_weight_schedule=w_schedule,
            cfg_mode=cfg_mode,
            cfg_text=cfg_text,
            cfg_style=cfg_style,
        )
    latent_time = time.perf_counter() - t0
    if not hasattr(generate_latent, '_last_time'):
        generate_latent._last_time = 0.0
    generate_latent._last_time = latent_time
    return pred_latents


def apply_trajectory_refinement(motion, refine_net):
    """Replace the root-trajectory features (dims 0:4) using the refinement net.

    motion: [B, T, 67] normalized decoder output (matches train_refinement.py).
    Returns motion of the same shape.
    """
    motion_cf = motion.permute(0, 2, 1)                # [B, 67, T]
    body_local = motion_cf[:, 4:, :]                    # [B, 63, T]
    velocities = compute_velocities(body_local)         # [B, 63, T]
    contacts = detect_foot_contact(motion_cf)           # [B, 4, T]
    input_feats = torch.cat([body_local, velocities, contacts], dim=1)  # [B, 130, T]
    pred_root = refine_net(input_feats)                 # [B, 4, T]
    motion_cf = motion_cf.clone()
    motion_cf[:, :4, :] = pred_root
    return motion_cf.permute(0, 2, 1)                   # [B, T, 67]


def decode_and_recover(latents, decoder, mean, std, m_length, joints_num=22, refine_net=None):
    """Decode latents and recover joint positions.

    If refine_net is provided, its predicted root trajectory replaces
    the decoded root features before unnormalization.
    """
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_motion = decoder.decode(latents)
        if refine_net is not None:
            pred_motion = apply_trajectory_refinement(pred_motion, refine_net)
        pred_motion_np = pred_motion.cpu().numpy() * std + mean
        motion_features = pred_motion_np[0, :m_length]
        joint_data = recover_from_ric(torch.from_numpy(motion_features).float(), joints_num).numpy()
    decode_and_recover._last_time = time.perf_counter() - t0
    return joint_data, motion_features


# def generate_motion(model, dae, ae, text_prompt, m_length, device, mean, std, raw_style_latents=None, w_schedule=None, args=None, use_dae=True, return_latents=False):
def generate_motion(model, dae, text_prompt, m_length, device, mean, std, raw_style_latents=None, w_schedule=None, args=None, use_dae=True, return_latents=False):

    """Generate a single motion sample with specified decoder."""
    t_total = time.perf_counter()
    pred_latents = generate_latent(model, text_prompt, m_length, device,
                                   raw_style_latents, w_schedule, args)
    t_latent = getattr(generate_latent, '_last_time', 0.0)

    refine_net = getattr(args, 'refine_net', None) if args is not None else None
    joint_data, motion_features = decode_and_recover(
        pred_latents, dae, mean, std, m_length, refine_net=refine_net
    )
    t_decode = getattr(decode_and_recover, '_last_time', 0.0)
    t_total = time.perf_counter() - t_total

    print(f"  [timing] latent={t_latent:.2f}s  decode={t_decode:.2f}s  total={t_total:.2f}s")

    if return_latents:
        return joint_data, motion_features, pred_latents
    return joint_data, motion_features


#################################################################################
#                     EXPERIMENT 8.1: Reverse Decoder Test                       #
#################################################################################

def run_exp_reverse_decoder(mardm, dae, ae, prompts, lengths, device, 
                            hml3d_mean, hml3d_std, style_mean, style_std,
                            kinematic_chain, result_dir, args):
    """
    Experiment 8.1: Generate WITHOUT style, decode with both DAE and AE.
    Check if DAE adds spurious style to neutral latents.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8.1: REVERSE DECODER TEST")
    print("=" * 70)
    print("Goal: Check if DAE adds spurious style to neutral latents\n")
    
    exp_dir = pjoin(result_dir, "exp_8.1_reverse_decoder")
    os.makedirs(exp_dir, exist_ok=True)
    
    results = []
    
    for idx, (prompt, m_length) in enumerate(zip(prompts, lengths)):
        print(f"[{idx+1}/{len(prompts)}] '{prompt[:50]}...'")
        
        # Generate latent WITHOUT style
        latents = generate_latent(mardm, prompt, m_length, device,
                                  raw_style_latents=None, w_schedule=None, args=args)
        
        # Decode with BOTH decoders
        refine_net = getattr(args, 'refine_net', None)
        joints_dae, _ = decode_and_recover(latents, dae, style_mean, style_std, m_length, refine_net=refine_net)
        joints_ae, _ = decode_and_recover(latents, ae, hml3d_mean, hml3d_std, m_length, refine_net=refine_net)
        
        # Compute distance between outputs
        distance = compute_motion_distance(joints_dae, joints_ae)
        results.append({
            'prompt': prompt,
            'length': m_length,
            'dae_ae_distance': float(distance)
        })
        print(f"  DAE vs AE distance: {distance:.4f}")
        
        # Save visualizations
        safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:30]
        
        plot_3d_motion_gif(
            pjoin(exp_dir, f'{idx}_{safe_prompt}_DAE.gif'),
            kinematic_chain, joints_dae,
            title="Neutral Latent → DAE",
            fps=20, text_prompt=prompt[:60], style_label="None (DAE)"
        )
        
        plot_3d_motion_gif(
            pjoin(exp_dir, f'{idx}_{safe_prompt}_AE.gif'),
            kinematic_chain, joints_ae,
            title="Neutral Latent → AE",
            fps=20, text_prompt=prompt[:60], style_label="None (AE)"
        )
    
    # Save results
    with open(pjoin(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    avg_distance = np.mean([r['dae_ae_distance'] for r in results])
    print(f"\n✓ Average DAE vs AE distance: {avg_distance:.4f}")
    print(f"  (Lower = decoders behave similarly on neutral latents)")
    print(f"  Results saved to: {exp_dir}")
    
    return results


#################################################################################
#                     EXPERIMENT 8.2: Decoder Interpolation                      #
#################################################################################

def run_exp_decoder_interp(mardm, dae, ae, prompts, lengths, device,
                           hml3d_mean, hml3d_std, style_mean, style_std,
                           raw_style_latents, w_schedule,
                           kinematic_chain, result_dir, args):
    """
    Experiment 8.2: Blend decoder outputs for controllable style strength.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8.2: DECODER INTERPOLATION FOR STYLE STRENGTH")
    print("=" * 70)
    print("Goal: Create controllable style intensity by blending decoder outputs\n")
    
    exp_dir = pjoin(result_dir, "exp_8.2_decoder_interp")
    os.makedirs(exp_dir, exist_ok=True)
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for idx, (prompt, m_length) in enumerate(zip(prompts, lengths)):
        print(f"[{idx+1}/{len(prompts)}] '{prompt[:50]}...'")
        
        # Generate styled latent
        latents = generate_latent(mardm, prompt, m_length, device,
                                  raw_style_latents=raw_style_latents,
                                  w_schedule=w_schedule, args=args)
        
        # Decode with both decoders
        refine_net = getattr(args, 'refine_net', None)
        joints_styled, _ = decode_and_recover(latents, dae, style_mean, style_std, m_length, refine_net=refine_net)
        joints_neutral, _ = decode_and_recover(latents, ae, hml3d_mean, hml3d_std, m_length, refine_net=refine_net)
        
        # ==========================================
        # NEW FIX: Align the sequence lengths
        # ==========================================
        min_len = min(joints_styled.shape[0], joints_neutral.shape[0])
        joints_styled = joints_styled[:min_len]
        joints_neutral = joints_neutral[:min_len]
        
        safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:30]
        sample_dir = pjoin(exp_dir, f'sample_{idx}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Generate interpolated outputs
        for alpha in alphas:
            blended_joints = alpha * joints_styled + (1 - alpha) * joints_neutral
            
            plot_3d_motion_gif(
                pjoin(sample_dir, f'{safe_prompt}_alpha_{alpha:.2f}.gif'),
                kinematic_chain, blended_joints,
                title=f"Style Strength: {alpha:.0%}",
                fps=20, text_prompt=prompt[:60], style_label=f"α={alpha:.2f}"
            )
            print(f"  Generated α={alpha:.2f}")
        
        np.save(pjoin(sample_dir, 'joints_styled.npy'), joints_styled)
        np.save(pjoin(sample_dir, 'joints_neutral.npy'), joints_neutral)
    
    print(f"\n✓ Results saved to: {exp_dir}")


#################################################################################
#                     EXPERIMENT 8.3: Content Preservation                       #
#################################################################################

def run_exp_content_preserve(mardm, dae, ae, device,
                             hml3d_mean, hml3d_std, style_mean, style_std,
                             raw_style_latents, w_schedule,
                             kinematic_chain, result_dir, args):
    """
    Experiment 8.3: Same style + different texts → verify content differs.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8.3: CONTENT PRESERVATION VERIFICATION")
    print("=" * 70)
    print("Goal: Different texts + same style should produce different content\n")
    
    exp_dir = pjoin(result_dir, "exp_8.3_content_preserve")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Test prompts with clearly different actions
    test_prompts = [
        ("A person walks forward slowly", 120),
        ("A person raises both arms above their head", 80),
        ("A person kicks with their right leg", 60),
        ("A person sits down on a chair", 100),
        ("A person jumps in place", 60),
        ("A person waves their hand", 80),
    ]
    
    all_joints_styled = []
    all_joints_content = []
    
    for idx, (prompt, m_length) in enumerate(test_prompts):
        print(f"[{idx+1}/{len(test_prompts)}] '{prompt}'")
        
        # Generate with style
        latents = generate_latent(mardm, prompt, m_length, device,
                                  raw_style_latents=raw_style_latents,
                                  w_schedule=w_schedule, args=args)
        
        # Decode with DAE (styled) and AE (content only)
        refine_net = getattr(args, 'refine_net', None)
        joints_styled, features_styled = decode_and_recover(latents, dae, style_mean, style_std, m_length, refine_net=refine_net)
        # joints_content, _ = decode_and_recover(latents, ae, hml3d_mean, hml3d_std, m_length)
        
        all_joints_styled.append(joints_styled)
        # all_joints_content.append(joints_content)
        
        safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:30]

        np.save(pjoin(exp_dir, f'{safe_prompt}_joints.npy'), joints_styled)
        np.save(pjoin(exp_dir, f'{safe_prompt}_features.npy'), features_styled)
        
        plot_3d_motion_gif(
            pjoin(exp_dir, f'{idx}_{safe_prompt}_styled.gif'),
            kinematic_chain, joints_styled,
            title="Styled Output",
            fps=20, text_prompt=prompt[:60], style_label="Styled"
        )
        
        # plot_3d_motion_gif(
        #     pjoin(exp_dir, f'{idx}_{safe_prompt}_content.gif'),
        #     kinematic_chain, joints_content,
        #     title="Content Only (AE)",
        #     fps=20, text_prompt=prompt[:60], style_label="Content"
        # )
    
    # Compute pairwise distances
    # print("\n--- Content Pairwise Distances (should be HIGH) ---")
    # distance_matrix = []
    # for i in range(len(test_prompts)):
    #     row = []
    #     for j in range(len(test_prompts)):
    #         if i != j:
    #             dist = float(compute_motion_distance(all_joints_content[i], all_joints_content[j]))
    #             row.append(dist)
    #             if i < j:
    #                 print(f"  '{test_prompts[i][0][:25]}' vs '{test_prompts[j][0][:25]}': {dist:.4f}")
    #         else:
    #             row.append(0.0)
    #     distance_matrix.append(row)
    
    # avg_distance = np.mean([d for row in distance_matrix for d in row if d > 0])
    
    # results = {
    #     'prompts': [p[0] for p in test_prompts],
    #     'distance_matrix': distance_matrix,
    #     'avg_content_distance': float(avg_distance)
    # }
    
    # with open(pjoin(exp_dir, 'results.json'), 'w') as f:
    #     json.dump(results, f, indent=2)
    
    # print(f"\n✓ Average content distance: {avg_distance:.4f}")
    print(f"  Results saved to: {exp_dir}")


#################################################################################
#                     EXPERIMENT 8.5: Per-Style Breakdown                        #
#################################################################################

def run_exp_per_style(mardm, dae, ae, vmodel, processor, prompts, lengths, device,
                      hml3d_mean, hml3d_std, style_mean, style_std,
                      w_schedule, kinematic_chain, result_dir, args):
    """
    Experiment 8.5: Generate for each style video and compare.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8.5: PER-STYLE PERFORMANCE BREAKDOWN")
    print("=" * 70)
    
    exp_dir = pjoin(result_dir, "exp_8.5_per_style")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Find style videos
    style_video_dir = pjoin(args.dataset_dir, '100STYLE-SMPL', 'videos')
    if args.style_videos:
        style_videos = args.style_videos
    else:
        all_videos = sorted(glob(pjoin(style_video_dir, '*.mp4')))[:args.num_styles]
        style_videos = all_videos
    
    print(f"Testing {len(style_videos)} styles\n")
    
    results = {}
    
    for style_path in tqdm(style_videos, desc="Styles"):
        style_name = Path(style_path).stem
        
        style_dir = pjoin(exp_dir, style_name)
        os.makedirs(style_dir, exist_ok=True)
        
        try:
            raw_style = extract_style_latents(style_path, dae, vmodel, processor, device)
        except Exception as e:
            print(f"  Error extracting style from {style_path}: {e}")
            continue
        
        for idx, (prompt, m_length) in enumerate(zip(prompts, lengths)):
            # joints, _ = generate_motion(
            #     mardm, dae, ae, prompt, m_length, device, style_mean, style_std,
            #     raw_style_latents=raw_style, w_schedule=w_schedule, args=args,
            #     use_dae=True
            # )

            joints, features = generate_motion(
                mardm, dae, prompt, m_length, device, style_mean, style_std,
                raw_style_latents=raw_style, w_schedule=w_schedule, args=args,
                use_dae=True
            )
            
            safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:30]
            np.save(pjoin(style_dir, f'{safe_prompt}_joints.npy'), joints)
            np.save(pjoin(style_dir, f'{safe_prompt}_features.npy'), features)
            
            plot_3d_motion_gif(
                pjoin(style_dir, f'{idx}_{safe_prompt}.gif'),
                kinematic_chain, joints,
                title=f"Style: {style_name}",
                fps=20, text_prompt=prompt[:40], style_label=style_name
            )
        
        results[style_name] = {'num_samples': len(prompts)}
    
    with open(pjoin(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Generated samples for {len(results)} styles")
    print(f"  Results saved to: {exp_dir}")


#################################################################################
#                     EXPERIMENT 8.6: Schedule Ablation                          #
#################################################################################

def run_exp_schedule_ablation(mardm, dae, ae, prompts, lengths, device,
                               hml3d_mean, hml3d_std, style_mean, style_std,
                               raw_style_latents, kinematic_chain, result_dir, args):
    """
    Experiment 8.6: Test different weight schedules.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8.6: WEIGHT SCHEDULE ABLATION")
    print("=" * 70)
    
    exp_dir = pjoin(result_dir, "exp_8.6_schedule_ablation")
    os.makedirs(exp_dir, exist_ok=True)
    
    schedule_types = [
        'uniform',           # [1.0, 1.0, ..., 1.0]
        'linear',            # [0.0, ..., 1.0]
        'linear_0.2_1',      # [0.2, ..., 1.0]
        'style_blocks_only', # [0.0]*16 + [1.0]*8
        'two_phase',         # [0.2→0.6]*16 + [1.0]*8
        'cosine',            # sin curve
        'none',              # No style
    ]
    
    results = {}
    
    for schedule_type in schedule_types:
        print(f"\n--- Schedule: {schedule_type} ---")
        
        schedule = get_style_weight_schedule(mardm, schedule_type=schedule_type)
        schedule_dir = pjoin(exp_dir, schedule_type)
        os.makedirs(schedule_dir, exist_ok=True)
        
        if schedule:
            print(f"  Values: [{schedule[0]:.2f}, ..., {schedule[-1]:.2f}]")
        
        for idx, (prompt, m_length) in enumerate(zip(prompts, lengths)):
            style_input = raw_style_latents if schedule_type != 'none' else None
            
            # joints, _ = generate_motion(
            #     mardm, dae, ae, prompt, m_length, device, style_mean, style_std,
            #     raw_style_latents=style_input, w_schedule=schedule, args=args,
            #     use_dae=True
            # )

                        
            joints, _ = generate_motion(
                mardm, dae, prompt, m_length, device, style_mean, style_std,
                raw_style_latents=style_input, w_schedule=schedule, args=args,
                use_dae=True
            )
            
            safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:20]
            
            plot_3d_motion_gif(
                pjoin(schedule_dir, f'{idx}_{safe_prompt}.gif'),
                kinematic_chain, joints,
                title=f"Schedule: {schedule_type}",
                fps=20, text_prompt=prompt[:40], style_label=schedule_type
            )
        
        results[schedule_type] = {'schedule_values': schedule}
    
    with open(pjoin(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Tested {len(schedule_types)} schedule types")
    print(f"  Results saved to: {exp_dir}")


#################################################################################
#                     EXPERIMENT 8.7: CFG Scale Sweep                            #
#################################################################################

def run_exp_cfg_sweep(mardm, dae, ae, prompts, lengths, device,
                      hml3d_mean, hml3d_std, style_mean, style_std,
                      raw_style_latents, w_schedule,
                      kinematic_chain, result_dir, args):
    """
    Experiment 8.7: Sweep CFG scales.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8.7: CFG SCALE SWEEP")
    print("=" * 70)
    
    exp_dir = pjoin(result_dir, "exp_8.7_cfg_sweep")
    os.makedirs(exp_dir, exist_ok=True)
    
    cfg_scales = [1.0, 2.0, 3.0, 4.5, 6.0, 7.5, 10.0]
    
    prompt, m_length = prompts[0], lengths[0]
    print(f"Prompt: '{prompt[:50]}...'\n")
    
    original_cfg = args.cfg_scale
    results = {}
    
    for cfg in cfg_scales:
        print(f"--- CFG Scale: {cfg} ---")
        
        cfg_dir = pjoin(exp_dir, f'cfg_{cfg:.1f}')
        os.makedirs(cfg_dir, exist_ok=True)
        
        args.cfg_scale = cfg
        
        # # Styled
        # joints_styled, _ = generate_motion(
        #     mardm, dae, ae, prompt, m_length, device, style_mean, style_std,
        #     raw_style_latents=raw_style_latents, w_schedule=w_schedule, args=args,
        #     use_dae=True
        # )
        
        # # Unstyled
        # joints_unstyled, _ = generate_motion(
        #     mardm, dae, ae, prompt, m_length, device, hml3d_mean, hml3d_std,
        #     raw_style_latents=None, w_schedule=None, args=args,
        #     use_dae=False
        # )

        # Styled
        joints_styled, _ = generate_motion(
            mardm, dae, prompt, m_length, device, style_mean, style_std,
            raw_style_latents=raw_style_latents, w_schedule=w_schedule, args=args,
            use_dae=True
        )
        
        # Unstyled
        joints_unstyled, _ = generate_motion(
            mardm, dae, prompt, m_length, device, hml3d_mean, hml3d_std,
            raw_style_latents=None, w_schedule=None, args=args,
            use_dae=False
        )
        
        safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:20]
        
        plot_3d_motion_gif(
            pjoin(cfg_dir, f'{safe_prompt}_styled.gif'),
            kinematic_chain, joints_styled,
            title=f"CFG={cfg} (Styled)",
            fps=20, text_prompt=prompt[:40], style_label=f"cfg={cfg}"
        )
        
        plot_3d_motion_gif(
            pjoin(cfg_dir, f'{safe_prompt}_unstyled.gif'),
            kinematic_chain, joints_unstyled,
            title=f"CFG={cfg} (Unstyled)",
            fps=20, text_prompt=prompt[:40], style_label="None"
        )
        
        results[f'cfg_{cfg}'] = {'cfg_scale': cfg}
    
    args.cfg_scale = original_cfg
    
    with open(pjoin(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Tested {len(cfg_scales)} CFG scales")
    print(f"  Results saved to: {exp_dir}")


#################################################################################
#                EXPERIMENT 8.8: 3-way CFG (text x style) Grid Search            #
#################################################################################

def run_exp_cfg_grid(mardm, dae, prompts, lengths, device,
                     style_mean, style_std,
                     raw_style_latents, w_schedule,
                     kinematic_chain, result_dir, args):
    """
    Experiment 8.8: Grid-search over (cfg_text, cfg_style) for both 3-way CFG modes.

    For each prompt, iterates (mode) x (cfg_text) x (cfg_style) and generates the
    styled "both" output via DAE. GIFs are organized as:
        exp_8.8_cfg_grid/<prompt>/<mode>/t{cfg_text}_s{cfg_style}.gif
    to make side-by-side visual comparison easy.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8.8: 3-WAY CFG GRID SEARCH")
    print("=" * 70)

    exp_dir = pjoin(result_dir, "exp_8.8_cfg_grid")
    os.makedirs(exp_dir, exist_ok=True)

    # Grid can be overridden via CLI; otherwise use sensible defaults.
    cfg_text_grid = getattr(args, 'grid_cfg_text', None) or [2.5, 4.0, 5.5, 7.0]
    cfg_style_grid = getattr(args, 'grid_cfg_style', None) or [1.0, 1.5, 2.0, 3.0]
    modes = getattr(args, 'grid_cfg_modes', None) or ['3way_additive', '3way_style_first']

    print(f"cfg_text grid:  {cfg_text_grid}")
    print(f"cfg_style grid: {cfg_style_grid}")
    print(f"modes:          {modes}")
    print(f"Total generations per prompt: "
          f"{len(modes) * len(cfg_text_grid) * len(cfg_style_grid)}\n")

    # Save original args we will mutate, so we can restore afterwards.
    original_cfg_mode = getattr(args, 'cfg_mode', '2way')
    original_cfg_text = getattr(args, 'cfg_text', None)
    original_cfg_style = getattr(args, 'cfg_style', None)

    results = {}

    for p_idx, (prompt, m_length) in enumerate(zip(prompts, lengths)):
        safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:30]
        prompt_dir = pjoin(exp_dir, f'{p_idx:02d}_{safe_prompt}')
        os.makedirs(prompt_dir, exist_ok=True)

        print(f"\n[{p_idx+1}/{len(prompts)}] Prompt: '{prompt[:60]}' ({m_length} frames)")

        results[f'{p_idx:02d}_{safe_prompt}'] = {'prompt': prompt, 'length': m_length, 'runs': []}

        for mode in modes:
            mode_dir = pjoin(prompt_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            args.cfg_mode = mode

            print(f"  --- Mode: {mode} ---")

            for ct in cfg_text_grid:
                for cs in cfg_style_grid:
                    args.cfg_text = ct
                    args.cfg_style = cs

                    joints, _ = generate_motion(
                        mardm, dae, prompt, m_length, device, style_mean, style_std,
                        raw_style_latents=raw_style_latents, w_schedule=w_schedule,
                        args=args, use_dae=True
                    )

                    gif_name = f't{ct:.1f}_s{cs:.1f}.gif'
                    plot_3d_motion_gif(
                        pjoin(mode_dir, gif_name),
                        kinematic_chain, joints,
                        title=f"{mode} | cfg_t={ct} cfg_s={cs}",
                        fps=20, text_prompt=prompt[:40],
                        style_label=f"t={ct}/s={cs}"
                    )

                    print(f"    cfg_text={ct:.1f}  cfg_style={cs:.1f}  -> {gif_name}")
                    results[f'{p_idx:02d}_{safe_prompt}']['runs'].append({
                        'mode': mode, 'cfg_text': ct, 'cfg_style': cs,
                        'gif': pjoin(mode, gif_name)
                    })

    # Restore args.
    args.cfg_mode = original_cfg_mode
    args.cfg_text = original_cfg_text
    args.cfg_style = original_cfg_style

    with open(pjoin(exp_dir, 'results.json'), 'w') as f:
        json.dump({
            'cfg_text_grid': cfg_text_grid,
            'cfg_style_grid': cfg_style_grid,
            'modes': modes,
            'per_prompt': results,
        }, f, indent=2)

    total = sum(len(v['runs']) for v in results.values())
    print(f"\n✓ CFG grid search complete — {total} generations")
    print(f"  Results saved to: {exp_dir}")


#################################################################################
#                                    Main                                        #
#################################################################################

def main(args):
    set_seed(args.seed)
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    #################################################################################
    #                                 Data Setup                                    #
    #################################################################################
    dim_pose = 67
    kinematic_chain = t2m_kinematic_chain
    
    style_data_root = f'{args.dataset_dir}/100STYLE-SMPL/'
    hml3d_data_root = f'{args.dataset_dir}/HumanML3D/'
    
    style_mean = np.load(pjoin(style_data_root, 'Mean.npy'))[:dim_pose]
    style_std = np.load(pjoin(style_data_root, 'Std.npy'))[:dim_pose]
    hml3d_mean = np.load(pjoin(hml3d_data_root, 'Mean.npy'))[:dim_pose]
    hml3d_std = np.load(pjoin(hml3d_data_root, 'Std.npy'))[:dim_pose]
    
    #################################################################################
    #                                Model Setup                                    #
    #################################################################################
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)
    
    num_classes = len(args.styles) if args.styles else 100
    
    print("\nLoading AE for HumanML3D...")
    ae = AE_models["AE_Model"](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, 't2m', 'AE', 'model', 'latest.tar'), map_location=device)
    ae.load_state_dict(ckpt['ae'])
    ae.to(device).eval()
    
    print("\nLoading DAE for 100STYLES...")
    dae = DAE_models[args.ae_model](window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose)
    dae_ckpt_path = args.dae_ckpt or pjoin(args.checkpoints_dir, '100styles', args.ae_name, 'final_finetune_diffmlp_decoder_fixed_hybrid_ema_fix.tar')
    # dae_ckpt_path = pjoin(args.checkpoints_dir, '100styles', args.ae_name, 'epoch_119_detach_nostyle_disc.tar')
    dae_ckpt = torch.load(dae_ckpt_path, map_location=device, weights_only=False)
    dae.load_state_dict(dae_ckpt['ae'])
    dae.to(device).eval()
    
    print(f"\nLoading video encoder ({args.video_encoder})...")
    MODEL_CONFIG = {
        'vivit': {"name": "google/vivit-b-16x2-kinetics400", "processor": "google/vivit-b-16x2-kinetics400"},
        'timesformer': {"name": "facebook/timesformer-base-finetuned-k400", "processor": "MCG-NJU/videomae-base"},
    }
    
    if args.video_encoder == 'vivit':
        processor = VivitImageProcessor.from_pretrained(MODEL_CONFIG['vivit']['processor'])
        vmodel = VivitModel.from_pretrained(MODEL_CONFIG['vivit']['name']).to(device)
    else:
        processor = AutoProcessor.from_pretrained(MODEL_CONFIG['timesformer']['processor'])
        vmodel = TimesformerModel.from_pretrained(MODEL_CONFIG['timesformer']['name']).to(device)
    
    vmodel.eval()
    for p in vmodel.parameters():
        p.requires_grad = False
    
    print(f"\nLoading MARDM ({args.model})...")
    mardm = MARDM_models[args.model](
        ae_dim=dae.output_emb_width,
        cond_mode='text',
        style_routing=args.style_routing,
        style_dim=512
    )
    
    mardm_ckpt_path = pjoin(args.checkpoints_dir, 't2m', args.model, 'model', args.checkpoint_name)
    ckpt_stem = Path(mardm_ckpt_path).name.removesuffix('.tar')
    load_checkpoint(mardm, mardm_ckpt_path, device, key=args.checkpoint_key)
    mardm.to(device).eval()
    
    # Optional trajectory refinement net -------------------------------------
    args.refine_net = None
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
        args.refine_net = refine_net
        print(f"  Refinement net loaded (epoch {refine_ckpt.get('epoch', '?')}).")

    print("\nLoading length estimator...")
    length_estimator = LengthEstimator(512, 50)
    le_path = pjoin(args.checkpoints_dir, 't2m', 'length_estimator', 'model', 'finest.tar')
    le_ckpt = torch.load(le_path, map_location=device, weights_only=False)
    length_estimator.load_state_dict(le_ckpt['estimator'])
    length_estimator.to(device).eval()
    
    w_schedule = get_style_weight_schedule(mardm, args)
    
    #################################################################################
    #                              Parse Input                                      #
    #################################################################################
    print("\n" + "=" * 60)
    print("PARSING INPUT")
    print("=" * 60)
    
    prompts = []
    lengths = []
    
    if args.text_prompt:
        prompts.append(args.text_prompt)
        lengths.append(args.motion_length if args.motion_length > 0 else 0)
    elif args.text_path:
        with open(args.text_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('#')
                prompts.append(parts[0].strip())
                if len(parts) > 1 and parts[1].strip().isdigit():
                    lengths.append(int(parts[1].strip()))
                else:
                    lengths.append(0)
    else:
        prompts = ["A person walks forward"]
        lengths = [120]
    
    if any(l == 0 for l in lengths):
        print("Estimating motion lengths...")
        with torch.no_grad():
            text_emb = mardm.encode_text(prompts)
            pred_dist = length_estimator(text_emb)
            probs = F.softmax(pred_dist, dim=-1)
            token_lens = Categorical(probs).sample()
            
            for i, l in enumerate(lengths):
                if l == 0:
                    lengths[i] = token_lens[i].item() * 4
    
    print(f"\nPrompts to generate: {len(prompts)}")
    for i, (p, l) in enumerate(zip(prompts, lengths)):
        print(f"  {i+1}. '{p[:60]}' ({l} frames)")
    
    #################################################################################
    #                         Check for Experiment Mode                             #
    #################################################################################
    exp_flags = [
        args.exp_reverse_decoder, args.exp_decoder_interp,
        args.exp_content_preserve, args.exp_per_style,
        args.exp_schedule_ablation, args.exp_cfg_sweep,
        args.exp_cfg_grid,
    ]
    is_experiment_mode = any(exp_flags)
    
    #################################################################################
    #                         Extract Style (if needed)                             #
    #################################################################################
    raw_style_latents = None
    style_label = "base"
    
    if args.style_video:
        print(f"\nExtracting style from: {args.style_video}")
        raw_style_latents = extract_style_latents(args.style_video, dae, vmodel, processor, device, dim_pose)
        style_label = os.path.basename(args.style_video).split('.')[0]
    
    #################################################################################
    #                              Create Output Dir                                #
    #################################################################################
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        result_dir = pjoin(args.output_dir, ckpt_stem, f'{style_label}_{timestamp}')
    else:
        result_dir = pjoin('./generation', f'{args.model}_{style_label}_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    print(f"\nOutput directory: {result_dir}")

    # Record everything needed to replay this run exactly (prompts/lengths after length estimation).
    run_config = {
        'timestamp': timestamp,
        'command': ' '.join(sys.argv),
        'args': {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))},
        'resolved': {
            'mardm_ckpt': os.path.abspath(mardm_ckpt_path),
            'dae_ckpt': os.path.abspath(dae_ckpt_path),
            'style_video': os.path.abspath(args.style_video) if args.style_video else None,
            'prompts': prompts,
            'lengths': [int(l) for l in lengths],
            'style_weight_schedule': [float(w) for w in w_schedule],
        },
        'versions': {
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu',
        },
    }
    with open(pjoin(result_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)

    #################################################################################
    #                              Run Experiments                                  #
    #################################################################################
    if is_experiment_mode:
        print("\n" + "=" * 60)
        print("RUNNING EXPERIMENTS")
        print("=" * 60)
        
        if args.exp_reverse_decoder:
            run_exp_reverse_decoder(
                mardm, dae, ae, prompts, lengths, device,
                hml3d_mean, hml3d_std, style_mean, style_std,
                kinematic_chain, result_dir, args
            )
        
        if args.exp_decoder_interp:
            if raw_style_latents is None:
                print("\n⚠️  --exp_decoder_interp requires --style_video")
            else:
                run_exp_decoder_interp(
                    mardm, dae, ae, prompts, lengths, device,
                    hml3d_mean, hml3d_std, style_mean, style_std,
                    raw_style_latents, w_schedule,
                    kinematic_chain, result_dir, args
                )
        
        if args.exp_content_preserve:
            if raw_style_latents is None:
                print("\n⚠️  --exp_content_preserve requires --style_video")
            else:
                run_exp_content_preserve(
                    mardm, dae, ae, device,
                    hml3d_mean, hml3d_std, style_mean, style_std,
                    raw_style_latents, w_schedule,
                    kinematic_chain, result_dir, args
                )
        
        if args.exp_per_style:
            run_exp_per_style(
                mardm, dae, ae, vmodel, processor, prompts, lengths, device,
                hml3d_mean, hml3d_std, style_mean, style_std,
                w_schedule, kinematic_chain, result_dir, args
            )
        
        if args.exp_schedule_ablation:
            if raw_style_latents is None:
                print("\n⚠️  --exp_schedule_ablation requires --style_video")
            else:
                run_exp_schedule_ablation(
                    mardm, dae, ae, prompts, lengths, device,
                    hml3d_mean, hml3d_std, style_mean, style_std,
                    raw_style_latents, kinematic_chain, result_dir, args
                )
        
        if args.exp_cfg_sweep:
            if raw_style_latents is None:
                print("\n⚠️  --exp_cfg_sweep requires --style_video")
            else:
                run_exp_cfg_sweep(
                    mardm, dae, ae, prompts, lengths, device,
                    hml3d_mean, hml3d_std, style_mean, style_std,
                    raw_style_latents, w_schedule,
                    kinematic_chain, result_dir, args
                )

        if args.exp_cfg_grid:
            if raw_style_latents is None:
                print("\n⚠️  --exp_cfg_grid requires --style_video")
            else:
                run_exp_cfg_grid(
                    mardm, dae, prompts, lengths, device,
                    style_mean, style_std,
                    raw_style_latents, w_schedule,
                    kinematic_chain, result_dir, args
                )
        
        print("\n" + "=" * 60)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {result_dir}")
        return
    
    #################################################################################
    #                         Standard Generation Modes                             #
    #################################################################################
    print("\n" + "=" * 60)
    print("GENERATING SAMPLES")
    print("=" * 60)
    _session_start = time.perf_counter()
    _sample_count = 0
    
    # Determine mode
    if args.interpolate:
        if not args.style_video or not args.style_video_b:
            raise ValueError("--interpolate requires BOTH --style_video and --style_video_b.")
        mode = "STYLE INTERPOLATION"
        
        style_a = raw_style_latents
        style_b = extract_style_latents(args.style_video_b, dae, vmodel, processor, device, dim_pose)
        raw_style_latents = blend_style_latents(style_a, style_b, args.blend_weight)
        style_label = f"interpolate_w{args.blend_weight:.2f}"
        active_w_schedule = w_schedule
        mean, std = style_mean, style_std

    elif args.generate_quad:
        if not args.style_video:
            raise ValueError("--generate_quad requires --style_video.")
        mode = "QUAD VARIATIONS"
        style_label = f"quad_{os.path.basename(args.style_video).split('.')[0]}"
        active_w_schedule = w_schedule
        mean, std = None, None

    elif args.style_video:
        mode = "SINGLE STYLE"
        active_w_schedule = w_schedule
        mean, std = style_mean, style_std

    else:
        mode = "BASE (NO STYLE)"
        style_label = "base"
        raw_style_latents = None
        active_w_schedule = None
        mean, std = hml3d_mean, hml3d_std
    
    print(f"\nGeneration mode: {mode}")
    
    for repeat in range(args.repeat_times):
        if args.repeat_times > 1:
            print(f"\n--- Repeat {repeat + 1}/{args.repeat_times} ---")

        for prompt_idx, (prompt, m_length) in enumerate(zip(prompts, lengths)):
            print(f"\nGenerating: '{prompt[:50]}...' (length={m_length})")

            sample_dir = pjoin(result_dir, f'sample_{prompt_idx}')
            os.makedirs(sample_dir, exist_ok=True)
            safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:30]

            if args.generate_quad:
                variations = [
                    {"name": "uncond",     "text": "",     "style": None,               "mean": hml3d_mean, "std": hml3d_std},
                    {"name": "text_only",  "text": prompt, "style": None,               "mean": hml3d_mean, "std": hml3d_std},
                    {"name": "style_only", "text": "",     "style": raw_style_latents,  "mean": style_mean, "std": style_std},
                    {"name": "both",       "text": prompt, "style": raw_style_latents,  "mean": style_mean, "std": style_std}
                ]

                for var in variations:
                    print(f"  -> Variation: {var['name']}")

                    # joint_data, motion_features = generate_motion(
                    #     mardm, dae, ae, var["text"], m_length, device, var["mean"], var["std"],
                    #     raw_style_latents=var["style"],
                    #     w_schedule=active_w_schedule if var["style"] is not None else None,
                    #     args=args
                    # )

                    joint_data, motion_features = generate_motion(
                        mardm, dae, var["text"], m_length, device, var["mean"], var["std"],
                        raw_style_latents=var["style"],
                        w_schedule=active_w_schedule if var["style"] is not None else None,
                        args=args
                    )
                    _sample_count += 1

                    var_label = f"{style_label}_{var['name']}"
                    gif_path = pjoin(sample_dir, f'{safe_prompt}_r{repeat}_{var_label}.gif')

                    display_prompt = prompt if var["text"] else "(Unconditional)"
                    display_style = style_label if var["style"] is not None else "None"

                    plot_3d_motion_gif(
                        gif_path, kinematic_chain, joint_data,
                        title=f"Var: {var['name']}",
                        fps=20, text_prompt=display_prompt[:60], style_label=display_style
                    )

                    np.save(pjoin(sample_dir, f'{safe_prompt}_r{repeat}_{var_label}_joints.npy'), joint_data)
                    np.save(pjoin(sample_dir, f'{safe_prompt}_r{repeat}_{var_label}_features.npy'), motion_features)
                    print(f"    Saved: {gif_path}")

            else:
                # joint_data, motion_features = generate_motion(
                #     mardm, dae, ae, prompt, m_length, device, mean, std,
                #     raw_style_latents=raw_style_latents,
                #     w_schedule=active_w_schedule,
                #     args=args
                # )

                joint_data, motion_features = generate_motion(
                    mardm, dae, prompt, m_length, device, mean, std,
                    raw_style_latents=raw_style_latents,
                    w_schedule=active_w_schedule,
                    args=args
                )
                _sample_count += 1

                gif_path = pjoin(sample_dir, f'{safe_prompt}_r{repeat}_{style_label}.gif')

                plot_3d_motion_gif(
                    gif_path, kinematic_chain, joint_data,
                    title=f"Style: {style_label}",
                    fps=20, text_prompt=prompt[:60], style_label=style_label
                )

                np.save(pjoin(sample_dir, f'{safe_prompt}_r{repeat}_joints.npy'), joint_data)
                np.save(pjoin(sample_dir, f'{safe_prompt}_r{repeat}_features.npy'), motion_features)
                print(f"  Saved: {gif_path}")

    _session_elapsed = time.perf_counter() - _session_start
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Samples generated : {_sample_count}")
    print(f"Total session time: {_session_elapsed:.2f}s  ({_session_elapsed / max(_sample_count, 1):.2f}s/sample)")
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from MARDM with Style Conditioning")
    
    # Model arguments
    parser.add_argument('--model', type=str, default='MARDM-DDPM-XL', choices=['MARDM-DDPM-XL', 'MARDM-SiT-XL'])
    parser.add_argument('--checkpoint_name', type=str, default='final_diffmlp_diff_v4_cfg_cross_batch_hybrid_ema_fix_best_fid.tar')
    parser.add_argument('--checkpoint_key', type=str, default='ema_mardm')
    
    # AE/DAE arguments
    parser.add_argument('--ae_name', type=str, default='DAE')
    parser.add_argument('--ae_model', type=str, default='DAE_Model')
    parser.add_argument('--dae_ckpt', type=str, default=None,
                        help='Stage-2 fine-tuned DualAE. Default: <checkpoints_dir>/100styles/<ae_name>/final_finetune_diffmlp_decoder_fixed_hybrid_ema_fix.tar')
    parser.add_argument('--window_size', type=int, default=32)
    
    # Style routing
    parser.add_argument('--style_routing', type=str, default='diffmlp', choices=['diffmlp', 'mart'])
    # parser.add_argument('--use_weight_schedule', action='store_true')  # original: OFF unless passed
    parser.add_argument('--use_weight_schedule', action=argparse.BooleanOptionalAction, default=True,
                        help='Block-wise linear style weight w in [0,1] across DiffMLP blocks (thesis Eq. 4.11-4.13). '
                             'ON by default; disable with --no-use_weight_schedule.')
    parser.add_argument('--mart_style_weight', type=float, default=1.0)
    
    # Dataset arguments
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--styles', type=str, nargs='+', default=["Aeroplane", "Chicken", "Robot", "Superman", "ArmsFolded"])
    
    # Video encoder
    parser.add_argument('--video_encoder', type=str, default='vivit', choices=['vivit', 'timesformer'])
    
    # Input arguments
    parser.add_argument('--text_prompt', type=str, default='')
    parser.add_argument('--text_path', type=str, default='')
    parser.add_argument('--motion_length', type=int, default=120)
    
    # Style video arguments
    parser.add_argument('--style_video', type=str, default=None)
    parser.add_argument('--style_video_b', type=str, default=None)
    
    # Standard mode flags
    parser.add_argument('--generate_quad', action='store_true')
    parser.add_argument('--interpolate', action='store_true')
    parser.add_argument('--blend_weight', type=float, default=0.5)
    parser.add_argument('--save_blend_endpoints', action='store_true')
    
    # Generation arguments
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--cfg_scale', type=float, default=4.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--repeat_times', type=int, default=1)
    parser.add_argument('--hard_pseudo_reorder', action='store_true')

    # 3-way CFG arguments (Phase 1 experiments)
    parser.add_argument('--cfg_mode', type=str, default='2way',
                        choices=['2way', '3way_additive', '3way_style_first'],
                        help='CFG formulation. 2way is the baseline; 3way modes require style input.')
    parser.add_argument('--cfg_text', type=float, default=None,
                        help='Text CFG scale for 3-way modes. Falls back to --cfg_scale if unset.')
    parser.add_argument('--cfg_style', type=float, default=None,
                        help='Style CFG scale for 3-way modes. Falls back to --cfg_scale if unset.')
    
    # Trajectory refinement
    parser.add_argument('--use_refinement', action='store_true',
                        help='Apply TrajectoryRefinementNet to decoded motion root features.')
    parser.add_argument('--refinement_ckpt', type=str,
                        default='./checkpoints/refinement/refine_nofoot/best.tar',
                        help='Path to refinement net checkpoint (.tar with "model" state dict).')
    parser.add_argument('--refinement_width', type=int, default=512)
    parser.add_argument('--refinement_depth', type=int, default=3)
    parser.add_argument('--refinement_dilation_growth_rate', type=int, default=3)

    # Output & Runtime
    parser.add_argument('--output_dir', type=str, default='./generation')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    
    # =========================================================================
    # EXPERIMENT FLAGS (8.1-8.7, excluding 8.4)
    # =========================================================================
    parser.add_argument('--exp_reverse_decoder', action='store_true',
                        help='8.1: Decode neutral latents with both DAE and AE')
    parser.add_argument('--exp_decoder_interp', action='store_true',
                        help='8.2: Blend decoder outputs for style strength control')
    parser.add_argument('--exp_content_preserve', action='store_true',
                        help='8.3: Verify content differs with same style + different texts')
    parser.add_argument('--exp_per_style', action='store_true',
                        help='8.5: Generate for multiple styles')
    parser.add_argument('--exp_schedule_ablation', action='store_true',
                        help='8.6: Test different weight schedules')
    parser.add_argument('--exp_cfg_sweep', action='store_true',
                        help='8.7: Sweep CFG scales')
    parser.add_argument('--exp_cfg_grid', action='store_true',
                        help='8.8: Grid-search (cfg_text x cfg_style) across 3-way modes')

    # Experiment-specific arguments
    parser.add_argument('--style_videos', type=str, nargs='+', default=None,
                        help='List of style videos for per-style experiment')
    parser.add_argument('--num_styles', type=int, default=5,
                        help='Number of styles to test (if --style_videos not provided)')
    parser.add_argument('--grid_cfg_text', type=float, nargs='+', default=None,
                        help='cfg_text grid for --exp_cfg_grid (default: 2.5 4.0 5.5 7.0)')
    parser.add_argument('--grid_cfg_style', type=float, nargs='+', default=None,
                        help='cfg_style grid for --exp_cfg_grid (default: 1.0 1.5 2.0 3.0)')
    parser.add_argument('--grid_cfg_modes', type=str, nargs='+', default=None,
                        choices=['3way_additive', '3way_style_first'],
                        help='Modes for --exp_cfg_grid (default: both)')
    
    args = parser.parse_args()
    main(args)