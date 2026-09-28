"""
Export MM-MARDM generated motions for comparative evaluation.

Reads a shared manifest of test samples (styled + base) and generates:
  - feat67_{manifest_id:05d}.npy  (shape [T, 67], normalized features)
  - joints_{manifest_id:05d}.npy  (shape [T, 22, 3], recovered joint positions)

Two CFG modes are supported via --cfg_mode:
  - 2way:           cfg_scale=4.5  (standard 2-way CFG)
  - 3way_additive:  cfg_text=4.5, cfg_style=2.0  (independent text+style CFG)

Usage:
    python export_for_comparison.py --cfg_mode 2way
    python export_for_comparison.py --cfg_mode 3way_additive
"""

import os
import sys
import json
import time
import logging
import argparse
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch
import random
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Model imports  (mirrors evaluate_MARDM.py)
# ---------------------------------------------------------------------------
from models.AE import DAE_models, AE_models
from models.MARDM import MARDM_models

# Video encoder imports
from transformers import VivitModel, VivitImageProcessor

# Motion recovery
from utils.motion_process import recover_from_ric

# Video loading (decord for frame extraction, same as dataset class)
try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ===========================================================================
#  Logging
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
#  Seed
# ===========================================================================
def set_seed(seed: int):
    """Exact copy of evaluate_MARDM.py set_seed()."""
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
#  Checkpoint loading  (mirrors evaluate_MARDM.py)
# ===========================================================================
def load_checkpoint(model, checkpoint_path, device, key="ema_mardm"):
    """Load MARDM checkpoint, returning (epoch, weight_schedule, style_routing)."""
    log.info("Loading checkpoint from: %s", checkpoint_path)
    # checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Load on CPU: the file also holds optimizer state and a second model copy (~7.8 GB); only the
    # selected weights should reach the GPU (load_state_dict copies onto the model's device).
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    keys_to_try = [key, "ema_mardm", "mardm", "model", "state_dict"]
    state_dict = None
    for k in keys_to_try:
        if k in checkpoint:
            state_dict = checkpoint[k]
            log.info("  Using weights from key: '%s'", k)
            break
    if state_dict is None:
        state_dict = checkpoint
        log.info("  Using checkpoint directly as state_dict")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_filtered = [k for k in missing if not k.startswith("clip_model.")]
    if missing_filtered:
        log.warning("  Missing keys (non-CLIP): %s", missing_filtered[:5])
    if unexpected:
        log.warning("  Unexpected keys: %s", unexpected[:5])

    epoch = checkpoint.get("ep", 0)
    weight_schedule = checkpoint.get("weight_schedule", None)
    style_routing = checkpoint.get("style_routing", None)

    if weight_schedule is not None:
        log.info("  Loaded weight_schedule (length=%d)", len(weight_schedule))
    else:
        log.warning("  No weight_schedule found in checkpoint")

    return epoch, weight_schedule, style_routing


def get_style_weight_schedule(model, args):
    """Generate the per-block style weight schedule (mirrors evaluate_MARDM.py)."""
    if args.style_routing == "diffmlp":
        num_blocks = model.DiffMLPs.get_total_blocks()
        if args.use_weight_schedule:
            return np.linspace(0.0, 1.0, num_blocks).tolist()
        else:
            return [1.0] * num_blocks
    else:
        num_blocks = len(model.MARTransformer)
        return [args.mart_style_weight] * num_blocks


# ===========================================================================
#  Video loading helpers
# ===========================================================================
def load_video_frames_decord(video_path: str, num_frames: int = 32) -> list:
    """Load video frames using decord (same approach as the dataset class)."""
    vr = VideoReader(str(video_path), ctx=decord_cpu(0), width=224, height=224)
    video_length = len(vr)
    if video_length == 0:
        raise ValueError(f"Empty video: {video_path}")
    frame_indices = np.linspace(0, video_length - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()
    return list(frames)


def load_video_frames_cv2(video_path: str, num_frames: int = 32,
                          target_size: tuple = (224, 224)) -> list:
    """Fallback: load video frames using OpenCV (mirrors sample_new.py)."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
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


def load_video_frames(video_path: str, num_frames: int = 32) -> list:
    """Load video frames using best available backend."""
    if DECORD_AVAILABLE:
        return load_video_frames_decord(video_path, num_frames)
    elif CV2_AVAILABLE:
        return load_video_frames_cv2(video_path, num_frames)
    else:
        raise ImportError("Neither decord nor cv2 is available for video loading.")


# ===========================================================================
#  Joint recovery
# ===========================================================================
def feat67_to_joints(feat67_denorm: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """Convert denormalized [T, 67] features to [T, 22, 3] joint positions."""
    joints = recover_from_ric(
        torch.from_numpy(feat67_denorm).float(), joints_num
    ).numpy()
    return joints


# ===========================================================================
#  Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Export MM-MARDM generated motions for comparative evaluation"
    )

    # --- Manifest ---
    parser.add_argument(
        "--manifest", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparative_eval", "manifest.json"),
        help="Path to the shared test manifest",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Override path_roots['mm_mardm'] from manifest (for cluster runs)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory (default: manifest_parent/mardm_{cfg_mode})",
    )

    # --- CFG mode ---
    parser.add_argument(
        "--cfg_mode", type=str, default="2way",
        choices=["2way", "3way_additive"],
        help="CFG mode: 2way (cfg_scale=4.5) or 3way_additive (cfg_text=4.5, cfg_style=2.0)",
    )
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="Override cfg_scale (2way mode, default 4.5)")
    parser.add_argument("--cfg_text", type=float, default=None,
                        help="Override cfg_text (3way mode, default 4.5)")
    parser.add_argument("--cfg_style", type=float, default=None,
                        help="Override cfg_style (3way mode, default 2.0)")

    # --- Model architecture (must match training) ---
    parser.add_argument("--model", type=str, default="MARDM-DDPM-XL",
                        choices=["MARDM-DDPM-XL", "MARDM-SiT-XL"])
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_key", type=str, default="ema_mardm")
    parser.add_argument("--ae_name", type=str, default="DAE")
    parser.add_argument("--ae_model", type=str, default="DAE_Model")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--style_routing", type=str, default="diffmlp",
                        choices=["diffmlp", "mart"])
    # parser.add_argument("--use_weight_schedule", action="store_true",
    #                     help="Use gradual weight schedule for style injection")
    parser.add_argument("--use_weight_schedule", action=argparse.BooleanOptionalAction, default=True,
                        help="Block-wise style weight schedule (thesis default ON); disable with --no-use_weight_schedule")
    parser.add_argument("--mardm_ckpt", type=str, default=None, help="Override Stage-2 MARDM checkpoint path")
    parser.add_argument("--dae_ckpt", type=str, default=None, help="Override Stage-2 fine-tuned DualAE checkpoint path")
    parser.add_argument("--mart_style_weight", type=float, default=1.0)

    # --- Generation ---
    parser.add_argument("--timesteps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=3407)

    # --- Runtime ---
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for base sample generation")
    parser.add_argument("--use_dae_for_base", action="store_true", default=True,
                        help="Decode base (unstyled) samples with DAE instead of AE (default: True)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples whose output files already exist")
    parser.add_argument("--only_transfer", action="store_true",
                        help="Only export transfer samples (skip styled + base)")
    parser.add_argument("--skip_base", action="store_true",
                        help="Skip base sample export (useful for styled-only sweeps)")
    parser.add_argument("--skip_styled", action="store_true",
                        help="Skip styled sample export")
    parser.add_argument("--cfg_style_sweep", nargs="+", type=float, default=None,
                        help="Run styled-only sweep over multiple cfg_style values. "
                             "Loads models once, generates for each value. "
                             "E.g. --cfg_style_sweep 1.0 1.5 2.0 2.5 3.0")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve CFG parameters from mode
    # ------------------------------------------------------------------
    if args.cfg_style_sweep:
        args.cfg_mode = "3way_additive"
        args.skip_base = True

    if args.cfg_mode == "2way":
        cfg_scale = args.cfg_scale if args.cfg_scale is not None else 4.5
        cfg_text = None
        cfg_style = None
    elif args.cfg_mode == "3way_additive":
        cfg_scale = args.cfg_scale if args.cfg_scale is not None else 4.5
        cfg_text = args.cfg_text if args.cfg_text is not None else 4.5
        cfg_style = args.cfg_style if args.cfg_style is not None else 2.0
    else:
        raise ValueError(f"Unknown cfg_mode: {args.cfg_mode}")

    # ------------------------------------------------------------------
    # Seed & device
    # ------------------------------------------------------------------
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("  GPU: %s", torch.cuda.get_device_name(args.device))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    manifest_path = Path(args.manifest)
    log.info("Loading manifest from: %s", manifest_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    metadata = manifest["metadata"]
    styles = metadata["styles"]
    num_classes = len(styles)

    if args.data_root:
        path_root = Path(args.data_root)
        log.info("Using --data_root override: %s", path_root)
    else:
        path_root = Path(manifest["path_roots"]["mm_mardm"])
    rel = manifest["relative_paths"]

    # 100STYLE paths
    style_motion_dir = path_root / rel["100style"]["mm_mardm"]["motion_dir"]
    style_video_dir = path_root / rel["100style"]["mm_mardm"]["video_dir"]
    style_mean_path = path_root / rel["100style"]["mm_mardm"]["mean"]
    style_std_path = path_root / rel["100style"]["mm_mardm"]["std"]

    # Load stats
    dim_pose = 67
    style_mean = np.load(str(style_mean_path))[:dim_pose]
    style_std = np.load(str(style_std_path))[:dim_pose]

    hml3d_mean_path = path_root / rel["humanml3d"]["mm_mardm"]["mean"]
    hml3d_std_path = path_root / rel["humanml3d"]["mm_mardm"]["std"]
    hml3d_latent_dir = path_root / rel["humanml3d"]["mm_mardm"]["latent_dir"]
    hml3d_mean = np.load(str(hml3d_mean_path))[:dim_pose]
    hml3d_std = np.load(str(hml3d_std_path))[:dim_pose]

    styled_samples = manifest["styled_samples"]
    base_samples = manifest["base_samples"]
    transfer_samples = manifest.get("transfer_samples", [])
    log.info("Manifest loaded: %d styled, %d base, %d transfer samples",
             len(styled_samples), len(base_samples), len(transfer_samples))

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    if args.cfg_mode == "3way_additive":
        out_tag = f"mardm_3way_t{cfg_text}_s{cfg_style}"
    elif args.cfg_mode == "2way" and args.cfg_scale is not None:
        out_tag = f"mardm_2way_c{cfg_scale}"
    else:
        out_tag = f"mardm_{args.cfg_mode}"
    if args.output_dir:
        comp_root = Path(args.output_dir)
    else:
        comp_root = manifest_path.parent
    styled_out = comp_root / out_tag / "styled"
    base_out = comp_root / out_tag / "base"
    transfer_out = comp_root / out_tag / "transfer"
    styled_out.mkdir(parents=True, exist_ok=True)
    base_out.mkdir(parents=True, exist_ok=True)
    transfer_out.mkdir(parents=True, exist_ok=True)
    log.info("Output dirs: styled=%s  base=%s  transfer=%s", styled_out, base_out, transfer_out)

    # ------------------------------------------------------------------
    # Load models  (mirrors evaluate_MARDM.py lines 1037-1088)
    # ------------------------------------------------------------------
    log.info("Loading AE (HumanML3D)...")
    ae = AE_models["AE_Model"](input_width=dim_pose)
    ae_ckpt = torch.load(
        pjoin(args.checkpoints_dir, "t2m", "AE", "model", "latest.tar"),
        map_location=device,
    )
    ae.load_state_dict(ae_ckpt["ae"])
    ae.to(device).eval()

    log.info("Loading DAE (100STYLES)...")
    dae = DAE_models["DAE_Model"](
        window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose
    )
    dae_ckpt_path = args.dae_ckpt or pjoin(
        args.checkpoints_dir, "100styles", args.ae_name,
        "final_finetune_diffmlp_decoder_fixed_hybrid_ema_fix.tar",
    )
    log.info("  DAE checkpoint: %s", dae_ckpt_path)
    dae_ckpt = torch.load(dae_ckpt_path, map_location=device, weights_only=False)
    dae.load_state_dict(dae_ckpt["ae"])
    dae.to(device).eval()

    log.info("Loading ViViT video encoder...")
    processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    vmodel = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400").to(device)
    vmodel.eval()
    for p in vmodel.parameters():
        p.requires_grad = False

    log.info("Loading MARDM (%s)...", args.model)
    mardm = MARDM_models[args.model](
        ae_dim=dae.output_emb_width,
        cond_mode="text",
        style_routing=args.style_routing,
        style_dim=512,
    )
    mardm_ckpt_path = args.mardm_ckpt or pjoin(
        args.checkpoints_dir, "t2m", args.model, "model",
        "final_diffmlp_diff_v4_cfg_cross_batch_hybrid_ema_fix.tar",
    )
    _, w_schedule_ckpt, _ = load_checkpoint(mardm, mardm_ckpt_path, device, key=args.checkpoint_key)
    mardm.to(device).eval()

    w_schedule = get_style_weight_schedule(mardm, args)
    log.info("Style weight schedule: %d blocks, range [%.2f, %.2f]",
             len(w_schedule), w_schedule[0], w_schedule[-1])

    # ==================================================================
    #  STYLED SAMPLES  (with optional cfg_style sweep)
    # ==================================================================
    skip_styled = args.only_transfer or args.skip_styled
    skip_base = args.only_transfer or args.skip_base

    if args.only_transfer:
        log.info("--only_transfer: skipping styled + base")

    sweep_values = args.cfg_style_sweep if args.cfg_style_sweep else [cfg_style]
    failed_styled = []

    if not skip_styled:
        if args.cfg_style_sweep:
            log.info("=" * 70)
            log.info("CFG STYLE SWEEP: %d values  cfg_text=%.1f  cfg_style=%s",
                     len(sweep_values), cfg_text, sweep_values)
            log.info("=" * 70)

            # Pre-encode all samples (motion + video → style latents)
            # These are identical across cfg_style values
            log.info("Pre-encoding %d styled samples (ViViT + DAE)...", len(styled_samples))
            encoded_cache = {}
            for sample in tqdm(styled_samples, desc="Pre-encoding"):
                manifest_id = sample["manifest_id"]
                sample_id = sample["sample_id"]
                try:
                    motion_path = style_motion_dir / f"{sample_id}.npy"
                    raw_motion = np.load(str(motion_path))[:, :dim_pose]

                    unit_length = metadata.get("unit_length", 4)
                    max_motion_length = metadata.get("max_motion_length", 196)
                    m_length = (sample["length_frames"] // unit_length) * unit_length
                    m_length = min(max_motion_length, m_length)

                    motion_cropped = raw_motion[:m_length]
                    motion_norm = (motion_cropped - style_mean) / style_std

                    if m_length < max_motion_length:
                        padding = np.zeros((max_motion_length - m_length, dim_pose))
                        motion_padded = np.concatenate([motion_norm, padding], axis=0)
                    else:
                        motion_padded = motion_norm

                    motion_tensor = torch.from_numpy(motion_padded).float().unsqueeze(0).to(device)

                    video_ref_name = sample["video_ref"]["mm_mardm"]
                    video_path = style_video_dir / video_ref_name
                    video_frames = load_video_frames(str(video_path), num_frames=32)
                    inputs = processor([video_frames], return_tensors="pt").to(device)

                    with torch.no_grad():
                        vid_tensors = vmodel(**inputs).last_hidden_state
                        z_style, raw_video_latents = dae.encode(motion_tensor, vid_tensors)

                    encoded_cache[manifest_id] = {
                        "caption": sample["captions"][0],
                        "m_length": m_length,
                        "m_lens": torch.tensor([m_length // unit_length], device=device),
                        "raw_video_latents": raw_video_latents,
                    }
                except Exception as e:
                    log.error("FAILED pre-encode manifest_id=%d: %s", manifest_id, e)

            log.info("Pre-encoded %d/%d samples", len(encoded_cache), len(styled_samples))

            # Sweep over cfg_style values
            for sweep_s in sweep_values:
                sweep_tag = f"mardm_3way_t{cfg_text}_s{sweep_s}"
                sweep_out = comp_root / sweep_tag / "styled"
                sweep_out.mkdir(parents=True, exist_ok=True)

                log.info("-" * 70)
                log.info("Sweep cfg_style=%.1f  -> %s", sweep_s, sweep_out)

                sweep_failed = []
                sweep_skipped = 0

                for sample in tqdm(styled_samples, desc=f"cfg_style={sweep_s}"):
                    manifest_id = sample["manifest_id"]
                    if manifest_id not in encoded_cache:
                        continue

                    if args.resume:
                        f_path = sweep_out / f"feat67_{manifest_id:05d}.npy"
                        j_path = sweep_out / f"joints_{manifest_id:05d}.npy"
                        if f_path.exists() and j_path.exists():
                            sweep_skipped += 1
                            continue

                    enc = encoded_cache[manifest_id]
                    try:
                        set_seed(args.seed)
                        with torch.no_grad():
                            generated_latents = mardm.generate(
                                conds=[enc["caption"]],
                                m_lens=enc["m_lens"],
                                timesteps=args.timesteps,
                                cond_scale=cfg_scale,
                                raw_style_latents=enc["raw_video_latents"],
                                style_weight_schedule=w_schedule,
                                cfg_mode="3way_additive",
                                cfg_text=cfg_text,
                                cfg_style=sweep_s,
                            )
                            generated_motion = dae.decode(generated_latents)

                        feat67 = generated_motion[0].cpu().numpy()[:enc["m_length"]]
                        np.save(str(sweep_out / f"feat67_{manifest_id:05d}.npy"), feat67)

                        feat67_denorm = feat67 * hml3d_std + hml3d_mean
                        joints = feat67_to_joints(feat67_denorm, joints_num=22)
                        np.save(str(sweep_out / f"joints_{manifest_id:05d}.npy"), joints)

                    except Exception as e:
                        log.error("FAILED sweep cfg_style=%.1f manifest_id=%d: %s", sweep_s, manifest_id, e)
                        sweep_failed.append(manifest_id)

                n_ok = len(encoded_cache) - len(sweep_failed) - sweep_skipped
                log.info("  cfg_style=%.1f: %d ok, %d skipped (resume), %d failed",
                         sweep_s, n_ok, sweep_skipped, len(sweep_failed))

                # Save per-sweep run log
                sweep_log = {
                    "cfg_mode": "3way_additive", "cfg_text": cfg_text, "cfg_style": sweep_s,
                    "cfg_scale": cfg_scale, "timesteps": args.timesteps, "seed": args.seed,
                    "styled_total": len(styled_samples), "styled_failed": len(sweep_failed),
                    "use_weight_schedule": args.use_weight_schedule,
                }
                log_path = comp_root / sweep_tag / "export_run_log.json"
                with open(str(log_path), "w") as f:
                    json.dump(sweep_log, f, indent=2)

            log.info("=" * 70)
            log.info("SWEEP COMPLETE: %d cfg_style values x %d samples", len(sweep_values), len(encoded_cache))
            log.info("=" * 70)

        else:
            # Normal single-run styled generation
            log.info("=" * 70)
            log.info("Generating %d STYLED samples  (cfg_mode=%s)", len(styled_samples), args.cfg_mode)
            log.info("=" * 70)

            for sample in tqdm(styled_samples, desc="Styled samples"):
                manifest_id = sample["manifest_id"]
                sample_id = sample["sample_id"]
                caption = sample["captions"][0]
                length_frames = sample["length_frames"]

                if args.resume:
                    f_path = styled_out / f"feat67_{manifest_id:05d}.npy"
                    j_path = styled_out / f"joints_{manifest_id:05d}.npy"
                    if f_path.exists() and j_path.exists():
                        continue

                try:
                    motion_path = style_motion_dir / f"{sample_id}.npy"
                    raw_motion = np.load(str(motion_path))[:, :dim_pose]

                    unit_length = metadata.get("unit_length", 4)
                    max_motion_length = metadata.get("max_motion_length", 196)
                    m_length = (length_frames // unit_length) * unit_length
                    m_length = min(max_motion_length, m_length)

                    motion_cropped = raw_motion[:m_length]
                    motion_norm = (motion_cropped - style_mean) / style_std

                    if m_length < max_motion_length:
                        padding = np.zeros((max_motion_length - m_length, dim_pose))
                        motion_padded = np.concatenate([motion_norm, padding], axis=0)
                    else:
                        motion_padded = motion_norm

                    motion_tensor = torch.from_numpy(motion_padded).float().unsqueeze(0).to(device)

                    video_ref_name = sample["video_ref"]["mm_mardm"]
                    video_path = style_video_dir / video_ref_name
                    video_frames = load_video_frames(str(video_path), num_frames=32)

                    inputs = processor([video_frames], return_tensors="pt").to(device)

                    with torch.no_grad():
                        vid_tensors = vmodel(**inputs).last_hidden_state
                        z_style, raw_video_latents = dae.encode(motion_tensor, vid_tensors)
                        m_lens = torch.tensor([m_length // unit_length], device=device)

                        generated_latents = mardm.generate(
                            conds=[caption],
                            m_lens=m_lens,
                            timesteps=args.timesteps,
                            cond_scale=cfg_scale,
                            raw_style_latents=raw_video_latents,
                            style_weight_schedule=w_schedule,
                            cfg_mode=args.cfg_mode,
                            cfg_text=cfg_text,
                            cfg_style=cfg_style,
                        )
                        generated_motion = dae.decode(generated_latents)

                    feat67 = generated_motion[0].cpu().numpy()[:m_length]

                    np.save(str(styled_out / f"feat67_{manifest_id:05d}.npy"), feat67)

                    feat67_denorm = feat67 * hml3d_std + hml3d_mean
                    joints = feat67_to_joints(feat67_denorm, joints_num=22)
                    np.save(str(styled_out / f"joints_{manifest_id:05d}.npy"), joints)

                except Exception as e:
                    log.error("FAILED styled sample manifest_id=%d sample_id=%s: %s", manifest_id, sample_id, e)
                    failed_styled.append({"manifest_id": manifest_id, "sample_id": sample_id, "error": str(e)})

            log.info("Styled generation complete. %d/%d succeeded, %d failed.",
                     len(styled_samples) - len(failed_styled), len(styled_samples), len(failed_styled))

    # ==================================================================
    #  BASE SAMPLES  (batched for efficiency)
    # ==================================================================
    failed_base = []
    base_decoder = dae if args.use_dae_for_base else ae

    if not skip_base:
        log.info("=" * 70)
        log.info("Generating %d BASE samples  (cfg_mode=%s)", len(base_samples), args.cfg_mode)
        log.info("=" * 70)

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(base_samples) + batch_size - 1) // batch_size

    for batch_idx in (tqdm(range(num_batches), desc="Base batches") if not skip_base else []):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(base_samples))
        batch = base_samples[batch_start:batch_end]

        # Filter out already-completed samples if resuming
        if args.resume:
            batch = [s for s in batch if not (
                (base_out / f"feat67_{s['manifest_id']:05d}.npy").exists() and
                (base_out / f"joints_{s['manifest_id']:05d}.npy").exists()
            )]
            if not batch:
                continue

        # Collect batch data
        captions = []
        m_lens_list = []
        manifest_ids = []
        sliced_ids = []

        for sample in batch:
            captions.append(sample["caption"])
            m_lens_list.append(sample["length_latent_frames"])
            manifest_ids.append(sample["manifest_id"])
            sliced_ids.append(sample["sliced_id"])

        try:
            m_lens = torch.tensor(m_lens_list, device=device)

            with torch.no_grad():
                # Generate without style (same as evaluate_MARDM.py base generation)
                generated_latents = mardm.generate(
                    conds=captions,
                    m_lens=m_lens,
                    timesteps=args.timesteps,
                    cond_scale=cfg_scale,
                    raw_style_latents=None,       # No style for base
                    style_weight_schedule=None,
                    cfg_mode=args.cfg_mode,
                    cfg_text=cfg_text,
                    cfg_style=cfg_style,
                )

                # Decode base samples (DAE by default for consistency with styled path)
                generated_motion = base_decoder.decode(generated_latents)  # [B, T_out, 67]

            # Per-sample extraction
            gen_np = generated_motion.cpu().numpy()  # [B, T_out, 67]

            for i, sample in enumerate(batch):
                try:
                    mid = sample["manifest_id"]
                    raw_frames = sample["length_raw_frames"]
                    latent_frames = sample["length_latent_frames"]
                    unit_length = metadata.get("unit_length", 4)

                    # The actual output frame count = latent_frames * unit_length
                    expected_frames = latent_frames * unit_length
                    feat67 = gen_np[i]  # [T_out, 67]
                    feat67 = feat67[:expected_frames]  # Truncate to expected length

                    # Save feat67
                    np.save(str(base_out / f"feat67_{mid:05d}.npy"), feat67)

                    # Denormalize with HumanML3D stats for joint recovery
                    feat67_denorm = feat67 * hml3d_std + hml3d_mean
                    joints = feat67_to_joints(feat67_denorm, joints_num=22)  # [T, 22, 3]

                    # Save joints
                    np.save(str(base_out / f"joints_{mid:05d}.npy"), joints)

                except Exception as e:
                    log.error("FAILED base sample manifest_id=%d sliced_id=%s: %s",
                              sample["manifest_id"], sample["sliced_id"], e)
                    failed_base.append({
                        "manifest_id": sample["manifest_id"],
                        "sliced_id": sample["sliced_id"],
                        "error": str(e),
                    })

        except Exception as e:
            # Entire batch failed -- fall back to per-sample generation
            log.warning("Batch %d failed (%s). Falling back to per-sample generation.", batch_idx, e)
            for sample in batch:
                mid = sample["manifest_id"]
                try:
                    m_lens_single = torch.tensor([sample["length_latent_frames"]], device=device)

                    with torch.no_grad():
                        gen_lat = mardm.generate(
                            conds=[sample["caption"]],
                            m_lens=m_lens_single,
                            timesteps=args.timesteps,
                            cond_scale=cfg_scale,
                            raw_style_latents=None,
                            style_weight_schedule=None,
                            cfg_mode=args.cfg_mode,
                            cfg_text=cfg_text,
                            cfg_style=cfg_style,
                        )
                        gen_motion = base_decoder.decode(gen_lat)  # [1, T_out, 67]

                    unit_length = metadata.get("unit_length", 4)
                    expected_frames = sample["length_latent_frames"] * unit_length
                    feat67 = gen_motion[0].cpu().numpy()[:expected_frames]

                    np.save(str(base_out / f"feat67_{mid:05d}.npy"), feat67)

                    feat67_denorm = feat67 * hml3d_std + hml3d_mean
                    joints = feat67_to_joints(feat67_denorm, joints_num=22)
                    np.save(str(base_out / f"joints_{mid:05d}.npy"), joints)

                except Exception as e2:
                    log.error("FAILED base sample (fallback) manifest_id=%d sliced_id=%s: %s",
                              mid, sample["sliced_id"], e2)
                    failed_base.append({
                        "manifest_id": mid,
                        "sliced_id": sample["sliced_id"],
                        "error": str(e2),
                    })

    if not skip_base:
        log.info("Base generation complete. %d/%d succeeded, %d failed.",
                 len(base_samples) - len(failed_base), len(base_samples), len(failed_base))

    # ==================================================================
    #  TRANSFER SAMPLES
    # ==================================================================
    failed_transfer = []

    if transfer_samples:
        transfer_out.mkdir(parents=True, exist_ok=True)
        log.info("=" * 70)
        log.info("Generating %d TRANSFER samples  (cfg_mode=%s)", len(transfer_samples), args.cfg_mode)
        log.info("=" * 70)

        for sample in tqdm(transfer_samples, desc="Transfer samples"):
            manifest_id = sample["manifest_id"]
            style_ref_id = sample["style_ref_sample_id"]
            caption = sample["text"]
            length_frames = sample["length_frames"]

            if args.resume:
                f_path = transfer_out / f"feat67_{manifest_id:05d}.npy"
                j_path = transfer_out / f"joints_{manifest_id:05d}.npy"
                if f_path.exists() and j_path.exists():
                    continue

            try:
                motion_path = style_motion_dir / f"{style_ref_id}.npy"
                raw_motion = np.load(str(motion_path))[:, :dim_pose]

                unit_length = metadata.get("unit_length", 4)
                max_motion_length = metadata.get("max_motion_length", 196)
                m_length = (length_frames // unit_length) * unit_length
                m_length = min(max_motion_length, m_length)

                motion_cropped = raw_motion[:m_length]
                motion_norm = (motion_cropped - style_mean) / style_std

                if m_length < max_motion_length:
                    padding = np.zeros((max_motion_length - m_length, dim_pose))
                    motion_padded = np.concatenate([motion_norm, padding], axis=0)
                else:
                    motion_padded = motion_norm

                motion_tensor = torch.from_numpy(motion_padded).float().unsqueeze(0).to(device)

                video_ref_name = sample["video_ref"]["mm_mardm"]
                video_path = style_video_dir / video_ref_name
                video_frames = load_video_frames(str(video_path), num_frames=32)

                inputs = processor([video_frames], return_tensors="pt").to(device)

                with torch.no_grad():
                    vid_tensors = vmodel(**inputs).last_hidden_state
                    z_style, raw_video_latents = dae.encode(motion_tensor, vid_tensors)
                    m_lens = torch.tensor([m_length // unit_length], device=device)

                    generated_latents = mardm.generate(
                        conds=[caption],
                        m_lens=m_lens,
                        timesteps=args.timesteps,
                        cond_scale=cfg_scale,
                        raw_style_latents=raw_video_latents,
                        style_weight_schedule=w_schedule,
                        cfg_mode=args.cfg_mode,
                        cfg_text=cfg_text,
                        cfg_style=cfg_style,
                    )
                    generated_motion = dae.decode(generated_latents)

                feat67 = generated_motion[0].cpu().numpy()
                feat67 = feat67[:m_length]

                np.save(str(transfer_out / f"feat67_{manifest_id:05d}.npy"), feat67)

                feat67_denorm = feat67 * hml3d_std + hml3d_mean
                joints = feat67_to_joints(feat67_denorm, joints_num=22)
                np.save(str(transfer_out / f"joints_{manifest_id:05d}.npy"), joints)

            except Exception as e:
                log.error("FAILED transfer sample manifest_id=%d: %s", manifest_id, e)
                failed_transfer.append({"manifest_id": manifest_id, "error": str(e)})

        log.info("Transfer generation complete. %d/%d succeeded, %d failed.",
                 len(transfer_samples) - len(failed_transfer), len(transfer_samples), len(failed_transfer))
    else:
        log.info("No transfer samples in manifest, skipping.")

    # ==================================================================
    #  Summary
    # ==================================================================
    total_time = time.time()  # We'll log the timestamp at least

    log.info("=" * 70)
    log.info("EXPORT COMPLETE")
    log.info("=" * 70)
    log.info("  CFG mode:          %s", args.cfg_mode)
    log.info("  Styled output:     %s", styled_out)
    log.info("  Base output:       %s", base_out)
    log.info("  Transfer output:   %s", transfer_out)
    log.info("  Styled: %d/%d ok   Base: %d/%d ok   Transfer: %d/%d ok",
             len(styled_samples) - len(failed_styled), len(styled_samples),
             len(base_samples) - len(failed_base), len(base_samples),
             len(transfer_samples) - len(failed_transfer), len(transfer_samples))

    # Save a run log
    run_log = {
        "cfg_mode": args.cfg_mode,
        "cfg_scale": cfg_scale,
        "cfg_text": cfg_text,
        "cfg_style": cfg_style,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "checkpoint_key": args.checkpoint_key,
        "model": args.model,
        "style_routing": args.style_routing,
        "use_weight_schedule": args.use_weight_schedule,
        "batch_size": args.batch_size,
        "mardm_ckpt": os.path.abspath(mardm_ckpt_path),
        "dae_ckpt": os.path.abspath(dae_ckpt_path),
        "manifest": os.path.abspath(args.manifest),
        "command": " ".join(sys.argv),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
        "styled_total": len(styled_samples),
        "styled_failed": len(failed_styled),
        "base_total": len(base_samples),
        "base_failed": len(failed_base),
        "transfer_total": len(transfer_samples),
        "transfer_failed": len(failed_transfer),
        "failed_styled": failed_styled,
        "failed_base": failed_base,
        "failed_transfer": failed_transfer,
    }
    log_path = comp_root / out_tag / "export_run_log.json"
    with open(str(log_path), "w") as f:
        json.dump(run_log, f, indent=2)
    log.info("Run log saved to: %s", log_path)

    if failed_styled or failed_base or failed_transfer:
        log.warning("Some samples failed. See %s for details.", log_path)
        return 1
    return 0


if __name__ == "__main__":
    start_time = time.time()
    rc = main()
    elapsed = time.time() - start_time
    log.info("Total wall time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    sys.exit(rc)
