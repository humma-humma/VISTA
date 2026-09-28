import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from models.AE import DAE_models, AE_models
from models.MARDM import MARDM_models
from utils.evaluators import Evaluators
# import wandb
import functools
import glob
import time
import copy
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss, update_ema, savediff, save_upd, ema_decay_warmup, build_training_schedule, reset_ema_style_params
from utils.train_utils import lengths_to_mask
from utils.evaluators import MotionCLIP
from utils.eval_utils import calculate_activation_statistics, calculate_frechet_distance
from train_style_classification import StyleClassification

# === NEW DATASETS & COLLATORS ===
from utils.datasets import (
    Text2MotionDatasetCombined_v4, 
    Text2MotionDatasetCombined_v5,
    mld_collate_paired, 
    mld_collate_async
)

import argparse
import math
from transformers import VivitModel, TimesformerModel, VivitImageProcessor, AutoProcessor

#################################################################################
#                                 Helper functions                              #
#################################################################################

def verify_dae_weights(dae, original_frozen_weights, original_decoder_weights, device, tolerance=1e-6):
    # Check frozen components
    frozen_diffs = []
    
    for name, param in dae.motion_encoder.named_parameters():
        key = f'motion_encoder.{name}'
        if key in original_frozen_weights:
            original = original_frozen_weights[key].to(device)
            diff = (param - original).abs().max().item()
            frozen_diffs.append((key, diff))
    
    for name, param in dae.video_adapter.named_parameters():
        key = f'video_adapter.{name}'
        if key in original_frozen_weights:
            original = original_frozen_weights.to(device)
            diff = (param - original).abs().max().item()
            frozen_diffs.append((key, diff))
    
    for name, param in dae.video_encoder.named_parameters():
        key = f'video_encoder.{name}'
        if key in original_frozen_weights:
            original = original_frozen_weights[key].to(device)
            diff = (param - original).abs().max().item()
            frozen_diffs.append((key, diff))
    
    if hasattr(dae, 'classifier_head'):
        for name, param in dae.classifier_head.named_parameters():
            key = f'classifier_head.{name}'
            if key in original_frozen_weights:
                original = original_frozen_weights[key].to(device)
                diff = (param - original).abs().max().item()
                frozen_diffs.append((key, diff))
    
    frozen_max_diff = max([d[1] for d in frozen_diffs]) if frozen_diffs else 0
    frozen_unchanged = frozen_max_diff < tolerance
    
    # Check decoder
    decoder_diffs = []
    for name, param in dae.decoder.named_parameters():
        key = f'decoder.{name}'
        if key in original_decoder_weights:
            original = original_decoder_weights[key].to(device)
            diff = (param - original).abs().max().item()
            decoder_diffs.append((key, diff))
    
    decoder_max_diff = max([d[1] for d in decoder_diffs]) if decoder_diffs else 0
    decoder_changed = decoder_max_diff > tolerance
    
    # Build report
    report = []
    report.append(f"  Frozen components max diff: {frozen_max_diff:.2e} ({'✓ FROZEN' if frozen_unchanged else '✗ CHANGED!'})")
    report.append(f"  Decoder max diff: {decoder_max_diff:.2e} ({'✓ TRAINED' if decoder_changed else '✗ UNCHANGED!'})")
    
    if not frozen_unchanged:
        frozen_diffs.sort(key=lambda x: x[1], reverse=True)
        report.append(f"  WARNING: Top changed frozen params:")
        for name, diff in frozen_diffs[:3]:
            report.append(f"    - {name}: {diff:.2e}")
    
    if not decoder_changed:
        report.append(f"  WARNING: Decoder weights haven't changed! Training may not be working.")
    
    report_str = "\n".join(report)
    
    return frozen_unchanged, decoder_changed, report_str

def _finite_item(x):
    """float(x) for logging, 0.0 if non-finite (so one bad step cannot poison epoch averages)."""
    v = x.item() if isinstance(x, torch.Tensor) else float(x)
    return v if math.isfinite(v) else 0.0


def delete_old_checkpoints(model_dir, patterns=("dae_epoch_*.tar", "epoch_*.tar", "LoRA_epoch_*.tar")):
    for pat in patterns:
        for path in glob.glob(pjoin(model_dir, pat)):
            try:
                os.remove(path)
            except OSError as e:
                print(f"[Cleanup] Could not remove {path}: {e}")

def compute_weighted_loss(pred, target, mask, lengths, focus_ratio=0.9):
    """
    Compute reconstruction loss with higher weight on generated (masked) positions.
    """
    b, d, t = pred.shape
    l = mask.shape[1]
    
    mask_upsampled = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(b, -1)
    mask_upsampled = mask_upsampled[:, :t]
    
    pad_mask = lengths_to_mask(lengths, t)
    mse_per_pos = ((pred - target) ** 2).mean(dim=1)
    
    weights = torch.ones_like(mse_per_pos)
    generated_weight = focus_ratio
    original_weight = 1.0 - focus_ratio
    
    weights = torch.where(mask_upsampled, 
                          torch.full_like(weights, generated_weight),
                          torch.full_like(weights, original_weight))
    
    weights = weights * pad_mask.float()
    mse_per_pos = mse_per_pos * pad_mask.float()
    
    weighted_loss = (mse_per_pos * weights).sum() / (weights.sum() + 1e-8)
    
    return weighted_loss

def compute_style_loss(motion_pred, motion_gt, style_classifier):
    """
    Style loss via feature matching with trained classifier.
    """
    if motion_pred.shape[-1] != 67:
        pred_input = motion_pred.permute(0, 2, 1)
        gt_input = motion_gt.permute(0, 2, 1)
    else:
        pred_input = motion_pred
        gt_input = motion_gt
    
    _, feat_pred = style_classifier(pred_input, stage="Both")
    with torch.no_grad():
        _, feat_gt = style_classifier(gt_input, stage="Both")
    
    return F.mse_loss(feat_pred, feat_gt)

def compute_content_loss(motion_pred, m_lens, texts, motion_clip):
    """
    Content loss with proper gradient flow.
    MotionCLIP expects [B, T, 67].
    """
    if motion_pred.shape[-1] == 67:
        motion_input = motion_pred
    elif motion_pred.shape[1] == 67:
        motion_input = motion_pred.permute(0, 2, 1)
    else:
        raise ValueError(f"Cannot determine motion format: {motion_pred.shape}")
    
    motion_emb = motion_clip.encode_motion(motion_input, m_lens)
    
    with torch.no_grad():
        text_emb = motion_clip.encode_text(texts)
    
    motion_emb = F.normalize(motion_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    
    similarity = (motion_emb * text_emb).sum(dim=-1)
    content_loss = (1 - similarity).mean()
    
    return content_loss


#################################################################################
#                    PHASE 3: Cross-Batch Loss Functions                        #
#################################################################################

def compute_cross_batch_direct(motion_cross, motion_style_ref, text_hml3d, m_lens, 
                                style_classifier, motion_clip):
    """
    Expt 1: Direct supervision with content + style losses.
    """
    content_loss = compute_content_loss(motion_cross, m_lens, text_hml3d, motion_clip)
    style_loss = compute_style_loss(motion_cross, motion_style_ref, style_classifier)
    total_loss = content_loss + style_loss
    return total_loss, content_loss, style_loss


def compute_cross_batch_style_only(motion_cross, motion_style_ref, style_classifier):
    """
    Expt 2: Style cycle only — trust text conditioning for content.
    """
    style_loss = compute_style_loss(motion_cross, motion_style_ref, style_classifier)
    return style_loss


@torch.no_grad()
def compute_styled_fid(ema_mardm, dae, val_loader, eval_wrapper, vmodel, processor,
                       w_schedule, args, device):
    ema_mardm.eval()
    dae.eval()
    vmodel.eval()

    gt_embs, gen_embs = [], []
    for batch_data in tqdm(val_loader, desc="Styled FID"):
        motion_style = batch_data['motion_styled'].float().to(device)
        len_style    = batch_data['length_styled'].to(device) // 4
        text_style   = batch_data['text_styled']
        video_style  = batch_data['video_styled']

        inputs = processor(video_style, return_tensors="pt").to(device)
        vid_tensors = vmodel(**inputs).last_hidden_state
        _, raw_video_latents = dae.encode(motion_style, vid_tensors)

        generated_latents = ema_mardm.generate(
            conds=text_style, m_lens=len_style,
            timesteps=args.fid_timesteps, cond_scale=args.fid_cond_scale,
            raw_style_latents=raw_video_latents,
            style_weight_schedule=w_schedule,
        )
        generated_motion = dae.decode(generated_latents)

        m_lens_full = len_style * 4
        gt_emb,  _ = eval_wrapper.get_motion_embeddings(motion_style,     m_lens_full)
        gen_emb, _ = eval_wrapper.get_motion_embeddings(generated_motion, m_lens_full)
        gt_embs.append(gt_emb.cpu().numpy())
        gen_embs.append(gen_emb.cpu().numpy())

    gt_arr  = np.concatenate(gt_embs,  axis=0)
    gen_arr = np.concatenate(gen_embs, axis=0)
    gt_mu,  gt_cov  = calculate_activation_statistics(gt_arr)
    gen_mu, gen_cov = calculate_activation_statistics(gen_arr)
    return float(calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov))


def main(args):
    #################################################################################
    #                                   Monitoring                                  #
    #################################################################################
    
    
    # wandb.init(
    #     project='Multimodal MARDM Diffusion', 
    #     name=args.exp_name,
    #     config=args,
    # )

    # wandb.init(
    #     project='Multimodal MARDM Diffusion Debugging and Analysis', 
    #     name=args.exp_name,
    #     config=args,
    # )

    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_idx)
        print(f"Using device: {device} ({dev_name})")
    else:
        print(f"Using device: {device} (CPU)")

    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    data_root = f'{args.dataset_dir}/100STYLE-SMPL/'
    prior_data_root = f'{args.dataset_dir}/HumanML3D/'
    dim_pose = 67
    
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    video_dir = pjoin(data_root, 'videos')
    text_dir = pjoin(data_root, 'texts')
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    dict_file = pjoin(data_root, '100STYLE_name_dict_length.txt')
    train_split_file = pjoin(data_root, 'train_100STYLE_Full.txt')
    val_split_file = pjoin(data_root, 'test_100STYLE_Full.txt')

    prior_motion_dir = pjoin(prior_data_root, 'sliced_joint_vecs')
    prior_latent_dir = pjoin(prior_data_root, 'latent_vecs')
    prior_text_dir = pjoin(prior_data_root, 'splits_sliced/texts_sliced')
    prior_mean = np.load(pjoin(prior_data_root, 'Mean.npy'))
    prior_std = np.load(pjoin(prior_data_root, 'Std.npy'))
    prior_train_split_file = pjoin(prior_data_root, 'splits_sliced/train.txt')
    prior_val_split_file = pjoin(prior_data_root, 'splits_sliced/val.txt')

    print("Initializing Datasets...")

    if args.data_mode == 'v4':
        train_dataset = Text2MotionDatasetCombined_v4(
            style_mean=mean, style_std=std, style_split_file=train_split_file, 
            style_motion_dir=motion_dir, style_text_dir=text_dir, style_video_dir=video_dir, style_dict_file=dict_file,
            humanml_mean=prior_mean, humanml_std=prior_std, humanml_split_file=prior_train_split_file, humnaml_motion_dir=prior_motion_dir,
            humanml_latent_dir=prior_latent_dir, humanml_text_dir=prior_text_dir, humanml_dict_file=pjoin(prior_data_root, 'splits_sliced/all_lengths.txt'),
            dim_pose=dim_pose, unit_length=args.unit_length, max_motion_length=args.max_motion_length, epoch_mode='100styles', tiny=False  # was tiny=True: a debug leftover that keeps only 11 clips (0 batches)
        )

        val_dataset_full = Text2MotionDatasetCombined_v4(
            style_mean=mean, style_std=std, style_split_file=val_split_file, 
            style_motion_dir=motion_dir, style_text_dir=text_dir, style_video_dir=video_dir, style_dict_file=dict_file,
            humanml_mean=prior_mean, humanml_std=prior_std, humanml_split_file=prior_val_split_file, humnaml_motion_dir=prior_motion_dir,
            humanml_latent_dir=prior_latent_dir, humanml_text_dir=prior_text_dir, humanml_dict_file=pjoin(prior_data_root, 'splits_sliced/all_lengths.txt'),
            dim_pose=dim_pose, unit_length=args.unit_length, max_motion_length=args.max_motion_length, epoch_mode='100styles', tiny=False  # was tiny=True: a debug leftover that keeps only 11 clips (0 batches)
        )

        val_size = len(val_dataset_full)*2 // 3
        test_size = len(val_dataset_full) - val_size

        val_dataset, test_dataset = torch.utils.data.random_split(
            val_dataset_full, 
            [val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        print(f"Dataset loaded - full val: {len(val_dataset_full)}, val: {len(val_dataset)}, test: {len(test_dataset)}, train: {len(train_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=mld_collate_paired)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=mld_collate_paired)
        eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=mld_collate_paired)

        print(f"DataLoaders initialized. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Eval batches: {len(eval_loader)}")

    #################################################################################
    #                                      Models                                   #
    #################################################################################
    print("Initializing Models...")
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    num_classes = len(args.styles) if args.styles else 100 

      # 1. Load Pretrained DAE for 100STYLES
    print("Loading DAE for 100STYLES...")
    dae = DAE_models[args.ae_model](window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose)
    # dae_ckpt = pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'epoch_119_detach_nostyle_disc.tar')           ## system
    # dae_ckpt = pjoin(args.checkpoints_dir, args.dataset_name, 'DAE_vivit_120_detach_disc_nostyle', 'model', 'final.tar')     ## cluster
    dae_ckpt = args.dae_ckpt or pjoin(args.checkpoints_dir, '100styles', args.ae_name, 'epoch_119_detach_nostyle_disc.tar')
    print(f"  DAE checkpoint: {dae_ckpt}")
    dae.load_state_dict(torch.load(dae_ckpt, map_location=device)['ae'])

    # Store original weights for verification
    original_frozen_weights = {}
    for name, param in dae.motion_encoder.named_parameters():
        original_frozen_weights[f'motion_encoder.{name}'] = param.clone().detach()
    for name, param in dae.video_adapter.named_parameters():
        original_frozen_weights[f'video_adapter.{name}'] = param.clone().detach()
    for name, param in dae.video_encoder.named_parameters():
        original_frozen_weights[f'video_encoder.{name}'] = param.clone().detach()
    if hasattr(dae, 'classifier_head'):
        for name, param in dae.classifier_head.named_parameters():
            original_frozen_weights[f'classifier_head.{name}'] = param.clone().detach()

    original_decoder_weights = {}
    for name, param in dae.decoder.named_parameters():
        original_decoder_weights[f'decoder.{name}'] = param.clone().detach()

    print(f"Stored {len(original_frozen_weights)} frozen params and {len(original_decoder_weights)} trainable decoder params")  

    for param in dae.parameters():
        param.requires_grad = False
    for param in dae.decoder.parameters():
        param.requires_grad = True

    dae.to(device).eval()

    pc_dae = sum(param.numel() for param in dae.parameters())
    pc_dae_decoder = sum(param.numel() for param in dae.decoder.parameters())

    # 2. Load Video Encoder (ViViT)
    print(f"Loading video encoder ({args.video_encoder}) for video style extraction...")
    MODEL_CONFIG = {
        'vivit': {"name": "google/vivit-b-16x2-kinetics400", "processor": "google/vivit-b-16x2-kinetics400", "num_frames": 32},
        'timesformer': {"name": "facebook/timesformer-base-finetuned-k400", "processor": "MCG-NJU/videomae-base", "num_frames": 32},
    }

    if args.video_encoder == 'vivit':
        processor = VivitImageProcessor.from_pretrained(MODEL_CONFIG['vivit']['processor'])
        vmodel = VivitModel.from_pretrained(MODEL_CONFIG['vivit']['name']).to(device)
    elif args.video_encoder == 'timesformer':
        processor = AutoProcessor.from_pretrained(MODEL_CONFIG['timesformer']['processor'])
        vmodel = TimesformerModel.from_pretrained(MODEL_CONFIG['timesformer']['name']).to(device)
    

    # 3. Load Style Classifier for Style Loss
    print("Loading pre-trained style classifier for style loss computation...")
    # style_classifier = StyleClassification(nclasses=20, input_dim=dim_pose, latent_dim=[1, args.latent_dim], ff_size=args.ff_size, num_layers=args.num_layers, num_heads=args.num_heads, ropout=args.dropout).to(device)
    # Thesis run used a 20-class head (see README 'Known issues'); the released classifier has 21 classes.
    style_classifier_path = args.style_classifier_ckpt or pjoin(args.checkpoints_dir, 'style_classifier', 'style_classifier_final.pt')
    sc_ckpt = torch.load(style_classifier_path, map_location=device, weights_only=False)
    sc_nclasses = args.style_cls_nclasses or sc_ckpt['model_state_dict']['classifier.weight'].shape[0]
    style_classifier = StyleClassification(nclasses=sc_nclasses, input_dim=dim_pose, latent_dim=[1, args.latent_dim], ff_size=args.ff_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout).to(device)
    style_classifier.load_state_dict(sc_ckpt['model_state_dict'])
    style_classifier.eval()
    for param in style_classifier.parameters():
        param.requires_grad = False

    pc_style_classifier = sum(param.numel() for param in style_classifier.parameters())

    # 4. Load MotionCLIP for Content Loss
    print("Loading MotionCLIP for content loss...")
    motion_clip = MotionCLIP(in_dim=67).to(device)
    motion_clip_ckpt = torch.load(
        pjoin(args.checkpoints_dir, 't2m', 'text_mot_match_clip', 'model', 'finest.tar'),
        map_location=device
    )
    motion_clip.load_state_dict(motion_clip_ckpt['contrast_model'])
    motion_clip.eval()
    
    pc_motion_clip = sum(param.numel() for param in motion_clip.parameters())
        
    # 5. Initialize Dual-AdaLN MARDM
    print(f"Building MARDM with routing={args.style_routing}...")
    mardm = MARDM_models[args.model](
        ae_dim=dae.output_emb_width, 
        cond_mode='text',
        style_routing=args.style_routing,
        style_dim=512
    )

    # Parameter Count
    pc_mardm_original = 0
    pc_mardm_new_style = 0
    for name, param in mardm.named_parameters():
        if name.startswith('clip_model.'):
            continue
        if 'style_proj' in name or 'style_modulation' in name or 'style_blocks' in name:
            pc_mardm_new_style += param.numel()
        else:
            pc_mardm_original += param.numel()

    pc_mardm_total = pc_mardm_original + pc_mardm_new_style
    total_trainable = pc_dae_decoder + pc_mardm_total

    print('\n================ PARAMETER BREAKDOWN ================')
    print('DAE Encoder/Decoder:         {:>8.3f}M'.format(pc_dae / 1_000_000))
    print('MARDM (Original Base):       {:>8.3f}M'.format(pc_mardm_original / 1_000_000))
    print('MARDM (New Style Additions): {:>8.3f}M'.format(pc_mardm_new_style / 1_000_000))
    print('Style Classifier:            {:>8.3f}M'.format(pc_style_classifier / 1_000_000))
    print('MotionCLIP:                  {:>8.3f}M'.format(pc_motion_clip / 1_000_000))
    print('-----------------------------------------------------')
    print('MARDM (Total Capacity):      {:>8.3f}M'.format(pc_mardm_total / 1_000_000))
    print('Total (DAE Decoder + MARDM): {:>8.3f}M'.format(total_trainable / 1_000_000))
    print('=====================================================\n')
    
    # 6. Load Pretrained MARDM Weights
    resume_ep = None  # epoch stored in the pretrained checkpoint (used for the first save after resuming)
    if args.is_continue:
        pretrained_path = pjoin(args.checkpoints_dir, 't2m', args.model, 'model','humanml3d_latest.tar')
        # checkpoint = torch.load(pretrained_path, map_location=device)
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)  # 4.7 GB file incl. optimizer state
        mardm.load_state_dict(checkpoint['mardm'], strict=False)
        print("Pre-trained MARDM weights loaded.")
        epoch, it = checkpoint['ep'], checkpoint['total_it']
        print("Loaded model epoch:%d iterations:%d" % (epoch, it))
        resume_ep = checkpoint['ep']
        del checkpoint  # ~4.7 GB incl. optimizer state; only the weights and counters are needed

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)

    schedule = build_training_schedule(args.epoch, len(train_loader))
    print(f"[schedule] total_iters={total_iters}, warm_up_iter={schedule['warm_up_iter']}, "
          f"milestones={schedule['lr_milestones']}, ema_warmup_iters={schedule['ema_warmup_iters']}, "
          f"ema_max_decay={schedule['ema_max_decay']}")

    ema_mardm = copy.deepcopy(mardm).eval()
    for param in ema_mardm.parameters():
        param.requires_grad_(False)
        
    mardm.to(device)
    ema_mardm.to(device)

    #################################################################################
    #                            Optimization & Freezing                            #
    #################################################################################
    if args.is_continue and args.freeze_mode != 'none':
        print(f"Routing mode: {args.style_routing}, Freeze mode: {args.freeze_mode}")
        
        old_blocks = mardm.DiffMLPs.net.res_blocks
        new_blocks = mardm.DiffMLPs.net.style_blocks
        
        lr_new = args.lr
        lr_base = args.lr * args.mardm_lr_mult
        param_groups = []

        decoder_params = [p for p in dae.decoder.parameters() if p.requires_grad]
        param_groups.append({'params': decoder_params, 'lr': lr_new})

        for block in mardm.MARTransformer:
            param_groups.append({'params': block.attn.parameters(), 'lr': lr_base})
            param_groups.append({'params': block.mlp.parameters(), 'lr': lr_base})
            param_groups.append({'params': block.adaLN_modulation.parameters(), 'lr': lr_base})
            param_groups.append({'params': block.norm1.parameters(), 'lr': lr_base})
            param_groups.append({'params': block.norm2.parameters(), 'lr': lr_base})
            
            if args.style_routing == 'mart':
                param_groups.append({'params': block.style_modulation.parameters(), 'lr': lr_new})

        param_groups.append({'params': mardm.input_process.parameters(), 'lr': lr_base})
        param_groups.append({'params': mardm.position_enc.parameters(), 'lr': lr_base})
        param_groups.append({'params': mardm.cond_emb.parameters(), 'lr': lr_base})
        param_groups.append({'params': [mardm.mask_latent], 'lr': lr_base})
        param_groups.append({'params': mardm.style_proj.parameters(), 'lr': lr_new})

        if args.freeze_mode == 'differential':
            gradual_lrs = np.linspace(0.0, lr_base, len(old_blocks))
            for i, block in enumerate(old_blocks):
                param_groups.append({'params': block.parameters(), 'lr': gradual_lrs[i]})
        elif args.freeze_mode == 'strict':
            param_groups.append({'params': old_blocks.parameters(), 'lr': 0.0})

        param_groups.append({'params': new_blocks.parameters(), 'lr': lr_new})
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.99), weight_decay=1e-5)

    else:
        print("Applying flat learning rates to all components...")
        combined_params = list(mardm.parameters()) + list(dae.decoder.parameters())
        optimizer = optim.AdamW(combined_params, lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schedule['lr_milestones'], gamma=args.lr_decay)

    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    logger = SummaryWriter(model_dir)
    start_time = time.time()
    
    epoch = 0 if not args.is_continue else epoch
    it = 0 if not args.is_continue else it
    total_epochs = epoch + args.epoch
    print(f'Total Epochs: {total_epochs}, Total Iters: {total_iters}')
    print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
    
    # Calculate Weight Schedule
    if args.style_routing == 'diffmlp':
        num_blocks = mardm.DiffMLPs.get_total_blocks()
        if args.use_weight_schedule:
            w_schedule = np.linspace(0.0, 1.0, num_blocks).tolist()
        else:
            w_schedule = [1.0] * num_blocks
    else:
        num_blocks = len(mardm.MARTransformer)
        w_schedule = [args.mart_style_weight] * num_blocks

    print(f"Style routing: {args.style_routing}")
    print(f"Weight schedule: {len(w_schedule)} blocks, range [{w_schedule[0]:.2f}, {w_schedule[-1]:.2f}]")

    # Styled-FID evaluation setup
    eval_wrapper = Evaluators('t2m', device=device)
    best_fid_styled = 1000.0
    
    # === PHASE 3 CONFIG LOGGING ===
    if args.enable_cross_batch:
        print(f"\n=== PHASE 3: Cross-Batch Training ENABLED ===")
        print(f"  Mode: {args.cross_batch_mode}")
        print(f"  Probability: {args.cross_batch_prob}")
        print(f"  Weight: {args.cross_batch_weight}")
        if 'hybrid' in args.cross_batch_mode:
            print(f"  Cycle Loss Weight: {args.cycle_loss_weight}")
            print(f"  Style Loss Weight: {args.cross_style_loss_weight}")
        if 'weighted' in args.cross_batch_mode:
            print(f"  Focus Ratio: {args.cycle_focus_ratio}")
        print(f"=============================================\n")
    else:
        print("\n=== PHASE 3: Cross-Batch Training DISABLED ===\n")

    log_interval = 100
    worst_loss = 100.0

    while epoch < total_epochs:
        mardm.train()
        dae.train()
        logs = defaultdict(def_value, OrderedDict())
        
        epoch_loss_total = 0.0
        epoch_loss_base = 0.0
        epoch_loss_style = 0.0
        epoch_loss_base_mse = 0.0
        epoch_loss_style_mse = 0.0
        epoch_loss_style_ce = 0.0
        epoch_loss_content_base = 0.0
        epoch_loss_content_style = 0.0
        
        # === PHASE 3: Cross-batch tracking ===
        epoch_loss_cross_cycle = 0.0
        epoch_loss_cross_content = 0.0
        epoch_loss_cross_style = 0.0
        cross_batch_count = 0
        skipped_cross = 0      # cross-batch terms dropped because they were non-finite
        skipped_steps = 0      # optimizer steps skipped because loss/gradients were non-finite

        is_save_epoch = (epoch >= total_epochs - 100) and (epoch % 10 == 0 or epoch == total_epochs - 1)

        with tqdm(train_loader, desc=f"Epoch {epoch} | Training") as t:
            for i, batch_data in enumerate(t):
                it += 1
                if it < schedule['warm_up_iter']:
                    update_lr_warm_up(it, schedule['warm_up_iter'], optimizer, args.lr)

                optimizer.zero_grad()
                loss_total = 0.0

                # =============================================================
                # PASS 1: HumanML3D Base Content (NO STYLE)
                # =============================================================
                z_hml3d = batch_data['latent_humanml'].to(device)
                motion_hml3d = batch_data['motion_humanml'].float().to(device)
                len_hml3d = batch_data['length_humanml'].to(device)
                text_hml3d = batch_data['text_humanml']

                loss_base, full_pred_base, mask_base = mardm.forward_loss(z_hml3d, text_hml3d, len_hml3d, raw_style_latents=None)
                motion_pred_base = dae.decode(full_pred_base)
                recon_loss_base = F.smooth_l1_loss(motion_pred_base, motion_hml3d)
                content_loss_base = compute_content_loss(motion_pred_base, len_hml3d * 4, text_hml3d, motion_clip)

                # =============================================================
                # PASS 2: 100STYLES Stylization
                # =============================================================
                motion_style = batch_data['motion_styled'].float().to(device)
                len_style = batch_data['length_styled'].to(device) // 4
                text_style = batch_data['text_styled']
                
                inputs = processor(batch_data['video_styled'], return_tensors="pt").to(device)
                with torch.no_grad():
                    vid_tensors = vmodel(**inputs).last_hidden_state
                    z_style, raw_video_latents = dae.encode(motion_style, vid_tensors)

                active_style_latents = raw_video_latents
                active_text_style = text_style
                
                if args.enable_cfg_dropout:
                    rand_val = random.random()
                    if rand_val < 0.05:
                        active_text_style = [""] * len(text_style)
                        active_style_latents = None
                    elif rand_val < 0.15:
                        active_text_style = [""] * len(text_style)
                    elif rand_val < 0.25:
                        active_style_latents = None
                        
                loss_style, full_pred_style, mask_style = mardm.forward_loss(z_style, active_text_style, len_style, raw_style_latents=active_style_latents, style_weight_schedule=w_schedule)
                motion_pred = dae.decode(full_pred_style)
                
                recon_loss_style = F.smooth_l1_loss(motion_pred, motion_style)
                style_feat_loss = compute_style_loss(motion_pred, motion_style, style_classifier)
                content_loss_style = compute_content_loss(motion_pred, len_style * 4, text_style, motion_clip)

                # Accumulate Phase 1+2 losses
                loss_total += (loss_base + loss_style) + args.recons_weight*(recon_loss_style + recon_loss_base) + (args.style_weight * style_feat_loss) + (args.content_weight * (content_loss_base + 0.01 * content_loss_style))

                # =============================================================
                # PASS 3: Cross-Batch Training (Phase 3)
                # HumanML3D text + 100STYLES style -> ??? (no GT exists)
                # =============================================================
                cross_cycle_loss = torch.tensor(0.0, device=device)
                cross_content_loss = torch.tensor(0.0, device=device)
                cross_style_loss = torch.tensor(0.0, device=device)

                if args.enable_cross_batch and random.random() < args.cross_batch_prob:

                    # === Expt 1: Direct Supervision (Content + Style) ===
                    if args.cross_batch_mode == 'direct':
                        _, pred_cross, _ = mardm.forward_loss(
                            z_hml3d, text_hml3d, len_hml3d // 4,
                            raw_style_latents=raw_video_latents,
                            style_weight_schedule=w_schedule
                        )
                        motion_cross = dae.decode(pred_cross)

                        cross_loss, cross_content_loss, cross_style_loss = compute_cross_batch_direct(
                            motion_cross, motion_style, text_hml3d, len_hml3d,
                            style_classifier, motion_clip
                        )

                        epoch_loss_cross_content += _finite_item(cross_content_loss)
                        epoch_loss_cross_style += _finite_item(cross_style_loss)

                    # === Expt 2: Style Only ===
                    elif args.cross_batch_mode == 'style_only':
                        _, pred_cross, _ = mardm.forward_loss(
                            z_hml3d, text_hml3d, len_hml3d // 4,
                            raw_style_latents=raw_video_latents,
                            style_weight_schedule=w_schedule
                        )
                        motion_cross = dae.decode(pred_cross)

                        cross_style_loss = compute_cross_batch_style_only(
                            motion_cross, motion_style, style_classifier
                        )
                        cross_loss = cross_style_loss

                        epoch_loss_cross_style += _finite_item(cross_style_loss)

                    # === Expt 3: Latent Cycle (Masked Only) ===
                    elif args.cross_batch_mode == 'latent_cycle':
                        cross_cycle_loss, pred_styled, pred_recovered, cycle_mask = mardm.forward_cycle_loss_masked_only(
                            z_hml3d, text_hml3d, len_hml3d // 4,
                            style_latents=raw_video_latents,
                            style_weight_schedule=w_schedule,
                            detach_pass2=True
                        )
                        cross_loss = cross_cycle_loss

                        epoch_loss_cross_cycle += _finite_item(cross_cycle_loss)

                    # === Expt 4: Latent Cycle (Weighted) ===
                    elif args.cross_batch_mode == 'latent_cycle_weighted':
                        cross_cycle_loss, pred_styled, pred_recovered, cycle_mask = mardm.forward_cycle_loss_weighted(
                            z_hml3d, text_hml3d, len_hml3d // 4,
                            style_latents=raw_video_latents,
                            style_weight_schedule=w_schedule,
                            detach_pass2=True,
                            focus_ratio=args.cycle_focus_ratio
                        )
                        cross_loss = cross_cycle_loss

                        epoch_loss_cross_cycle += _finite_item(cross_cycle_loss)

                    # === Expt 5: Hybrid (Cycle Masked + Style Verification) ===
                    elif args.cross_batch_mode == 'hybrid':
                        # Step 1: Run cycle loss
                        cross_cycle_loss, pred_styled, pred_recovered, cycle_mask = mardm.forward_cycle_loss_masked_only(
                            z_hml3d, text_hml3d, len_hml3d // 4,
                            style_latents=raw_video_latents,
                            style_weight_schedule=w_schedule,
                            detach_pass2=True
                        )

                        # Step 2: Decode styled prediction and compute style loss
                        motion_styled_pred = dae.decode(pred_styled)
                        cross_style_loss = compute_style_loss(motion_styled_pred, motion_style, style_classifier)

                        # Step 3: Combine losses
                        cross_loss = args.cycle_loss_weight * cross_cycle_loss + args.cross_style_loss_weight * cross_style_loss

                        epoch_loss_cross_cycle += _finite_item(cross_cycle_loss)
                        epoch_loss_cross_style += _finite_item(cross_style_loss)

                    # === Expt 6: Hybrid Weighted (Cycle Weighted + Style Verification) ===
                    elif args.cross_batch_mode == 'hybrid_weighted':
                        # Step 1: Run weighted cycle loss
                        cross_cycle_loss, pred_styled, pred_recovered, cycle_mask = mardm.forward_cycle_loss_weighted(
                            z_hml3d, text_hml3d, len_hml3d // 4,
                            style_latents=raw_video_latents,
                            style_weight_schedule=w_schedule,
                            detach_pass2=True,
                            focus_ratio=args.cycle_focus_ratio
                        )

                        # Step 2: Decode styled prediction and compute style loss
                        motion_styled_pred = dae.decode(pred_styled)
                        cross_style_loss = compute_style_loss(motion_styled_pred, motion_style, style_classifier)

                        # Step 3: Combine losses
                        cross_loss = args.cycle_loss_weight * cross_cycle_loss + args.cross_style_loss_weight * cross_style_loss

                        epoch_loss_cross_cycle += _finite_item(cross_cycle_loss)
                        epoch_loss_cross_style += _finite_item(cross_style_loss)

                    # Add cross-batch loss to total.
                    # NaN guard: the x0 estimate used by the cycle loss is reconstructed from the predicted
                    # noise and can overflow at near-final diffusion timesteps (tiny alpha_bar). Drop the
                    # cross-batch term for this iteration instead of propagating NaN into the update.
                    if torch.isfinite(cross_loss):
                        loss_total += args.cross_batch_weight * cross_loss
                        cross_batch_count += 1

                        # Logging
                        logs['cross_cycle'] += _finite_item(cross_cycle_loss)
                        logs['cross_content'] += _finite_item(cross_content_loss)
                        logs['cross_style'] += _finite_item(cross_style_loss)
                    else:
                        skipped_cross += 1
                        print(f"[NaN-guard] it {it}: non-finite cross-batch loss "
                              f"(cycle={_finite_item(cross_cycle_loss) or 'nan'}) - term skipped")

                # =============================================================
                # BACKPROPAGATION
                # =============================================================
                # NaN guard: never apply a non-finite update (it would corrupt weights and the EMA).
                if not torch.isfinite(loss_total):
                    skipped_steps += 1
                    print(f"[NaN-guard] it {it}: non-finite total loss - optimizer step skipped")
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    continue
                loss_total.backward()

                gn_dec = torch.nn.utils.clip_grad_norm_(dae.decoder.parameters(), max_norm=1.0)
                gn_mardm = torch.nn.utils.clip_grad_norm_(mardm.parameters(), max_norm=1.0)
                if not (torch.isfinite(gn_dec) and torch.isfinite(gn_mardm)):
                    skipped_steps += 1
                    print(f"[NaN-guard] it {it}: non-finite gradient norm - optimizer step skipped")
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    continue

                optimizer.step()
                scheduler.step()

                # Logging
                epoch_loss_total += loss_total.item()
                epoch_loss_base += loss_base.item()
                epoch_loss_style += loss_style.item()
                epoch_loss_base_mse += recon_loss_base.item()
                epoch_loss_style_mse += recon_loss_style.item()
                epoch_loss_style_ce += style_feat_loss.item()
                epoch_loss_content_base += content_loss_base.item()
                epoch_loss_content_style += content_loss_style.item()
                
                logs['loss'] += loss_total.item()
                logs['loss_base'] += loss_base.item()
                logs['loss_style'] += loss_style.item()
                logs['recon_base'] += recon_loss_base.item()
                logs['recon_style'] += recon_loss_style.item()
                logs['style_feat_loss'] += style_feat_loss.item()  
                logs['lr'] += optimizer.param_groups[-1]['lr']

                update_ema(mardm, ema_mardm,
                           ema_decay_warmup(it, schedule['ema_warmup_iters'],
                                            max_decay=schedule['ema_max_decay']))

                # Progress bar
                postfix_dict = {
                    'total': f"{loss_total.item():.4f}",
                    'base': f"{loss_base.item():.4f}",
                    'style': f"{loss_style.item():.4f}",
                    'recon_b': f"{recon_loss_base.item():.4f}",
                    'style_feat': f"{style_feat_loss.item():.4f}",
                    'content_b': f"{content_loss_base.item():.4f}",
                }
                if args.enable_cross_batch and cross_batch_count > 0:
                    postfix_dict['xbatch'] = f"{cross_batch_count}"
                    if 'cycle' in args.cross_batch_mode or 'hybrid' in args.cross_batch_mode:
                        postfix_dict['cycle'] = f"{cross_cycle_loss.item():.4f}"
                t.set_postfix(postfix_dict)

                if it % log_interval == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        logger.add_scalar('Train/%s' % tag, value / log_interval, it)
                        mean_loss[tag] = value / log_interval
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

        # Surgical EMA reset on style-only params (base EMA preserved).
        if (args.ema_style_reset_every_epochs > 0
                and epoch > 0
                and epoch % args.ema_style_reset_every_epochs == 0):
            n_reset = reset_ema_style_params(ema_mardm, mardm)
            print(f"[ema-reset] epoch {epoch}: copied {n_reset} style params "
                  f"from online → EMA (base EMA preserved)")

        # if epoch == 0 or epoch == checkpoint['ep']+1 or epoch % args.save_every_epochs == 0:  # NameError without --is_continue
        if epoch == 0 or (resume_ep is not None and epoch == resume_ep + 1) or epoch % args.save_every_epochs == 0:
            savediff(pjoin(model_dir, f'epoch_{epoch}.tar'), epoch, mardm, optimizer, scheduler, it, 'mardm', args, ema_mardm=ema_mardm, weight_schedule=w_schedule)

            frozen_ok, decoder_ok, report = verify_dae_weights(dae, original_frozen_weights, original_decoder_weights, device)
            print(f"DAE Weight Integrity Check (Epoch {epoch}):")
            print(report)

            if not frozen_ok:
                print("⚠️  WARNING: Frozen components changed! This should not happen.")
            if not decoder_ok and epoch > 0:
                print("⚠️  WARNING: Decoder weights unchanged! Check if gradients are flowing.")

            save_upd(pjoin(model_dir, f'dae_epoch_{epoch}.tar'), epoch, dae, optimizer, scheduler, it, 'ae')
    
            print(f"MARDM checkpoint saved at epoch {epoch} at location {pjoin(model_dir, f'epoch_{epoch}.tar')}")
            print(f"DAE checkpoint saved at epoch {epoch} at location {pjoin(model_dir, f'dae_epoch_{epoch}.tar')}")

        # =================================================================
        # VALIDATION LOOP
        # =================================================================
        print('\nValidation time:')
        mardm.eval()
        dae.eval()

        val_epoch_total = 0.0
        val_epoch_base = 0.0
        val_epoch_style = 0.0
        val_mse_hml3d = 0.0
        val_mse_style = 0.0
        val_ce_style = 0.0 
        val_content_loss_hml3d = 0.0
        val_content_loss_style = 0.0
        
        val_cross_cycle = 0.0
        val_cross_content = 0.0
        val_cross_style = 0.0
        val_cross_count = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch} | Validation") as t:
                for i, batch_data in enumerate(t):
                    
                    z_hml3d = batch_data['latent_humanml'].to(device)
                    motion_hml3d = batch_data['motion_humanml'].float().to(device)
                    len_hml3d = batch_data['length_humanml'].to(device)
                    text_hml3d = batch_data['text_humanml']
                    
                    val_loss_base, full_pred_base, mask_base = mardm.forward_loss(z_hml3d, text_hml3d, len_hml3d, raw_style_latents=None)
                    motion_pred_base = dae.decode(full_pred_base)
                    recon_loss_base = F.smooth_l1_loss(motion_pred_base, motion_hml3d)
                    content_loss_base = compute_content_loss(motion_pred_base, len_hml3d * 4, text_hml3d, motion_clip)
                    
                    motion_style = batch_data['motion_styled'].float().to(device)
                    len_style = batch_data['length_styled'].to(device) // 4
                    text_style = batch_data['text_styled']
                    
                    inputs = processor(batch_data['video_styled'], return_tensors="pt").to(device)
                    vid_tensors = vmodel(**inputs).last_hidden_state
                    z_style, raw_video_latents = dae.encode(motion_style, vid_tensors)
                    
                    val_loss_style, full_pred_style, mask_style = mardm.forward_loss(
                        z_style, text_style, len_style, 
                        raw_style_latents=raw_video_latents, 
                        style_weight_schedule=w_schedule
                    )

                    motion_pred_style = dae.decode(full_pred_style)
                    recon_loss_style = F.smooth_l1_loss(motion_pred_style, motion_style)
                    style_feat_loss = compute_style_loss(motion_pred_style, motion_style, style_classifier)
                    content_loss_style = compute_content_loss(motion_pred_style, len_style * 4, text_style, motion_clip)

                    # === Cross-batch validation ===
                    if args.enable_cross_batch and i < 5:
                        
                        if args.cross_batch_mode == 'direct':
                            _, val_pred_cross, _ = mardm.forward_loss(
                                z_hml3d, text_hml3d, len_hml3d // 4,
                                raw_style_latents=raw_video_latents,
                                style_weight_schedule=w_schedule
                            )
                            val_motion_cross = dae.decode(val_pred_cross)
                            _, val_cross_content_loss, val_cross_style_loss = compute_cross_batch_direct(
                                val_motion_cross, motion_style, text_hml3d, len_hml3d,
                                style_classifier, motion_clip
                            )
                            val_cross_content += val_cross_content_loss.item()
                            val_cross_style += val_cross_style_loss.item()
                        
                        elif args.cross_batch_mode == 'style_only':
                            _, val_pred_cross, _ = mardm.forward_loss(
                                z_hml3d, text_hml3d, len_hml3d // 4,
                                raw_style_latents=raw_video_latents,
                                style_weight_schedule=w_schedule
                            )
                            val_motion_cross = dae.decode(val_pred_cross)
                            val_cross_style_loss = compute_cross_batch_style_only(
                                val_motion_cross, motion_style, style_classifier
                            )
                            val_cross_style += val_cross_style_loss.item()
                        
                        elif args.cross_batch_mode == 'latent_cycle':
                            val_cycle_loss, _, _, _ = mardm.forward_cycle_loss_masked_only(
                                z_hml3d, text_hml3d, len_hml3d // 4,
                                style_latents=raw_video_latents,
                                style_weight_schedule=w_schedule
                            )
                            val_cross_cycle += val_cycle_loss.item()
                        
                        elif args.cross_batch_mode == 'latent_cycle_weighted':
                            val_cycle_loss, _, _, _ = mardm.forward_cycle_loss_weighted(
                                z_hml3d, text_hml3d, len_hml3d // 4,
                                style_latents=raw_video_latents,
                                style_weight_schedule=w_schedule,
                                focus_ratio=args.cycle_focus_ratio
                            )
                            val_cross_cycle += val_cycle_loss.item()
                        
                        elif args.cross_batch_mode in ['hybrid', 'hybrid_weighted']:
                            if args.cross_batch_mode == 'hybrid':
                                val_cycle_loss, val_pred_styled, _, _ = mardm.forward_cycle_loss_masked_only(
                                    z_hml3d, text_hml3d, len_hml3d // 4,
                                    style_latents=raw_video_latents,
                                    style_weight_schedule=w_schedule
                                )
                            else:
                                val_cycle_loss, val_pred_styled, _, _ = mardm.forward_cycle_loss_weighted(
                                    z_hml3d, text_hml3d, len_hml3d // 4,
                                    style_latents=raw_video_latents,
                                    style_weight_schedule=w_schedule,
                                    focus_ratio=args.cycle_focus_ratio
                                )
                            val_motion_styled = dae.decode(val_pred_styled)
                            val_style_loss = compute_style_loss(val_motion_styled, motion_style, style_classifier)
                            val_cross_cycle += val_cycle_loss.item()
                            val_cross_style += val_style_loss.item()
                        
                        val_cross_count += 1

                    val_epoch_base += val_loss_base.item()
                    val_epoch_style += val_loss_style.item()
                    val_epoch_total += (val_loss_base.item() + val_loss_style.item()) + args.recons_weight*(recon_loss_style + recon_loss_base) + (args.style_weight * style_feat_loss) + (args.content_weight * (content_loss_base + 0.01 * content_loss_style))   
                    val_mse_hml3d += recon_loss_base.item()
                    val_mse_style += recon_loss_style.item()
                    val_ce_style += style_feat_loss.item()
                    val_content_loss_hml3d += content_loss_base.item()
                    val_content_loss_style += content_loss_style.item()

                    t.set_postfix({
                        'v_base': f"{val_loss_base.item():.4f}",
                        'v_style': f"{val_loss_style.item():.4f}",
                        'recons_base': f"{recon_loss_base.item():.4f}",
                        'recons_style': f"{recon_loss_style.item():.4f}",
                    })

                    train_metrics = {
                            'train/train_loss': epoch_loss_total/ len(train_loader),
                            'train/recon_loss_base': epoch_loss_base/ len(train_loader),
                            'train/recon_loss_style': epoch_loss_style/ len(train_loader),
                            'train/recon_loss_base_mse': epoch_loss_base_mse/ len(train_loader),
                            'train/recon_loss_style_mse': epoch_loss_style_mse/ len(train_loader),
                            'train/style_feat_loss': epoch_loss_style_ce/ len(train_loader),
                            'train/content_loss_base': epoch_loss_content_base/ len(train_loader),
                            'train/content_loss_style': epoch_loss_content_style/ len(train_loader),
                            'train/cross_cycle_loss': epoch_loss_cross_cycle / cross_batch_count if cross_batch_count > 0 else 0,
                            'train/cross_content_loss': epoch_loss_cross_content / cross_batch_count if cross_batch_count > 0 else 0,
                            'train/cross_style_loss': epoch_loss_cross_style / cross_batch_count if cross_batch_count > 0 else 0,
                            'train/cross_batch_count': cross_batch_count,
                            }
                            
            
            
                    val_metrics = {
                            'val/recon_loss': val_epoch_base / len(val_loader),
                            'val/style_recon_loss': val_epoch_style / len(val_loader),
                            'val/total_loss': val_epoch_total / len(val_loader),
                            'val/mse_hml3d': val_mse_hml3d / len(val_loader),
                            'val/mse_style': val_mse_style / len(val_loader),
                            'val/style_feat_loss': val_ce_style / len(val_loader),
                            'val/content_loss_hml3d': val_content_loss_hml3d / len(val_loader),
                            'val/content_loss_style': val_content_loss_style / len(val_loader),
                            'val/cross_cycle_loss': val_cross_cycle / val_cross_count if val_cross_count > 0 else 0,
                            'val/cross_content_loss': val_cross_content / val_cross_count if val_cross_count > 0 else 0,
                            'val/cross_style_loss': val_cross_style / val_cross_count if val_cross_count > 0 else 0,
                            'val/cross_batch_count': val_cross_count,
                            }
                            
            

                    # wandb.log({**train_metrics, **val_metrics, 'epoch': epoch-1}, step=it)


        avg_val_loss = val_epoch_total / len(val_loader)
        
        # Print training summary
        print(f"\nTraining loss: {epoch_loss_total/len(train_loader):.5f} | Base: {epoch_loss_base/len(train_loader):.5f} | Style: {epoch_loss_style/len(train_loader):.5f} | Recon Base MSE: {epoch_loss_base_mse/len(train_loader):.5f} | Recon Style MSE: {epoch_loss_style_mse/len(train_loader):.5f} | Style CE Loss: {epoch_loss_style_ce/len(train_loader):.5f} | Content Base Loss: {epoch_loss_content_base/len(train_loader):.5f} | Content Style Loss: {epoch_loss_content_style/len(train_loader):.5f}")
        
        # Print cross-batch training summary
        if skipped_cross or skipped_steps:
            print(f"[NaN-guard] epoch {epoch}: {skipped_cross} cross-batch term(s) and {skipped_steps} step(s) skipped")
        if args.enable_cross_batch and cross_batch_count > 0:
            avg_cross_cycle = epoch_loss_cross_cycle / cross_batch_count if cross_batch_count > 0 else 0
            avg_cross_content = epoch_loss_cross_content / cross_batch_count if cross_batch_count > 0 else 0
            avg_cross_style = epoch_loss_cross_style / cross_batch_count if cross_batch_count > 0 else 0
            
            print(f"Cross-Batch Training [{args.cross_batch_mode}]: Count={cross_batch_count}", end="")
            if 'cycle' in args.cross_batch_mode or 'hybrid' in args.cross_batch_mode:
                print(f" | Cycle={avg_cross_cycle:.5f}", end="")
            if args.cross_batch_mode == 'direct':
                print(f" | Content={avg_cross_content:.5f}", end="")
            if args.cross_batch_mode in ['direct', 'style_only', 'hybrid', 'hybrid_weighted']:
                print(f" | Style={avg_cross_style:.5f}", end="")
            print()
        
        # Print validation summary
        print(f"Validation loss: {avg_val_loss:.5f} | Base: {val_epoch_base/len(val_loader):.5f} | Style: {val_epoch_style/len(val_loader):.5f} | Recon Base MSE: {val_mse_hml3d/len(val_loader):.5f} | Recon Style MSE: {val_mse_style/len(val_loader):.5f} | Style CE Loss: {val_ce_style/len(val_loader):.5f} | Content Base Loss: {val_content_loss_hml3d/len(val_loader):.5f} | Content Style Loss: {val_content_loss_style/len(val_loader):.5f}")
        
        # Print cross-batch validation summary
        if args.enable_cross_batch and val_cross_count > 0:
            avg_val_cross_cycle = val_cross_cycle / val_cross_count
            avg_val_cross_content = val_cross_content / val_cross_count
            avg_val_cross_style = val_cross_style / val_cross_count
            
            print(f"Cross-Batch Validation [{args.cross_batch_mode}]: Count={val_cross_count}", end="")
            if 'cycle' in args.cross_batch_mode or 'hybrid' in args.cross_batch_mode:
                print(f" | Cycle={avg_val_cross_cycle:.5f}", end="")
            if args.cross_batch_mode == 'direct':
                print(f" | Content={avg_val_cross_content:.5f}", end="")
            if args.cross_batch_mode in ['direct', 'style_only', 'hybrid', 'hybrid_weighted']:
                print(f" | Style={avg_val_cross_style:.5f}", end="")
            print()

        if avg_val_loss < worst_loss:
            print(f"Improved validation loss from {worst_loss:.5f} to {avg_val_loss:.5f}!!!")
            worst_loss = avg_val_loss

        if epoch % args.fid_eval_every == 0:
            fid_styled = compute_styled_fid(ema_mardm, dae, val_loader, eval_wrapper,
                                            vmodel, processor, w_schedule, args, device)
            print(f"[FID] epoch {epoch}: styled={fid_styled:.4f} (best={best_fid_styled:.4f})")
            logger.add_scalar('./Val/FID_styled', fid_styled, epoch)

            if fid_styled < best_fid_styled:
                print(f"  Styled FID improved: {best_fid_styled:.4f} -> {fid_styled:.4f}")
                best_fid_styled = fid_styled
                savediff(pjoin(model_dir, 'net_best_fid_styled.tar'),
                         epoch, mardm, optimizer, scheduler, it, 'mardm', args,
                         ema_mardm=ema_mardm, weight_schedule=w_schedule)

        epoch += 1

    # Save final
    final_path = pjoin(model_dir, 'final.tar')
    savediff(final_path, epoch-1, mardm, optimizer, scheduler, it, 'mardm', args, ema_mardm=ema_mardm, weight_schedule=w_schedule)

    frozen_ok, decoder_ok, report = verify_dae_weights(dae, original_frozen_weights, original_decoder_weights, device)
    print(f"DAE Weight Integrity Check (Epoch {epoch}):")
    print(report)

    if not frozen_ok:
        print("⚠️  WARNING: Frozen components changed! This should not happen.")
    if not decoder_ok and epoch > 0:
        print("⚠️  WARNING: Decoder weights unchanged! Check if gradients are flowing.")

    save_upd(pjoin(model_dir, 'dae_final.tar'), epoch-1, dae, optimizer, scheduler, it, 'ae')
    
    print(f"Final DAE checkpoint saved at location {pjoin(model_dir, 'dae_final.tar')}")
    print(f"Final MARDM checkpoint saved at location {final_path}")
    
    delete_old_checkpoints(model_dir)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time / 60:.2f} minutes") 

    # wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MARDM')
    # parser.add_argument('--ae_name', type=str, default="AE")
    # parser.add_argument('--ae_model', type=str, default='AE_Model')
    # parser.add_argument('--model', type=str, default='MARDM-mld', choices=['MARDM-DDPM-XL', 'MARDM-SiT-XL', 'MARDM-MLD'])
    parser.add_argument('--ae_name', type=str, default='DAE')
    parser.add_argument('--ae_model', type=str, default='DAE_Model')
    parser.add_argument('--model', type=str, default='MARDM-DDPM-XL', choices=['MARDM-DDPM-XL', 'MARDM-SiT-XL'])
    parser.add_argument('--dae_ckpt', type=str, default=None,
                        help='Stage-1 DualAE checkpoint. Default: <checkpoints_dir>/100styles/<ae_name>/epoch_119_detach_nostyle_disc.tar')
    parser.add_argument('--style_classifier_ckpt', type=str, default=None,
                        help='Frozen kinematic style classifier. Default: <checkpoints_dir>/style_classifier/style_classifier_final.pt')
    parser.add_argument('--style_cls_nclasses', type=int, default=None,
                        help='Classifier head size. Default: inferred from the checkpoint (released classifier = 21).')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument("--max_motion_length", type=int, default=196)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--window_size', default=64, type=int)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--ema_style_reset_every_epochs', default=0, type=int,
                        help="Copy online->EMA for style-only params every N epochs. 0 disables.")
    parser.add_argument('--save_every_epochs', type=int, default=10,
                        help="Save epoch_{N}.tar and dae_epoch_{N}.tar every N epochs.")
    parser.add_argument('--fid_eval_every', type=int, default=1,
                        help="Run styled FID every N epochs.")
    parser.add_argument('--fid_timesteps', type=int, default=18,
                        help="Diffusion timesteps used for FID generation.")
    parser.add_argument('--fid_cond_scale', type=float, default=3.0,
                        help="Classifier-free guidance scale used for FID generation.")
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument("--seed", type=int, default=3407)
    # parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0 if os.name == 'nt' else 4,
                        help='DataLoader workers (default 0 on Windows, where worker processes can deadlock)')
    parser.add_argument('--is_continue', action="store_true",
                        help='Initialise from the pretrained HumanML3D MARDM (<checkpoints_dir>/t2m/<model>/model/humanml3d_latest.tar). '
                             'REQUIRED for the thesis setup (Sec. 4.4); --epoch counts on top of its epoch counter.')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--styles', type=str, nargs='+', default=["Aeroplane", "Chicken", "Robot", "Superman", "ArmsFolded"])
    parser.add_argument('--video_encoder', type=str, default='vivit', choices=['vivit', 'timesformer', 'xclip'])
    parser.add_argument('--exp_name', type=str, default='mardm_debug', help="Name for WandB experiment tracking.")
    
    # === EXPERIMENTAL MATRIX ARGS ===
    parser.add_argument('--data_mode', type=str, default='v4', choices=['v4', 'v5'])
    parser.add_argument('--style_routing', type=str, default='diffmlp', choices=['diffmlp', 'mart'])
    parser.add_argument('--freeze_mode', type=str, default='differential', choices=['strict', 'differential', 'none'])
    parser.add_argument('--enable_cfg_dropout', action='store_true')
    # parser.add_argument('--use_weight_schedule', action='store_true')  # original: OFF unless passed
    parser.add_argument('--use_weight_schedule', action=argparse.BooleanOptionalAction, default=True,
                        help='Block-wise linear style weight w in [0,1] across DiffMLP blocks (thesis Eq. 4.11-4.13). '
                             'ON by default; disable with --no-use_weight_schedule.')
    parser.add_argument('--mart_style_weight', type=float, default=1.0)
    parser.add_argument('--recons_weight', type=float, default=1.0)
    parser.add_argument('--style_weight', type=float, default=1.0)
    parser.add_argument('--mardm_lr_mult', type=float, default=0.1)
    parser.add_argument('--content_weight', type=float, default=0.5)

    # === PHASE 3: CROSS-BATCH ARGS ===
    parser.add_argument('--enable_cross_batch', action='store_true', help="Enable Phase 3 cross-batch training")
    parser.add_argument('--cross_batch_mode', type=str, default='direct', choices=['direct', 'style_only', 'latent_cycle', 'latent_cycle_weighted', 'hybrid', 'hybrid_weighted'], help="Cross-batch mode")
    parser.add_argument('--cross_batch_prob', type=float, default=0.4, help="Probability of running cross-batch per iteration")
    parser.add_argument('--cross_batch_weight', type=float, default=0.1, help="Weight for cross-batch losses")
    parser.add_argument('--cycle_loss_weight', type=float, default=1.0, help="Weight for cycle loss in hybrid modes")
    parser.add_argument('--cross_style_loss_weight', type=float, default=0.3, help="Weight for style verification in hybrid modes")
    parser.add_argument('--cycle_focus_ratio', type=float, default=0.9, help="Focus ratio for weighted cycle loss (weight on masked positions)")

    # === STYLE CLASSIFIER ARGS ===
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--ff_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    main(args)