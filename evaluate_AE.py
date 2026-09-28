import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoImageProcessor, VivitModel, TimesformerModel, VivitImageProcessor, AutoProcessor


from models.AE import DAE_models, DAE_models_full, DAED_models, DVAE_models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from utils.evaluators import Evaluators
from utils.datasets import AEDataset, AEVideoDataset_100styles, collate_fn, video_collate_fn, AEVideoDataset_100styles_v2, AEVideoDataset_100styles_v3
from models.LengthEstimator import LengthEstimator
from utils.motion_process import recover_from_ric, kit_kinematic_chain, t2m_kinematic_chain, plot_3d_motion_gif, plot_3d_motion_side_by_side, plot_3d_motion_three_way

import time
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss
from utils.eval_utils import evaluation_ae, calculate_activation_statistics, calculate_frechet_distance, calculate_rr_mpjpe, calculate_global_mpjpe
import argparse

def _flatten_time(embeds: np.ndarray) -> np.ndarray:
    bsz = embeds.shape[0]
    return embeds.reshape(bsz, -1)

def _pool_time_mean(embeds: np.ndarray) -> np.ndarray:
    return embeds.mean(axis=-1)


def temporal_interpolate_joints(joints: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or joints.shape[0] < 2:
        return joints
    joints_tensor = torch.from_numpy(joints).permute(2, 1, 0).reshape(1, -1, joints.shape[0]).float()
    upsampled = F.interpolate(joints_tensor, scale_factor=factor, mode='linear', align_corners=True)
    upsampled = upsampled.reshape(3, joints.shape[1], -1).permute(2, 1, 0).contiguous()
    return upsampled.numpy()

def visualize_video_embeddings(args, all_video_embeddings_vid, all_video_embeddings_motion, all_labels, style_names, flatten_time = True):
    """
    Visualises the video embeddings pre- and post-using PCA.
    """
   
    # --- Visualize with PCA ---
    if all_video_embeddings_vid.shape[0] < 2 or all_video_embeddings_motion.shape[0] < 2:
        print("Not enough data to generate a PCA plot.")
        return
    
    if args.video_encoder=='xclip':
        n_samples = all_video_embeddings_vid.shape[0]

        all_embeddings_vid_2d = all_video_embeddings_vid.reshape(n_samples, -1)
        all_embeddings_motion_2d = all_video_embeddings_motion.reshape(n_samples, -1)

        pca = PCA(n_components=2, random_state=0)
        embeddings_2d_vid = pca.fit_transform(all_embeddings_vid_2d)
        embeddings_2d_motion = pca.fit_transform(all_embeddings_motion_2d)
    
    else:
        pca = PCA(n_components=2, random_state=0)
        embeddings_2d_vid = pca.fit_transform(all_video_embeddings_vid)
        embeddings_2d_motion = pca.fit_transform(all_video_embeddings_motion)

    
    # Map numeric labels back to style names for plotting
    # labels_np = np.asarray(all_labels, dtype=np.int64).reshape(-1)
    style_map = {idx: name for idx, name in enumerate(style_names)}
    hue_labels = [style_map[l] for l in all_labels]

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    
    # Plot for pre-projection embeddings
    sns.scatterplot(
        ax=axes[0],
        x=embeddings_2d_vid[:, 0], 
        y=embeddings_2d_motion[:, 1], 
        hue=hue_labels, 
        alpha=0.9, 
        s=30, 
        palette='Set1'
    )
    axes[0].set_title('Projected video embeddings')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(title='Style')
    axes[0].grid(True)

    # Plot for post-projection embeddings
    sns.scatterplot(
        ax=axes[1],
        x=embeddings_2d_motion[:, 0], 
        y=embeddings_2d_motion[:, 1], 
        hue=hue_labels, 
        alpha=0.9, 
        s=30, 
        palette='Set1'
    )
    axes[1].set_title('Projected motion embeddings')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend(title='Style')
    axes[1].grid(True)

    fig.suptitle(f'Lower-Dim Video Projections by Style ({args.video_encoder}) - PCA')


    save_path = os.path.join(args.viz_dir, f"video_embeddings_{args.video_encoder}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"PCA plots saved to {save_path}")



def main(args):   
    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    
    if args.is_interp:
        interp_factor = max(1, args.interp_factor)
    else:
        interp_factor = 1

    if args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        joints_num = 22
        dim_pose = 67
    elif args.dataset_name == "100styles":
        data_root = f'{args.dataset_dir}/100STYLE-SMPL/'
        joints_num = 22
        dim_pose = 67
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        joints_num = 21
        dim_pose = 64
    print(f'Loading data from {data_root} with {joints_num} joints and {dim_pose} pose dimension')
    

    motion_dir = pjoin(data_root, 'new_joint_vecs')
    video_dir = pjoin(data_root, 'videos')
    text_dir = pjoin(data_root, 'texts')
    max_motion_length = 196
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    dict_file = pjoin(data_root, '100STYLE_name_dict_length.txt')

    # val_split_file = pjoin(data_root, 'test_100STYLE_Full.txt')
    val_split_file = pjoin(data_root, 'test_100STYLE_Filter.txt')

     
    # val_dataset_full = AEVideoDataset_100styles(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file)
    # val_dataset_full = AEVideoDataset_100styles(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file, snippets_per_sequence=args.snippets_per_sequence)
    # val_dataset_full = AEVideoDataset_100styles_v2(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file)
    # val_dataset_full = AEVideoDataset_100styles_v3(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file, snippets_per_sequence=args.snippets_per_sequence)
    val_dataset_full = AEVideoDataset_100styles_v3(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file, snippets_per_sequence=args.snippets_per_sequence)


    # val_size = (len(val_dataset_full) *2)// 3
    val_size = len(val_dataset_full) // 2


    test_size = len(val_dataset_full) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_dataset_full, 
        [val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed) # for reproducibility
    )

    print(f"length of full val dataset: {len(val_dataset_full)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    # print(f'Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}')

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            shuffle=True, pin_memory=True, collate_fn=video_collate_fn)
    
    eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True, collate_fn=video_collate_fn)
    
    
    print(f'Val batches: {len(val_loader)}, Eval batches: {len(eval_loader)}')





    #################################################################################
    #                                      Models                                   #
    #################################################################################
    print(f'Building model: {args.model}, checkpoints will be loaded from {args.checkpoints_dir}')
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name)


    num_classes = len(args.style_classes) if args.style_classes else 100 # Default to 100 if not specified

    if args.train_style == 'motion':
        dae = DAE_models_full[args.model](window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose, is_classification=True)
        style_classifier = DAE_models_full['Video arm'](num_classes=num_classes, encoder_name=args.video_encoder)  # Assuming 100 styles for classification
        if args.styleconv_dir is not None:
            if args.video_encoder == 'vivit':
                print(f"Loading style classifier from {args.styleconv_dir}")
                scc_path = pjoin(args.checkpoints_dir, args.styleconv_dir, args.video_encoder + '_adapter_final.pth')
            elif args.video_encoder == 'timesformer':
                print(f"Loading style classifier from {args.styleconv_dir}")
                scc_path = pjoin(args.checkpoints_dir, args.styleconv_dir, 'tsf_conv_final.pth')

            if os.path.exists(scc_path):
                checkpoint = torch.load(scc_path, map_location='cpu')
                style_classifier.load_state_dict(checkpoint)
            else:
                print(f"Warning: Checkpoint not found at {scc_path}")

        style_classifier.eval()

        pc_dae = sum(param.numel() for param in dae.parameters())
        print('Total parameters of all models: {}M'.format(pc_dae / 1000_000))
        
    elif args.train_style == 'full':
        print("NOTE: Training from scratch. Not loading custom style classifier weights.")
        # dae = DAE_models[args.model](window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose, is_classification=True)
        if args.is_var:
            dae = DVAE_models[args.model](window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose, is_classification=True)
        else:
            dae = DAED_models[args.model](window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose, is_classification=True)


        pc_dae = sum(param.numel() for param in dae.parameters())
        # pc_style = sum(p.numel() for p in style_classifier.parameters() if p.requires_grad)
        print('Total parameters - DAE: {:.3f}M'.format(pc_dae / 1_000_000))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_idx)
        print(f"Using device: {device} ({dev_name})")
    else:
        print(f"Using device: {device} (CPU)")

    MODEL_CONFIG = {
        'vivit': {"name": "google/vivit-b-16x2-kinetics400", "processor": "google/vivit-b-16x2-kinetics400", "num_frames": 32},
        'timesformer': {"name": "facebook/timesformer-base-finetuned-k400", "processor": "MCG-NJU/videomae-base", "num_frames": 32},
        'xclip': {"name": "microsoft/xclip-base-patch32", "processor": "microsoft/xclip-base-patch32", "num_frames": 32},
    }

    if args.video_encoder == 'vivit':
        processor = VivitImageProcessor.from_pretrained(MODEL_CONFIG['vivit']['processor'])
        vmodel = VivitModel.from_pretrained(MODEL_CONFIG['vivit']['name']).to(device)
    elif args.video_encoder == 'timesformer':
        processor = AutoProcessor.from_pretrained(MODEL_CONFIG['timesformer']['processor'])
        vmodel = TimesformerModel.from_pretrained(MODEL_CONFIG['timesformer']['name']).to(device)
    


    #################################################################################
    #                                    Eval setup                                 #
    #################################################################################
    print('Eval Loop:')
    result_dir = args.viz_dir
    os.makedirs(result_dir, exist_ok=True)

    # Define loss functions and logger
    criterion = torch.nn.SmoothL1Loss()
    style_loss_fn = torch.nn.CrossEntropyLoss()
    logger = SummaryWriter(log_dir=model_dir)

    if args.train_style == 'full':
        dae.to(device)
    if args.train_style == 'motion':
        dae.to(device)
        style_classifier.to(device)

    epoch = 0
    it = 0
    
    if args.is_continue:
        # 1. Determine the filename suffix based on encoder type
        # Mapping: 'vivit' -> 'vivit', 'timesformer' -> 'tsf'
        enc_name = 'tsf' if args.video_encoder == 'timesformer' else args.video_encoder
        
        # 2. Construct the full model path dynamically
        # filename = f'epoch_119_{enc_name}_{args.train_style}.tar'
        # filename = f'epoch_119_{enc_name}_sp.tar'
        filename = f'epoch_119_v0m1_detach_disc_nostyle_neutral.tar'

        model_path = pjoin(model_dir, filename)

        print(f"Loading checkpoint from {model_path}")

        # 3. Load the checkpoint (One single call)
        checkpoint = torch.load(model_path, map_location=device)

        # 4. Load common components
        dae.load_state_dict(checkpoint['ae'])
        # optimizer.load_state_dict(checkpoint['opt_ae'])
        # scheduler.load_state_dict(checkpoint['scheduler'])

        # 5. Load conditional components (Only for 'full' style)
        # if args.train_style == 'full':
        #     style_classifier.load_state_dict(checkpoint['style_classifier'])

        # 6. Extract metadata and print
        epoch = checkpoint.get('ep', 0)
        it = checkpoint.get('total_it', 0)
        print(f"Load model epoch:{epoch} iterations:{it}")
    
           
    kinematic_chain = kit_kinematic_chain if args.dataset_name == 'kit' else t2m_kinematic_chain

    #################################################################################
    #                                   Evaluation                                  #
    #################################################################################
    print('Evaluation time:')
    dae.eval()
    # if args.train_style == 'motion':
    #     style_classifier.eval()
    
    eval_loss_rec_m = []
    eval_loss_rec_v = []
    eval_loss_vel_m = []
    eval_loss_vel_v = []
    eval_loss_embed = []
    eval_loss_style_m = []
    eval_loss_style_v = []
    if args.is_var:
        eval_loss_klm = []
        eval_loss_klv = []

    eval_loss = []
    eval_correct_vid = 0
    eval_correct_motion = 0
    eval_correct_both = 0

    # Initialize FID and MPJPE tracking variables
    motion_embeddings_gt = []
    motion_embeddings_pred_motion = []
    motion_embeddings_pred_video = []
    mpjpe_results = []
    motion_ids = []
    
    # Initialize evaluator for FID computation
    eval_wrapper = Evaluators(args.dataset_name, device=device)


    
    ################################################################### Debug code ###################################################################
    # --- ADD VISUALIZATION 2 (Setup) ---
    all_video_embeddings_vid = []
    all_video_embeddings_motion = []
    all_labels = []
    # --- ADD VISUALIZATION 2 (Setup) ---
    ################################################################### Debug code ###################################################################
    
    with torch.no_grad():
        with tqdm(eval_loader, desc=f"Epoch {epoch} | Evaluation") as t:
            for i, batch_data in enumerate(t):
                motions, videos, labels = batch_data
                motions = motions.to(device).float()
                labels = labels.to(device).long() # CrossEntropyLoss expects long type for labels

                inputs = processor(videos, return_tensors="pt").to(device)
                with torch.no_grad():
                    vid_tensors = vmodel(**inputs)
                    vid_tensors = vid_tensors.last_hidden_state  # (batch_size, num_frames, hidden_size)

                output_dict = dae.forward_eval(motions, vid_tensors)  

                if args.is_var:
                    recons_motion = output_dict['recons_motion']
                    recons_video = output_dict['recons_video']
                    style_logits_vid = output_dict['video_logits']
                    style_logits_motion = output_dict['motion_logits']
                    m_latent = output_dict['motion_latent']
                    v_latent = output_dict['video_latent']
                    m_logvar = output_dict['motion_logvar']
                    v_logvar = output_dict['video_logvar']

                    loss_klm = -0.5 * torch.sum(1 + m_logvar - m_latent.pow(2) - m_logvar.exp())
                    loss_klm = loss_klm / motions.shape[0]

                    loss_klv = -0.5 * torch.sum(1 + v_logvar - v_latent.pow(2) - v_logvar.exp())
                    loss_klv = loss_klv / motions.shape[0]

                    loss_rec_m = criterion(recons_motion, motions)
                    loss_rec_v = criterion(recons_video, motions)

                    pred_local_pos_m = recons_motion[..., 4: (joints_num - 1) * 3 + 4] 
                    local_pos_m = motions[..., 4: (joints_num - 1) * 3 + 4]
                    loss_explicit_m = criterion(pred_local_pos_m, local_pos_m)

                    pred_local_pos_v = recons_video[..., 4: (joints_num - 1) * 3 + 4] 
                    local_pos_v = motions[..., 4: (joints_num - 1) * 3 + 4]
                    loss_explicit_v = criterion(pred_local_pos_v, local_pos_v)

                    if args.train_style == 'motion':
                            loss_embedding = criterion(m_latent, (v_latent.to(device).detach()))
                    elif args.train_style == 'full':
                            loss_embedding = criterion(m_latent, (v_latent.to(device)))

                    style_loss_vid = style_loss_fn(style_logits_vid, labels)  
                    style_loss_motion = style_loss_fn(style_logits_motion, labels)                   

                    loss = 0.5*(loss_rec_m + loss_rec_v) + \
                           args.aux_loss_joints * (0.5*(loss_explicit_v + loss_explicit_m)) + \
                           args.style_loss_multiplier * (0.5*(style_loss_motion + style_loss_vid)) + \
                           args.embed_loss_multiplier * loss_embedding + \
                           args.kl_loss_multiplier * (0.5*(loss_klm + loss_klv))

                else:
                    recons_motion = output_dict['recons_motion']
                    recons_video = output_dict['recons_video']
                    m_latent = output_dict['motion_latent']
                    v_latent = output_dict['video_latent']
                    style_logits_vid = output_dict['video_logits']
                    style_logits_motion = output_dict['motion_logits']

                    loss_rec_m = criterion(recons_motion, motions)
                    loss_rec_v = criterion(recons_video, motions)

                    pred_local_pos_m = recons_motion[..., 4: (joints_num - 1) * 3 + 4] 
                    local_pos_m = motions[..., 4: (joints_num - 1) * 3 + 4]
                    loss_explicit_m = criterion(pred_local_pos_m, local_pos_m)

                    pred_local_pos_v = recons_video[..., 4: (joints_num - 1) * 3 + 4] 
                    local_pos_v = motions[..., 4: (joints_num - 1) * 3 + 4]
                    loss_explicit_v = criterion(pred_local_pos_v, local_pos_v)

                    if args.train_style == 'motion':
                            loss_embedding = criterion(m_latent, (v_latent.to(device).detach()))
                    elif args.train_style == 'full':
                            loss_embedding = criterion(m_latent, (v_latent.to(device)))

                    style_loss_vid = style_loss_fn(style_logits_vid, labels)  
                    style_loss_motion = style_loss_fn(style_logits_motion, labels)                   

                    loss = 0.5*(loss_rec_m + loss_rec_v) + \
                           args.aux_loss_joints * (0.5*(loss_explicit_v + loss_explicit_m)) + \
                           args.style_loss_multiplier * (0.5*(style_loss_motion + style_loss_vid)) + \
                           args.embed_loss_multiplier * loss_embedding

        
                pred_motion_np = recons_motion.detach().cpu().numpy()
                pred_vid_motion_np = recons_video.detach().cpu().numpy()
                motion_np = motions.detach().cpu().numpy()

                pred_data = pred_motion_np * std[:67] + mean[:67]
                pred_vid_data = pred_vid_motion_np * std[:67] + mean[:67]
                gt_data = motion_np * std[:67] + mean[:67]
            
                sub_dir = pjoin(result_dir, str(i))
                os.makedirs(sub_dir, exist_ok=True)

                # Get motion embeddings for FID computation
                m_lens = torch.tensor([motions.shape[1]] * motions.shape[0], device=device)
                
                # Ground truth embeddings
                gt_embeddings, _ = eval_wrapper.get_motion_embeddings(motions, m_lens)
                
                # Motion-based reconstruction embeddings
                pred_motion_tensor = torch.from_numpy(pred_motion_np).to(device).float()
                pred_embeddings_motion, _ = eval_wrapper.get_motion_embeddings(pred_motion_tensor, m_lens)
                
                # Video-based reconstruction embeddings
                pred_video_tensor = torch.from_numpy(pred_vid_motion_np).to(device).float()
                pred_embeddings_video, _ = eval_wrapper.get_motion_embeddings(pred_video_tensor, m_lens)
                
                # Store embeddings for batch-wise FID computation
                motion_embeddings_gt.append(gt_embeddings.cpu().numpy())
                motion_embeddings_pred_motion.append(pred_embeddings_motion.cpu().numpy())
                motion_embeddings_pred_video.append(pred_embeddings_video.cpu().numpy())

                ################################################################### Debug code ###################################################################
                # --- ADD VISUALIZATION 2 (Collect) ---
                all_video_embeddings_vid.append(style_logits_vid.detach().cpu().numpy())
                all_video_embeddings_motion.append(style_logits_motion.detach().cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                # --- ADD VISUALIZATION 2 (Collect) ---
                ################################################################### Debug code ###################################################################


                for k, (pred, pred_vid, gt) in enumerate(zip(pred_data, pred_vid_data , gt_data)):
                    # Generate unique motion ID for this sample
                    motion_id = f"batch_{i}_sample_{k}"
                    motion_ids.append(motion_id)

                    pred_joints = recover_from_ric(torch.from_numpy(pred).float(), joints_num).numpy()
                    pred_joints = temporal_interpolate_joints(pred_joints, interp_factor)

                    gt_joints = recover_from_ric(torch.from_numpy(gt).float(), joints_num).numpy()
                    gt_joints = temporal_interpolate_joints(gt_joints, interp_factor)

                    # Compute MPJPE for this sample
                    gt_tensor = torch.from_numpy(gt_joints).float()
                    pred_motion_tensor = torch.from_numpy(pred_joints).float()
                    rr_mpjpe_motion = calculate_rr_mpjpe(gt_tensor, pred_motion_tensor).mean().item()
                    global_mpjpe_motion = calculate_global_mpjpe(gt_tensor, pred_motion_tensor).mean().item()                  
                    
                    # Save numpy arrays
                    np.save(pjoin(sub_dir, f"pred_motion_features_{k}.npy"), pred)
                    np.save(pjoin(sub_dir, f"pred_motion_rep_{k}.npy"), pred_joints)
                                       
                    np.save(pjoin(sub_dir, f"gt_motion_features_{k}.npy"), gt)
                    np.save(pjoin(sub_dir, f"gt_motion_rep_{k}.npy"), gt_joints)

                    if args.viz_format == 'three_way':
                        pred_vid_joints = recover_from_ric(torch.from_numpy(pred_vid).float(), joints_num).numpy()
                        pred_vid_joints = temporal_interpolate_joints(pred_vid_joints, interp_factor)

                        pred_video_joints_tensor = torch.from_numpy(pred_vid_joints).float()
                        
                        np.save(pjoin(sub_dir, f"pred_video_features_{k}.npy"), pred_vid)
                        np.save(pjoin(sub_dir, f"pred_video_rep_{k}.npy"), pred_vid_joints)
                        
                        ## MPJPE calculation
                        rr_mpjpe_video = calculate_rr_mpjpe(gt_tensor, pred_video_joints_tensor).mean().item()
                        global_mpjpe_video = calculate_global_mpjpe(gt_tensor, pred_video_joints_tensor).mean().item()


                        mpjpe_results.append({
                            'motion_id': motion_id,
                            'rr_mpjpe_motion_based': rr_mpjpe_motion,
                            'rr_mpjpe_video_based': rr_mpjpe_video,
                            'global_mpjpe_motion_based': global_mpjpe_motion,
                            'global_mpjpe_video_based': global_mpjpe_video,
                        })
                       
                        # Create three-way comparison GIF
                        print("Generating visualizations")
                        comparison_save_path = pjoin(sub_dir, f"comparison_motion_{k}.gif")
                        plot_3d_motion_three_way(comparison_save_path, kinematic_chain, 
                                            gt_joints, pred_joints, pred_vid_joints, 
                                            "Ground Truth", "Motion Reconstruction", "Video Reconstruction", 
                                            fps=20)

                        print(f"Three-way comparison gif saved to {comparison_save_path}")

                    else:
                        ## MPJPE calculation
                        mpjpe_results.append({
                            'motion_id': motion_id,
                            'rr_mpjpe': rr_mpjpe_motion,
                            'global_mpjpe': global_mpjpe_motion,
                        })

                        # Create side-by-side comparison GIF
                        print("Generating visualizations")
                        comparison_save_path = pjoin(sub_dir, f"comparison_motion_{k}.gif")
                        plot_3d_motion_side_by_side(comparison_save_path, kinematic_chain, 
                                                gt_joints, pred_joints, 
                                                "Ground Truth", "Motion Reconstruction", 
                                                fps=20)
                        print(f"Comparison gif saved to {comparison_save_path}")

                        
                eval_loss.append(loss.item())
                eval_loss_rec_m.append(loss_rec_m.item())
                eval_loss_rec_v.append(loss_rec_v.item())
                eval_loss_vel_m.append(loss_explicit_m.item())
                eval_loss_vel_v.append(loss_explicit_v.item())
                eval_loss_embed.append(loss_embedding.item())
                eval_loss_style_v.append(style_loss_vid.item())
                eval_loss_style_m.append(style_loss_motion.item())

                if args.is_var:
                    eval_loss_klm.append(loss_klm.item())
                    eval_loss_klv.append(loss_klv.item())

                pred_vid = torch.argmax(style_logits_vid, dim=1)
                pred_motion = torch.argmax(style_logits_motion, dim=1)
                
                eval_correct_vid += (pred_vid == labels).sum().item()
                eval_correct_motion += (pred_motion == labels).sum().item()
                eval_correct_both += ((pred_vid == labels) & (pred_motion == labels)).sum().item()

    if args.is_var:
        eval_metrics = {
                'eval/loss': sum(eval_loss) / len(eval_loss),
                'eval/loss_rec_motion': sum(eval_loss_rec_m) / len(eval_loss_rec_m),
                'eval/loss_rec_video': sum(eval_loss_rec_v) / len(eval_loss_rec_v),

                'eval/loss_vel_motion': sum(eval_loss_vel_m) / len(eval_loss_vel_m),
                'eval/loss_vel_video': sum(eval_loss_vel_v) / len(eval_loss_vel_v),
                'eval/loss_embed': sum(eval_loss_embed) / len(eval_loss_embed),   

                'eval/loss_kl_motion': sum(eval_loss_klm) / len(eval_loss_klm),
                'eval/loss_kl_video': sum(eval_loss_klv) / len(eval_loss_klv),

                'eval/loss_style_motion': sum(eval_loss_style_v) / len(eval_loss_style_v),
                'eval/loss_style_video': sum(eval_loss_style_m) / len(eval_loss_style_m),
                
                'eval/accuracy_video': eval_correct_vid / len(test_dataset),
                'eval/accuracy_motion': eval_correct_motion / len(test_dataset),
                'eval/accuracy_both': eval_correct_both / len(test_dataset)
        }
    
    else:
         eval_metrics = {
                'eval/loss': sum(eval_loss) / len(eval_loss),
                'eval/loss_rec_motion': sum(eval_loss_rec_m) / len(eval_loss_rec_m),
                'eval/loss_rec_video': sum(eval_loss_rec_v) / len(eval_loss_rec_v),

                'eval/loss_vel_motion': sum(eval_loss_vel_m) / len(eval_loss_vel_m),
                'eval/loss_vel_video': sum(eval_loss_vel_v) / len(eval_loss_vel_v),
                'eval/loss_embed': sum(eval_loss_embed) / len(eval_loss_embed),   

                'eval/loss_style_motion': sum(eval_loss_style_v) / len(eval_loss_style_v),
                'eval/loss_style_video': sum(eval_loss_style_m) / len(eval_loss_style_m),
                
                'eval/accuracy_video': eval_correct_vid / len(test_dataset),
                'eval/accuracy_motion': eval_correct_motion / len(test_dataset),
                'eval/accuracy_both': eval_correct_both / len(test_dataset)
        }


    ################################################################### Debug code ####################################################################
    # --- ADD VISUALIZATION 2 (Plot) ---
    # Concatenate all collected data from the epoch
    all_video_embeddings_vid = np.concatenate(all_video_embeddings_vid, axis=0)
    all_video_embeddings_motion = np.concatenate(all_video_embeddings_motion, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    visualize_video_embeddings(args, all_video_embeddings_vid, all_video_embeddings_motion, all_labels, args.style_classes)
    # visualize_video_embeddings(args, all_video_embeddings_pre, all_labels, args.style_classes)
    # --- END VISUALIZATION 2 ---
    ################################################################### Debug code ###################################################################


    print("Computing FID metrics...")
    gt_embeddings_all = np.concatenate(motion_embeddings_gt, axis=0)
    pred_motion_embeddings_all = np.concatenate(motion_embeddings_pred_motion, axis=0)
    pred_video_embeddings_all = np.concatenate(motion_embeddings_pred_video, axis=0)

    # Compute activation statistics
    gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings_all)
    pred_motion_mu, pred_motion_cov = calculate_activation_statistics(pred_motion_embeddings_all)
    pred_video_mu, pred_video_cov = calculate_activation_statistics(pred_video_embeddings_all)
    
    # Compute FID scores
    fid_motion_based = calculate_frechet_distance(gt_mu, gt_cov, pred_motion_mu, pred_motion_cov)

    if args.viz_format == 'three_way':
        fid_video_based = calculate_frechet_distance(gt_mu, gt_cov, pred_video_mu, pred_video_cov)

        avg_rr_mpjpe_motion = np.mean([result['rr_mpjpe_motion_based'] for result in mpjpe_results])
        avg_global_mpjpe_motion = np.mean([result['global_mpjpe_motion_based'] for result in mpjpe_results])
        
        avg_rr_mpjpe_video = np.mean([result['rr_mpjpe_video_based'] for result in mpjpe_results])
        avg_global_mpjpe_video = np.mean([result['global_mpjpe_video_based'] for result in mpjpe_results])

        evaluation_results = {
            'dataset_info': {
                'dataset_name': args.dataset_name,
                'num_samples': len(mpjpe_results),
                'joints_num': joints_num,
                'dim_pose': dim_pose
            },
            'reconstruction_metrics': {
                'rr_mpjpe_motion_avg': float(avg_rr_mpjpe_motion),
                'rr_mpjpe_video_avg': float(avg_rr_mpjpe_video),
                'rr_mpjpe_difference': float(avg_rr_mpjpe_motion - avg_rr_mpjpe_video),
                'global_mpjpe_motion_avg': float(avg_global_mpjpe_motion),
                'global_mpjpe_video_avg': float(avg_global_mpjpe_video),
                'global_mpjpe_difference': float(avg_global_mpjpe_motion - avg_global_mpjpe_video),                
                'fid_motion_based': float(fid_motion_based),
                'fid_video_based': float(fid_video_based),
                'fid_difference': float(fid_motion_based - fid_video_based)
            }
        }
        
    else:
        rr_avg_mpjpe = np.mean([result['mpjpe'] for result in mpjpe_results])
        global_avg_mpjpe = np.mean([result['global_mpjpe'] for result in mpjpe_results])

        evaluation_results = {
            'dataset_info': {
                'dataset_name': args.dataset_name,
                'num_samples': len(mpjpe_results),
                'joints_num': joints_num,
                'dim_pose': dim_pose
            },
            'reconstruction_metrics': {
                'mpjpe_avg': float(rr_avg_mpjpe),
                'global_mpjpe_avg': float(global_avg_mpjpe),
                'fid': float(fid_motion_based)
            }
        }
        

    results_file = pjoin(result_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"Evaluation results saved to {results_file}")
                       

        
        

    for tag, value in eval_metrics.items():
        logger.add_scalar(tag.replace('eval', 'Eval'), value, epoch)

    if args.is_var:
        print(
        'Evaluation Loss: %.5f, Rec Video: %.5f, Rec Motion: %.5f, Velocity video: %.5f, Velocity motion: %.5f, Embedding: %.5f, Style Video: %.5f, Style Motion: %.5f, KL loss Motion: %.5f, KL loss Video: %.5f, Acc Video: %.5f, Acc Motion: %.5f, Acc Both: %.5f'
        % (    
            eval_metrics['eval/loss'],
            eval_metrics['eval/loss_rec_motion'],
            eval_metrics['eval/loss_rec_video'],

            eval_metrics['eval/loss_vel_motion'],
            eval_metrics['eval/loss_vel_video'],
            eval_metrics['eval/loss_embed'],

            eval_metrics['eval/loss_style_video'],
            eval_metrics['eval/loss_style_motion'],

            eval_metrics['eval/loss_kl_motion'],
            eval_metrics['eval/loss_kl_video'],

            eval_metrics['eval/accuracy_video'],
            eval_metrics['eval/accuracy_motion'],
            eval_metrics['eval/accuracy_both']
        )
    )
    else:    
        print(
            'Evaluation Loss: %.5f, Rec Video: %.5f, Rec Motion: %.5f, Velocity video: %.5f, Velocity motion: %.5f, Embedding: %.5f, Style Video: %.5f, Style Motion: %.5f, Acc Video: %.5f, Acc Motion: %.5f, Acc Both: %.5f'
            % (    
                eval_metrics['eval/loss'],
                eval_metrics['eval/loss_rec_motion'],
                eval_metrics['eval/loss_rec_video'],

                eval_metrics['eval/loss_vel_motion'],
                eval_metrics['eval/loss_vel_video'],
                eval_metrics['eval/loss_embed'],

                eval_metrics['eval/loss_style_video'],
                eval_metrics['eval/loss_style_motion'],

                eval_metrics['eval/accuracy_video'],
                eval_metrics['eval/accuracy_motion'],
                eval_metrics['eval/accuracy_both']
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE')
    parser.add_argument('--model', type=str, default='AE_Model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, choices=['t2m','100styles'])
    parser.add_argument("--style_classes", type=str, nargs='+', default=["Aeroplane", "Chicken", "Robot", "Superman", "ArmsFolded","Neutral"], help="List of style names (subdirectory names).")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--aux_loss_joints', type=float, default=1)
    parser.add_argument('--style_loss_multiplier', type=float, default=1)
    parser.add_argument('--embed_loss_multiplier', type=float, default=1)
    parser.add_argument('--kl_loss_multiplier', type=float, default=0.1)
    parser.add_argument('--recons_loss', type=str, default='l1_smooth')
    parser.add_argument('--device', default=0, type=int, help="GPU device ID.")
    parser.add_argument("--seed", type=int, default=3407)
    # parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0 if os.name == 'nt' else 4,
                        help='DataLoader workers (default 0 on Windows, where worker processes can deadlock)')
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--styleconv_dir', type=str, default=None)
    parser.add_argument("--video_encoder", type=str, default='vivit', choices=['vivit', 'timesformer', 'xclip'], help="Video encoder backbone to use.")
    parser.add_argument('--snippets_per_sequence', default=1, type=int)
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--exp_name', type=str, default='MMAE_eval')
    parser.add_argument('--viz_dir',type=str, default='./visualizations')
    parser.add_argument('--viz_format', type=str, default='three_way', choices=['side_by_side', 'three_way'])
    parser.add_argument('--train_style', type=str, default='full', choices=['full', 'motion', 'video'], help="Train full arch or parts only.")
    parser.add_argument('--interp_factor', type=int, default=4, help="Temporal interpolation factor (>=1) for joint sequences before evaluation/visualization.")
    parser.add_argument('--is_interp', action="store_true", help="If set, perform temporal interpolation before evaluation/visualization.")
    parser.add_argument('--is_var', action="store_true", help="If set, call the variational version of the AE.")


    arg = parser.parse_args()
    main(arg)