import os
from os.path import join as pjoin
from pathlib import Path
import torch
import numpy as np
import random
from tqdm import tqdm
# import wandb
import sklearn 
import copy 
import glob

import re
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, VivitModel, TimesformerModel, VivitImageProcessor, AutoProcessor

# from models.AE import AE_models, DAE_models, DAE_models_full
from models.AE import DAE_models, DAE_models_full, DAED_models  


from utils.evaluators import Evaluators
from utils.datasets import AEDataset, AEVideoDataset_100styles, collate_fn, video_collate_fn, AEVideoDataset_100styles_v2, AEVideoDataset_100styles_v3, AEVideoStyleDataset, load_and_split_data
from utils.profiling import set_profile_enabled, set_profile_log_file, enable_stdout_log, timer

import time
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, save_upd, save_disc_upd, print_current_loss
from utils.eval_utils import evaluation_ae
import argparse

################################### Physics Loss Constants ###################################
FOOT_JOINT_INDICES = {
    'l_ankle_y': 4 + 6*3 + 1,   # dim 23
    'r_ankle_y': 4 + 7*3 + 1,   # dim 26
    'l_foot_y':  4 + 9*3 + 1,   # dim 32
    'r_foot_y':  4 + 10*3 + 1,  # dim 35
}
FOOT_JOINT_XZ = {
    'l_ankle': (4 + 6*3, 4 + 6*3 + 2),    # (x, z) = (22, 24)
    'r_ankle': (4 + 7*3, 4 + 7*3 + 2),    # (25, 27)
    'l_foot':  (4 + 9*3, 4 + 9*3 + 2),    # (31, 33)
    'r_foot':  (4 + 10*3, 4 + 10*3 + 2),  # (34, 36)
}
ROOT_HEIGHT_IDX = 3



#################################################################################
#                                   Visualization                               #
#################################################################################
def _flatten_time(embeds: np.ndarray) -> np.ndarray:
    bsz = embeds.shape[0]
    return embeds.reshape(bsz, -1)

def _pool_time_mean(embeds: np.ndarray) -> np.ndarray:
    return embeds.mean(axis=-1)

def visualize_video_embeddings(args, all_video_embeddings, all_labels, style_names, epoch, flatten_time = True):
    """
    Visualises the video embeddings pre- and post-using PCA.
    """
   
    # --- Visualize with PCA ---
    if all_video_embeddings.shape[0] < 2:
        print("Not enough data to generate a PCA plot.")
        return
    

    if all_video_embeddings.ndim == 2:
        pca = PCA(n_components=2, random_state=0)
        embeddings_2d = pca.fit_transform(all_video_embeddings)
        plt_title = f"2D Video latent Embeddings by Style at #{epoch} using ({args.model} encoder)"
    
    elif all_video_embeddings.ndim ==3:
        if flatten_time:
            all_video_embeddings = _flatten_time(all_video_embeddings)
        else:
            all_video_embeddings = _pool_time_mean(all_video_embeddings)

        pca = PCA(n_components=2, random_state=0)    
        embeddings_2d = pca.fit_transform(all_video_embeddings)
        plt_title = f"2D Video latent Unprojected Embeddings by Style #{epoch} using ({args.model} encoder)"

    style_map = {idx: name for idx, name in enumerate(style_names)}
    hue_labels = [style_map[l] for l in all_labels]

    plt.figure(figsize=(12, 10))
        
    # --- KEY CHANGE: Added palette='viridis' to match the other plot ---
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1], 
        hue=hue_labels, 
        alpha=0.9, 
        s=30, 
        palette='Set1'  # This ensures color consistency
    )
    
    plt.title(plt_title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Style')
    plt.grid(True)



    fname = re.sub(r'[^\w\-]+', '_', plt_title).strip('_') + ".png"
    save_path = os.path.join(args.viz_dir, fname)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"PCA plot saved to {save_path}")

# def delete_old_checkpoints(model_dir, patterns=("epoch_*.tar")):
# ^ was a plain string: iterating it yields '*' and glob(model_dir/*) deleted final.tar as well.
def delete_old_checkpoints(model_dir, patterns=("epoch_*.tar",)):
    for pat in patterns:
        for path in glob.glob(pjoin(model_dir, pat)):
            try:
                os.remove(path)
            except OSError as e:
                print(f"[Cleanup] Could not remove {path}: {e}")



def main(args):
    #################################################################################
    #                                   Monitoring                                  #
    #################################################################################
    
    
    # wandb.init(
    #     project='Multimodal MARDM Testing', 
    #     name=args.exp_name,
    #     config=args,
    # )

    # wandb.init(
    #     project='Multimodal MARDM Debugging and Analysis', 
    #     name=args.exp_name,
    #     config=args,
    # )

    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    
    torch.backends.cudnn.benchmark = False
    # os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # Enable/disable profiling prints

    set_profile_enabled(getattr(args, "profile", False))
    if getattr(args, "profile_log", None):
        enable_stdout_log(args.profile_log)          # tee all stdout/stderr to file
        set_profile_log_file(args.profile_log)       # also append timing lines explicitly

    # set_profile_enabled(getattr(args, 'profile', False))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    
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
    # dict_file = pjoin(data_root, '100STYLE_name_dict.txt')
    dict_file = pjoin(data_root, '100STYLE_name_dict_length.txt')

    # train_data, val_data, test_data = load_and_split_data(
    #     root_path=data_root,
    #     style_classes=args.style_classes
    # )

    train_split_file = pjoin(data_root, 'train_100STYLE_Full.txt')
    # train_split_file = pjoin(data_root, 'train_100STYLE_Full.txt')

    val_split_file = pjoin(data_root, 'test_100STYLE_Filter.txt')
    
    ## Original code
    # train_split_file = pjoin(data_root, 'train.txt')
    # val_split_file = pjoin(data_root, 'val.txt')

    # train_dataset = AEDataset(mean, std, motion_dir, args.window_size, train_split_file, dim_pose=dim_pose)
    # val_dataset = AEDataset(mean, std, motion_dir, args.window_size, val_split_file, dim_pose=dim_pose)


    # # 100STYLES unimodal code
    # train_dataset = AEDataset_100styles(mean, std, motion_dir, args.window_size, train_split_file, dim_pose=dim_pose, dict_file=dict_file)
    # val_dataset = AEDataset_100styles(mean, std, motion_dir, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file)


    # val_dataset_full = AEDataset_100styles(mean, std, motion_dir, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file)

    # train_dataset = AEVideoDataset_100styles(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, train_split_file, dim_pose=dim_pose, dict_file=dict_file)
    # train_dataset = AEVideoDataset_100styles_v2(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, train_split_file, dim_pose=dim_pose, dict_file=dict_file)
    train_dataset = AEVideoDataset_100styles_v3(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, train_split_file, dim_pose=dim_pose, dict_file=dict_file, snippets_per_sequence=args.snippets_per_sequence)
    
    # train_dataset = AEVideoStyleDataset(train_data, mean=mean, std=std, dim_pose=dim_pose, num_frames_to_extract=args.window_size)
    # val_dataset = AEVideoStyleDataset(val_data, mean=mean, std=std, dim_pose=dim_pose, num_frames_to_extract=args.window_size)
    # test_dataset = AEVideoStyleDataset(test_data, mean=mean, std=std, dim_pose=dim_pose, num_frames_to_extract=args.window_size)


 
    # val_dataset = AEVideoDataset_100styles(mean, std, motion_dir, video_dir, args.style_classes,args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file)


    # val_dataset_full = AEVideoDataset_100styles(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file)
    # val_dataset_full = AEVideoDataset_100styles_v2(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file)
    val_dataset_full = AEVideoDataset_100styles_v3(mean, std, motion_dir, video_dir, args.style_classes, args.window_size, val_split_file, dim_pose=dim_pose, dict_file=dict_file, snippets_per_sequence=args.snippets_per_sequence)


    val_size = len(val_dataset_full)*2 // 3
    test_size = len(val_dataset_full) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_dataset_full, 
        [val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed) # for reproducibility
    )

    print(f"length of full val dataset: {len(val_dataset_full)}, val: {len(val_dataset)}, test: {len(test_dataset)}, train: {len(train_dataset)}")
    # print(f"length of train dataset: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}, train: {len(train_dataset)}")





    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # train_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, \
    # ^ trained on the held-out test split (bug introduced 2025-12-19, after the released DualAE was trained)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, \
                              shuffle=True, pin_memory=True, collate_fn=video_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, \
                            shuffle=True, pin_memory=True, collate_fn=video_collate_fn)
    
    # eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            #  shuffle=False, pin_memory=True, collate_fn=video_collate_fn)
    
    
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')



    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)
    print(f'Building model: {args.model}, checkpoints will be saved to {model_dir}')


    num_classes = len(args.style_classes) if args.style_classes else 100 # Default to 100 if not specified
    print(f"Number of style classes: {num_classes}")

    # ae = AE_models[args.model](num_style_classes=num_classes, input_width=dim_pose)
   
    # style_classifier = DAE_models_full['Video arm'](num_classes=num_classes, encoder_name=args.video_encoder)  # Assuming 100 styles for classification
    
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
        dae = DAED_models[args.model](window_size=args.window_size, num_style_classes=num_classes, input_width=dim_pose, is_classification=True)
        discriminator = DAED_models['Discriminator'](input_width=dim_pose)


        pc_dae = sum(param.numel() for param in dae.parameters())
        # pc_style = sum(p.numel() for p in style_classifier.parameters() if p.requires_grad)
        print('Total parameters - AE: {:.3f}M'.format(pc_dae / 1_000_000))

        # style_classifier.eval()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_idx)
        print(f"Using device: {device} ({dev_name})")
    else:
        print(f"Using device: {device} (CPU)")

    # eval_wrapper = Evaluators(args.dataset_name, device=device)
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
    #                                    Training Loop                              #
    #################################################################################
    print('Training Loop:')
    logger = SummaryWriter(model_dir)

    style_loss_fn = torch.nn.CrossEntropyLoss()
    if args.recons_loss == 'l1_smooth':
        criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = torch.nn.MSELoss()

    dae.to(device)
    discriminator.to(device)

    if args.train_style == 'motion':
        style_classifier.to(device).eval()
        optimizer = optim.AdamW(dae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    elif args.train_style == 'full':
        optimizer = optim.AdamW(dae.parameters(),  lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

        optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
        criterion_gan = torch.nn.BCELoss() # Binary Cross Entropy

        # optimizer = optim.AdamW([{'params': ae.parameters(), 'lr': 1e-4},
        #                         {'params': style_classifier.get_trainable_parameters(), 'lr': 1e-5}
        #                     ], weight_decay=args.weight_decay)


    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    epoch = 0
    it = 0
    # if args.is_continue:
    #     model_dir = pjoin(model_dir, 'latest.tar')
    #     checkpoint = torch.load(model_dir, map_location=device)
    #     ae.load_state_dict(checkpoint['ae'])
    #     optimizer.load_state_dict(checkpoint[f'opt_ae'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     epoch, it = checkpoint['ep'], checkpoint['total_it']
    #     print("Load model epoch:%d iterations:%d" % (epoch, it))
    
    if args.is_continue:
        # 1. Determine the filename suffix based on encoder type
        # Mapping: 'vivit' -> 'vivit', 'timesformer' -> 'tsf'
        enc_name = 'tsf' if args.video_encoder == 'timesformer' else args.video_encoder
        
        # 2. Construct the full model path dynamically
        filename = f'epoch_119_{enc_name}_{args.train_style}.tar'
        model_path = pjoin(model_dir, filename)

        print(f"Loading checkpoint from {model_path}")

        # 3. Load the checkpoint (One single call)
        checkpoint = torch.load(model_path, map_location=device)

        # 4. Load common components
        dae.load_state_dict(checkpoint['ae'])
        optimizer.load_state_dict(checkpoint['opt_ae'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        if 'discriminator' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator'])
            optimizer_disc.load_state_dict(checkpoint['opt_disc'])
            print("Discriminator loaded successfully.")

        # # 5. Load conditional components (Only for 'full' style)
        # if args.train_style == 'full':
        #     style_classifier.load_state_dict(checkpoint['style_classifier'])

        # 6. Extract metadata and print
        epoch = checkpoint.get('ep', 0)
        it = checkpoint.get('total_it', 0)
        print(f"Load model epoch:{epoch} iterations:{it}")

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

    current_lr = args.lr
    logs = defaultdict(def_value, OrderedDict())

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = 1000, 0, 0, 0, 0, 100, 100

    # checkpoint1 = args.epoch / 3
    # checkpoint2 = args.epoch*2 / 3

    while epoch < args.epoch:
        dae.train()
        
        epoch_train_loss = 0.0
        epoch_train_rec_loss = 0.0
        epoch_train_vel_loss = 0.0
        epoch_train_style_loss = 0.0
        epoch_train_embed_loss = 0.0
        epoch_train_adv_loss = 0.0
        epoch_train_disc_loss = 0.0
        epoch_train_cycle_loss = 0.0
        epoch_train_foot_sink_loss = 0.0
        epoch_train_foot_slide_loss = 0.0
        epoch_train_correct = 0

        print("=== Using randomized mixed training strategy: 50% w/ video projection, 50% w/ motion projection===")
        with tqdm(train_loader, desc=f"Epoch {epoch} | Training") as t:
            for i, batch_data in enumerate(t):
                it += 1
                if it < args.warm_up_iter:
                    current_lr = update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

                # -----------------------------------------------------------
                # PHASE CHECK
                # -----------------------------------------------------------
                use_discriminator = (epoch >= args.disc_start_epoch)
                use_cycle = (epoch >= args.cycle_start_epoch)
                use_physics = (epoch >= args.cycle_start_epoch)  # Enable physics losses alongside cycle

                motions, videos, labels = batch_data
                motions = motions.to(device).float()
                labels = labels.to(device).long() # CrossEntropyLoss expects long type for labels

                # -----------------------------------------------------------
                # 1. Forward Pass (Generator)
                # -----------------------------------------------------------

                inputs = processor(videos, return_tensors="pt").to(device)
                with torch.no_grad():
                    vid_tensors = vmodel(**inputs)
                    vid_tensors = vid_tensors.last_hidden_state  # (batch_size, num_frames, hidden_size)
                
                output_dict = dae(motions, vid_tensors)

                recons = output_dict['recons']
                m_latent = output_dict['motion_latent']
                v_latent = output_dict['video_latent']
                style_logits = output_dict['style_logits']

                # -----------------------------------------------------------
                # 2. Train Discriminator (Phase 2+)
                # -----------------------------------------------------------


                loss_disc = torch.tensor(0.0, device=device) # Default 0
                
                if use_discriminator:
                    optimizer_disc.zero_grad()
                    
                    # Real Data
                    real_scores = discriminator(motions)
                    real_labels = torch.ones_like(real_scores)
                    loss_d_real = criterion_gan(real_scores, real_labels)
                    
                    # Fake Data (Detached)
                    fake_scores = discriminator(recons.detach())
                    fake_labels = torch.zeros_like(fake_scores)
                    loss_d_fake = criterion_gan(fake_scores, fake_labels)
                    
                    loss_disc = (loss_d_real + loss_d_fake) / 2
                    loss_disc.backward()
                    optimizer_disc.step()

                # -----------------------------------------------------------
                # 3. Train Generator (DAE)
                # -----------------------------------------------------------
                
                optimizer.zero_grad()

                ## A. Standard Losses (Phase 1+)
                loss_rec = criterion(recons, motions)
                pred_local_pos = recons[..., 4: (joints_num - 1) * 3 + 4]
                local_pos = motions[..., 4: (joints_num - 1) * 3 + 4]
                loss_explicit = criterion(pred_local_pos, local_pos)
                
                if args.train_style == 'motion':
                    loss_embedding = criterion(m_latent.to(device).detach(), (v_latent.to(device).detach()))
                elif args.train_style == 'full':
                    loss_embedding = criterion(m_latent.to(device).detach(), (v_latent.to(device)))
                style_loss = style_loss_fn(style_logits, labels)                   

                loss = loss_rec + args.aux_loss_joints * loss_explicit + args.style_loss_multiplier * style_loss + args.embed_loss_multiplier * loss_embedding

                ## B. Adversarial Loss (Phase 2+)
                loss_adv = torch.tensor(0.0, device=device)
                if use_discriminator:
                    # We want the generator to fool the discriminator
                    pred_for_gen = discriminator(recons) # No detach!
                    target_for_gen = torch.ones_like(pred_for_gen)
                    loss_adv = criterion_gan(pred_for_gen, target_for_gen)
                    
                    loss += args.adv_loss_multiplier * loss_adv

                ## C. Cycle Consistency Loss (Phase 3+)

                loss_cycle = torch.tensor(0.0, device=device)
                if use_cycle:
                    # 1. Get the 'Real' Latent (Target)
                    # Note: We detach this because we don't want to update the encoder to match the recons.
                    # We want the decoder to produce recons that match the encoder.
                    z_real = output_dict['motion_latent'].detach() 
                    
                    # 2. Re-Encode the Reconstruction (The Cycle)
                    z_synth = dae.encode_motion(recons)
                    
                    # 3. Calculate Loss (MSE between latents)
                    loss_cycle = torch.nn.functional.mse_loss(z_synth, z_real)
                    
                    loss += args.cycle_loss_multiplier * loss_cycle

                ## D. Physical Losses (Foot Sliding + Foot Sinking)

                loss_foot_sink = torch.tensor(0.0, device=device)
                loss_foot_slide = torch.tensor(0.0, device=device)
                if use_physics:  # Phase 3+: enable physics losses alongside cycle
                    recons_bdt = recons.permute(0, 2, 1)  # [B, T, D] -> [B, D, T]

                    # Foot Sinking: penalize foot joints going below ground
                    root_h = recons_bdt[:, ROOT_HEIGHT_IDX, :]  # [B, T]
                    sink_losses = []
                    for name, dim_y in FOOT_JOINT_INDICES.items():
                        local_y = recons_bdt[:, dim_y, :]
                        global_y = root_h + local_y
                        penetration = F.relu(-global_y)  # penalize below floor (y=0)
                        sink_losses.append(penetration)
                    loss_foot_sink = torch.stack(sink_losses, dim=0).mean()

                    # Foot Sliding: penalize horizontal velocity when foot is near ground
                    slide_losses = []
                    for name, (dim_x, dim_z) in FOOT_JOINT_XZ.items():
                        dim_y = FOOT_JOINT_INDICES.get(name + '_y')
                        if dim_y is None:
                            continue
                        global_y = root_h + recons_bdt[:, dim_y, :]  # [B, T]
                        vel_x = recons_bdt[:, dim_x, 1:] - recons_bdt[:, dim_x, :-1]
                        vel_z = recons_bdt[:, dim_z, 1:] - recons_bdt[:, dim_z, :-1]
                        horiz_speed = vel_x ** 2 + vel_z ** 2  # [B, T-1]
                        near_ground = (global_y < args.foot_height_thresh).float()
                        contact = near_ground[:, :-1] * near_ground[:, 1:]  # [B, T-1]
                        slide_losses.append(horiz_speed * contact)
                    loss_foot_slide = torch.stack(slide_losses, dim=0).mean()

                    loss += args.foot_sink_multiplier * loss_foot_sink + args.foot_slide_multiplier * loss_foot_slide

                loss.backward()
                optimizer.step()

                # -----------------------------------------------------------
                # 4. Logging
                # -----------------------------------------------------------

                if it >= args.warm_up_iter:
                    scheduler.step()

                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                logs['loss_vel'] += loss_explicit.item()
                logs['style_loss'] += style_loss.item()
                logs['embed_loss'] += loss_embedding.item()
                logs['loss_adv'] += loss_adv.item()
                logs['loss_cycle'] += loss_cycle.item()
                logs['loss_disc'] += loss_disc.item()
                logs['loss_foot_sink'] += loss_foot_sink.item()
                logs['loss_foot_slide'] += loss_foot_slide.item()
                logs['accuracy'] += (torch.argmax(style_logits, dim=1) == labels).sum().item() / labels.size(0)

                epoch_train_loss += loss.item()
                epoch_train_rec_loss += loss_rec.item()
                epoch_train_vel_loss += loss_explicit.item()
                epoch_train_style_loss += style_loss.item()
                epoch_train_embed_loss += loss_embedding.item()
                epoch_train_disc_loss += loss_disc.item()
                epoch_train_adv_loss += loss_adv.item()
                epoch_train_cycle_loss += loss_cycle.item()
                epoch_train_foot_sink_loss += loss_foot_sink.item()
                epoch_train_foot_slide_loss += loss_foot_slide.item()
                epoch_train_correct += (torch.argmax(style_logits, dim=1) == labels).sum().item()


                if it % args.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        logger.add_scalar('Train/%s' % tag, value / args.log_every, it)
                        mean_loss[tag] = value / args.log_every

                    logs = defaultdict(def_value, OrderedDict())
                    # wandb.log({'train/' + k: v for k, v in mean_loss.items()}, step=it)
                    # print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                    # break

        # save(pjoin(model_dir, 'latest.tar'), epoch, ae, optimizer, scheduler, it, 'ae')
        if epoch % 10 == 0 and epoch != args.epoch-1:
            if args.train_style == 'motion':
                save(pjoin(model_dir, f'epoch_{epoch}.tar'), epoch, dae, optimizer, scheduler, it, 'ae')
            elif args.train_style == 'full':
                # save_upd(pjoin(model_dir, f'epoch_{epoch}.tar'), epoch, dae, optimizer, scheduler, it, 'ae')
                save_disc_upd(pjoin(model_dir, f'epoch_{epoch}.tar'), epoch, dae, optimizer, scheduler, it, 'ae', discriminator=discriminator, opt_disc=optimizer_disc)
            print(f"Saved checkpoint at epoch {epoch} at location: {pjoin(model_dir, f'epoch_{epoch}.tar')}")
        
        # save(pjoin(model_dir, f'epoch_{epoch}.tar'), epoch, ae, optimizer, scheduler, it, 'ae')
        epoch += 1
        
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################

        ## Sae predicted motion here after each epoch
        print('Validation time:')
        dae.eval()
        discriminator.eval()
        if args.train_style == 'motion':
            style_classifier.eval()

        val_loss_rec_m = []
        val_loss_rec_v = []
        val_loss_vel_m = []
        val_loss_vel_v = []
        val_loss_embed = []
        val_loss_disc = []
        val_loss_adv = []

        val_loss_cycle = []
        val_loss_cycle_m = []
        val_loss_cycle_v = []
        val_loss_cycle_vm = []


        val_loss_style_v = []
        val_loss_foot_sink = []
        val_loss_foot_slide = []
        val_loss = []
        val_correct_vid = 0
        # val_correct_motion = 0
        # val_correct_both = 0
        
        # --- ADD VISUALIZATION 2 (Setup) ---
        all_video_embeddings = []
        all_labels = []
        # --- ADD VISUALIZATION 2 (Setup) ---

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch-1} | Validation") as t:
                for i, batch_data in enumerate(t):

                    motions, videos, labels = batch_data
                    motions = motions.to(device).float()
                    labels = labels.to(device).long() # CrossEntropyLoss expects long type for labels

                    inputs = processor(videos, return_tensors="pt").to(device)
                    with torch.no_grad():
                        vid_tensors = vmodel(**inputs)
                        vid_tensors = vid_tensors.last_hidden_state  # (batch_size, num_frames, hidden_size)

                    output_dict = dae.forward_eval(motions, vid_tensors)  

                    recons_motion = output_dict['recons_motion']
                    recons_video = output_dict['recons_video']
                    m_latent = output_dict['motion_latent']
                    v_latent = output_dict['video_latent']
                    style_logits_vid = output_dict['video_logits']
                    # style_logits_motion = output_dict['motion_logits']
                    
                    # -----------------------------------------------------------------------------------
                    # 1. Standard Losses
                    # -----------------------------------------------------------------------------------

                    loss_rec_m = criterion(recons_motion, motions)
                    loss_rec_v = criterion(recons_video, motions)

                    pred_local_pos_m = recons_motion[..., 4: (joints_num - 1) * 3 + 4] 
                    local_pos_m = motions[..., 4: (joints_num - 1) * 3 + 4]
                    loss_explicit_m = criterion(pred_local_pos_m, local_pos_m)

                    pred_local_pos_v = recons_video[..., 4: (joints_num - 1) * 3 + 4] 
                    local_pos_v = motions[..., 4: (joints_num - 1) * 3 + 4]
                    loss_explicit_v = criterion(pred_local_pos_v, local_pos_v)
                    
                    if args.train_style == 'motion':
                        loss_embedding = criterion(m_latent.to(device).detach(), (v_latent.to(device).detach()))
                    elif args.train_style == 'full':
                        loss_embedding = criterion(m_latent.to(device).detach(), (v_latent.to(device)))
                    
                    style_loss_vid = style_loss_fn(style_logits_vid, labels)  
                    # style_loss_motion = style_loss_fn(style_logits_motion, labels)                   

                    # loss = 0.5*(loss_rec_m + loss_rec_v) + args.aux_loss_joints * (0.5*(loss_explicit_v + loss_explicit_m)) + args.style_loss_multiplier * (0.5*(style_loss_motion + style_loss_vid)) + args.embed_loss_multiplier * loss_embedding
                    
                    # -----------------------------------------------------------
                    # 2. Cycle Consistency Loss (Validation)
                    # -----------------------------------------------------------
                    # Check if re-encoding the reconstruction yields the original motion latent
                    # We check both branches: Motion->Recon->Latent AND Video->Recon->Latent

                    # Encode the reconstructions
                    z_cycle_m = dae.encode_motion(recons_motion)
                    z_cycle_v = dae.encode_motion(recons_video)
                    
                    # Calculate MSE against the "Anchor" (Original Motion Latent)
                    loss_cycle_m_val = torch.nn.functional.mse_loss(z_cycle_m, m_latent)
                    loss_cycle_v_val = torch.nn.functional.mse_loss(z_cycle_v, v_latent)
                    loss_cycle_vm_val = torch.nn.functional.mse_loss(z_cycle_v, m_latent)

                    loss_cycle_val = (1/3) * (loss_cycle_m_val + loss_cycle_v_val + loss_cycle_vm_val)


                    # -----------------------------------------------------------
                    # 3. GAN / Adversarial Losses (Validation)
                    # -----------------------------------------------------------
                    # Real Scores

                    real_scores = discriminator(motions)
                    real_labels = torch.ones_like(real_scores)
                    
                    # Fake Scores (from Motion branch and Video branch)
                    fake_scores_m = discriminator(recons_motion)
                    fake_scores_v = discriminator(recons_video)
                    fake_labels_m = torch.zeros_like(fake_scores_m)
                    fake_labels_v = torch.zeros_like(fake_scores_v)

                    # A. Discriminator Loss (How well it detects fakes)
                    loss_d_real = criterion_gan(real_scores, real_labels)
                    loss_d_fake_m = criterion_gan(fake_scores_m, fake_labels_m)
                    loss_d_fake_v = criterion_gan(fake_scores_v, fake_labels_v)

                    loss_disc_val = (loss_d_real + 0.5 * (loss_d_fake_m + loss_d_fake_v)) / 2


                    # B. Generator/Adversarial Loss (How well it fools the discriminator)
                    # We want fake scores to approach 1.0
                    target_ones_m = torch.ones_like(fake_scores_m)
                    target_ones_v = torch.ones_like(fake_scores_v)
                    
                    loss_adv_m_val = criterion_gan(fake_scores_m, target_ones_m)
                    loss_adv_v_val = criterion_gan(fake_scores_v, target_ones_v)
                    
                    loss_adv_val = 0.5 * (loss_adv_m_val + loss_adv_v_val)

                    # -----------------------------------------------------------
                    # 4. Physical Losses (Foot Sliding + Foot Sinking)
                    # -----------------------------------------------------------
                    loss_foot_sink_val = torch.tensor(0.0, device=device)
                    loss_foot_slide_val = torch.tensor(0.0, device=device)

                    # Average over both reconstruction branches
                    for recons_branch in [recons_motion, recons_video]:
                        recons_bdt = recons_branch.permute(0, 2, 1)  # [B, T, D] -> [B, D, T]
                        root_h = recons_bdt[:, ROOT_HEIGHT_IDX, :]

                        # Foot Sinking
                        sink_losses = []
                        for name, dim_y in FOOT_JOINT_INDICES.items():
                            local_y = recons_bdt[:, dim_y, :]
                            global_y = root_h + local_y
                            penetration = F.relu(-global_y)
                            sink_losses.append(penetration)
                        loss_foot_sink_val += torch.stack(sink_losses, dim=0).mean()

                        # Foot Sliding
                        slide_losses = []
                        for name, (dim_x, dim_z) in FOOT_JOINT_XZ.items():
                            dim_y = FOOT_JOINT_INDICES.get(name + '_y')
                            if dim_y is None:
                                continue
                            global_y = root_h + recons_bdt[:, dim_y, :]
                            vel_x = recons_bdt[:, dim_x, 1:] - recons_bdt[:, dim_x, :-1]
                            vel_z = recons_bdt[:, dim_z, 1:] - recons_bdt[:, dim_z, :-1]
                            horiz_speed = vel_x ** 2 + vel_z ** 2
                            near_ground = (global_y < args.foot_height_thresh).float()
                            contact = near_ground[:, :-1] * near_ground[:, 1:]
                            slide_losses.append(horiz_speed * contact)
                        loss_foot_slide_val += torch.stack(slide_losses, dim=0).mean()

                    loss_foot_sink_val = loss_foot_sink_val / 2
                    loss_foot_slide_val = loss_foot_slide_val / 2

                    # -----------------------------------------------------------
                    # Total Loss & Aggregation
                    # -----------------------------------------------------------

                    loss = (0.5*(loss_rec_m + loss_rec_v) +
                                        args.aux_loss_joints * (0.5*(loss_explicit_v + loss_explicit_m)) +
                                        args.style_loss_multiplier * style_loss_vid +
                                        args.embed_loss_multiplier * loss_embedding +
                                        args.adv_loss_multiplier * loss_adv_val +
                                        args.cycle_loss_multiplier * loss_cycle_val +
                                        args.foot_sink_multiplier * loss_foot_sink_val +
                                        args.foot_slide_multiplier * loss_foot_slide_val)

                    # --- ADD VISUALIZATION 2 (Collect) ---
                    all_video_embeddings.append(v_latent.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    # --- ADD VISUALIZATION 2 (Collect) ---


                    val_loss.append(loss.item())
                    val_loss_rec_m.append(loss_rec_m.item())
                    val_loss_rec_v.append(loss_rec_v.item())
                    val_loss_vel_m.append(loss_explicit_m.item())
                    val_loss_vel_v.append(loss_explicit_v.item())
                    val_loss_embed.append(loss_embedding.item())
                    val_loss_disc.append(loss_disc_val.item())
                    val_loss_adv.append(loss_adv_val.item())
                    val_loss_cycle.append(loss_cycle_val.item())
                    val_loss_cycle_m.append(loss_cycle_m_val.item())
                    val_loss_cycle_v.append(loss_cycle_v_val.item())
                    val_loss_cycle_vm.append(loss_cycle_vm_val.item())

                    val_loss_foot_sink.append(loss_foot_sink_val.item())
                    val_loss_foot_slide.append(loss_foot_slide_val.item())
                    val_loss_style_v.append(style_loss_vid.item())
                    # val_loss_style_m.append(style_loss_motion.item())
                    
                    pred_vid = torch.argmax(style_logits_vid, dim=1)
                    # pred_motion = torch.argmax(style_logits_motion, dim=1)

                    val_correct_vid += (pred_vid == labels).sum().item()
                    # val_correct_motion += (pred_motion == labels).sum().item()
                    # val_correct_both += ((pred_vid == labels) & (pred_motion == labels)).sum().item()
                    # break
        
        # if epoch % 10 == 0 or epoch != args.epoch-1 or epoch == 0:
        #     all_video_embeddings = np.concatenate(all_video_embeddings, axis=0)
        #     all_labels = np.concatenate(all_labels, axis=0)

        #     visualize_video_embeddings(args, all_video_embeddings, all_labels, args.style_classes, epoch)

        train_metrics = {
            'train/loss': epoch_train_loss / len(train_loader),
            'train/loss_rec': epoch_train_rec_loss / len(train_loader),
            'train/loss_vel': epoch_train_vel_loss / len(train_loader),
            'train/loss_style': epoch_train_style_loss / len(train_loader),
            'train/loss_embed': epoch_train_embed_loss / len(train_loader),
            'train/loss_disc': epoch_train_disc_loss / len(train_loader),
            'train/loss_adv': epoch_train_adv_loss / len(train_loader),
            'train/loss_cycle': epoch_train_cycle_loss / len(train_loader),
            'train/loss_foot_sink': epoch_train_foot_sink_loss / len(train_loader),
            'train/loss_foot_slide': epoch_train_foot_slide_loss / len(train_loader),
            'train/accuracy': epoch_train_correct / len(train_dataset)
        }

        val_metrics = {
            'val/loss': sum(val_loss) / len(val_loss),
            'val/loss_rec_motion': sum(val_loss_rec_m) / len(val_loss_rec_m),
            'val/loss_rec_video': sum(val_loss_rec_v) / len(val_loss_rec_v),

            'val/loss_vel_motion': sum(val_loss_vel_m) / len(val_loss_vel_m),
            'val/loss_vel_video': sum(val_loss_vel_v) / len(val_loss_vel_v),
            'val/loss_embed': sum(val_loss_embed) / len(val_loss_embed),   
            'val/loss_disc': sum(val_loss_disc) / len(val_loss_disc),
            'val/loss_adv': sum(val_loss_adv) / len(val_loss_adv),

            'val/loss_cycle': sum(val_loss_cycle) / len(val_loss_cycle),
            'val/loss_cycle_motion': sum(val_loss_cycle_m) / len(val_loss_cycle_m),
            'val/loss_cycle_video': sum(val_loss_cycle_v) / len(val_loss_cycle_v),
            'val/loss_cycle_vm': sum(val_loss_cycle_vm) / len(val_loss_cycle_vm),

            # 'val/loss_style_motion': sum(val_loss_style_v) / len(val_loss_style_v),
            'val/loss_style_video': sum(val_loss_style_v) / len(val_loss_style_v),
            
            'val/loss_foot_sink': sum(val_loss_foot_sink) / len(val_loss_foot_sink),
            'val/loss_foot_slide': sum(val_loss_foot_slide) / len(val_loss_foot_slide),
            'val/accuracy_video': val_correct_vid / len(val_loader)
            # 'val/accuracy_motion': val_correct_motion / len(val_loader),
            # 'val/accuracy_both': val_correct_both / len(val_loader)
        }

        # wandb.log({**train_metrics, **val_metrics, 'epoch': epoch-1}, step=it)
        # Log the validation metrics. Using the global iteration `it` as the step
        # keeps the x-axis consistent with the training logs.


        logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
        logger.add_scalar('Val/loss_rec_video', sum(val_loss_rec_v) / len(val_loss_rec_v), epoch)
        logger.add_scalar('Val/loss_rec_motion', sum(val_loss_rec_m) / len(val_loss_rec_m), epoch)
        logger.add_scalar('Val/loss_vel_motion', sum(val_loss_vel_m) / len(val_loss_vel_m), epoch)
        logger.add_scalar('Val/loss_vel_video', sum(val_loss_vel_v) / len(val_loss_vel_v), epoch)
        logger.add_scalar('Val/loss_embed', sum(val_loss_embed) / len(val_loss_embed), epoch)
        logger.add_scalar('Val/loss_disc', sum(val_loss_disc) / len(val_loss_disc), epoch)
        logger.add_scalar('Val/loss_adv', sum(val_loss_adv) / len(val_loss_adv), epoch)
        logger.add_scalar('Val/loss_cycle', sum(val_loss_cycle) / len(val_loss_cycle), epoch)
        logger.add_scalar('Val/loss_cycle_motion', sum(val_loss_cycle_m) / len(val_loss_cycle_m), epoch)
        logger.add_scalar('Val/loss_cycle_video', sum(val_loss_cycle_v) / len(val_loss_cycle_v), epoch)
        logger.add_scalar('Val/loss_cycle_vm', sum(val_loss_cycle_vm) / len(val_loss_cycle_vm), epoch)

        logger.add_scalar('Val/loss_style_video', sum(val_loss_style_v) / len(val_loss_style_v), epoch)
        logger.add_scalar('Val/loss_foot_sink', sum(val_loss_foot_sink) / len(val_loss_foot_sink), epoch)
        logger.add_scalar('Val/loss_foot_slide', sum(val_loss_foot_slide) / len(val_loss_foot_slide), epoch)
        # logger.add_scalar('Val/loss_style_motion', sum(val_loss_style_m) / len(val_loss_style_m), epoch)
        
        for tag, value in val_metrics.items():
            logger.add_scalar(tag.replace('val', 'Val'), value, epoch)

        # print('Validation Loss: %.5f, Reconstruction: %.5f, Velocity: %.5f,' %
        #       (val_metrics['val/loss'], val_metrics['val/loss_rec_video'], val_metrics['val/loss_rec_motion'],val_metrics['val/loss_vel']))
        
        print(
            "Training loss: %.5f, Rec: %.5f, Embedding : %.5f, Velocity: %.5f, Acc: %.5f, Discriminator: %5f, Generator: %.5f, Cycle loss: %.5f"
            % (
                train_metrics["train/loss"],
                train_metrics["train/loss_rec"],
                train_metrics["train/loss_embed"],
                train_metrics["train/loss_vel"],
                train_metrics["train/accuracy"],
                train_metrics["train/loss_disc"],
                train_metrics["train/loss_adv"],
                train_metrics["train/loss_cycle"],
            )
        )


        # print(
        #     "Validation Loss: %.5f, Rec Video: %.5f, Rec Motion: %.5f, Velocity video: %.5f, Velocity motion: %.5f, Embedding: %.5f, Style Video: %.5f, Style Motion: %.5f, Acc Video: %.5f, Acc Motion: %.5f, Acc Both: %.5f"
        #     % (
        #         val_metrics["val/loss"],
        #         val_metrics["val/loss_rec_video"],
        #         val_metrics["val/loss_rec_motion"],

        #         val_metrics["val/loss_vel_video"],
        #         val_metrics["val/loss_vel_motion"],
        #         val_metrics["val/loss_embed"],

        #         val_metrics["val/loss_style_video"],
        #         val_metrics["val/loss_style_motion"],

        #         val_metrics["val/accuracy_video"],
        #         val_metrics["val/accuracy_motion"],
        #         val_metrics["val/accuracy_both"],
        #     )
        # )

        print(
            "Validation Loss: %.5f, Rec Video: %.5f, Rec Motion: %.5f, Velocity video: %.5f, Velocity motion: %.5f, Embedding: %.5f, Discriminator: %.5f, Generator: %.5f, Cycle: %.5f, Cycle Motion: %.5f, Cycle Video: %.5f, Cycle Video-Motion: %.5f, Style Video: %.5f, Acc Video: %.5f"
            % (
                val_metrics["val/loss"],
                val_metrics["val/loss_rec_video"],
                val_metrics["val/loss_rec_motion"],

                val_metrics["val/loss_vel_video"],
                val_metrics["val/loss_vel_motion"],
                val_metrics["val/loss_embed"],

                val_metrics["val/loss_disc"],
                val_metrics["val/loss_adv"],
                val_metrics["val/loss_cycle"],
                val_metrics["val/loss_cycle_motion"],
                val_metrics["val/loss_cycle_video"],
                val_metrics["val/loss_cycle_vm"],

                val_metrics["val/loss_style_video"],
                # val_metrics["val/loss_style_motion"],

                val_metrics["val/accuracy_video"]
                # val_metrics["val/accuracy_motion"],
                # val_metrics["val/accuracy_both"],
            )
        )


        #################################################################################

    # save(pjoin(model_dir, 'final.tar'), epoch, ae, optimizer, scheduler, it, 'ae')
    if args.train_style == 'motion':
        save(pjoin(model_dir, 'final.tar'), epoch, dae, optimizer, scheduler, it, 'ae')
    elif args.train_style == 'full':
        # save_upd(pjoin(model_dir, 'final.tar'), epoch, dae, optimizer, scheduler, it, 'ae')
        save_disc_upd(pjoin(model_dir, 'final.tar'), epoch, dae, optimizer, scheduler, it, 'ae', discriminator=discriminator, opt_disc=optimizer_disc)
    print(f"Saved final model at location: {pjoin(model_dir, 'final.tar')}") 

    delete_old_checkpoints(model_dir)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time / 60:.2f} minutes")

    # wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='DAE')
    parser.add_argument('--model', type=str, default='DAE_Model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, choices=['t2m','100styles'])
    parser.add_argument("--style_classes", type=str, nargs='+', default=["Aeroplane", "Chicken", "Robot", "Superman", "ArmsFolded"])
                                                                        #  "BeatChest", \
                                                                        #   "BigSteps", "Rocket", "Monk", "LegsApart", "CrowdAvoidance", "Flapping", \
                                                                        #   "Flapping", "Elated", "Balance"], 
                                                                        #   help="List of style names (subdirectory names).")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[45000, 55000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--aux_loss_joints', type=float, default=1)
    parser.add_argument('--style_loss_multiplier', type=float, default=1)
    parser.add_argument('--embed_loss_multiplier', type=float, default=1)
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
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--snippets_per_sequence', default=15, type=int)
    parser.add_argument('--exp_name', type=str, default='MMAE_test')
    parser.add_argument('--viz_dir',type=str, default='./visualizations/test_runs')
    parser.add_argument('--train_style', type=str, default='full', choices=['full', 'motion', 'video'], help="Train full arch or parts only.")
    parser.add_argument('--adv_loss_multiplier', type=float, default=0.01, help="Weight for the GAN/Adversarial loss.")
    parser.add_argument('--disc_start_epoch', type=int, default=30, help="Epoch to start training the Discriminator (Phase 2).")
    parser.add_argument('--cycle_start_epoch', type=int, default=70, help="Epoch to start Cycle Consistency Loss (Phase 3).")
    parser.add_argument('--cycle_loss_multiplier', type=float, default=0.1, help="Weight for the Cycle Consistency loss.")
    parser.add_argument('--foot_sink_multiplier', type=float, default=0.05, help="Weight for the foot sinking loss.")
    parser.add_argument('--foot_slide_multiplier', type=float, default=0.01, help="Weight for the foot sliding loss.")
    parser.add_argument('--foot_height_thresh', type=float, default=0.05, help="Height threshold for foot-ground contact detection.")

    
    arg = parser.parse_args()
    main(arg)