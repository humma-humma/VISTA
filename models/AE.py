import torch
import math
import torch.nn as nn
import numpy as np
from typing import List
import torch.nn.functional as F

from transformers import AutoImageProcessor, VivitModel, TimesformerModel, VivitImageProcessor, AutoProcessor
from decord import VideoReader, cpu
import glob
from torchvision import transforms
from utils.profiling import timer

#################################################################################
#                           Unravelling heads post Encoder                      #
#################################################################################

class LVideoAdapter(nn.Module):
    def __init__(self, input_frames=32, target_frames=16, spatial_grid_size=14, hidden_dim=768):
        super().__init__()
        self.target_frames = target_frames
        self.spatial_tokens = spatial_grid_size * spatial_grid_size # 196
        self.vivit_temporal_steps = input_frames // 2  
        
        # --- NEW: Learnable Spatial Attention ---
        # A tiny neural net that looks at each patch and assigns an "Importance Score"
        self.attn_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        """
        Input: [Batch, 3137, 768]
        Output: [Batch, target_frames, 768]
        """
        b, tokens, c = x.shape
        
        # 1. Remove CLS token
        x = x[:, 1:, :] 
        
        # 2. Unflatten [Batch, Time, Space, Channels]
        # Shape: [Batch, 16, 196, 768]
        x = x.view(b, self.vivit_temporal_steps, self.spatial_tokens, c)
        
        # 3. Calculate Attention Scores
        # We want to know which of the 196 patches are important
        attn_scores = self.attn_projection(x)       # [Batch, 16, 196, 1]
        attn_weights = F.softmax(attn_scores, dim=2) # Normalize so they sum to 1
        
        # 4. Weighted Pooling (Learnable!)
        # Instead of x.mean(), we do x * weights
        x = (x * attn_weights).sum(dim=2) # [Batch, 16, 768]
        
        # 5. Interpolate (Same as before)
        if x.shape[1] != self.target_frames:
            x = x.permute(0, 2, 1)
            x = F.interpolate(x, size=self.target_frames, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)
            
        return x


class VideoAdapter(nn.Module):
    def __init__(self, input_frames=32):
        super().__init__()
        self.target_frames = input_frames // 4
        ## ViViT naturally compresses time by 2 (32 frames -> 16 tubelets)
        self.vivit_internal_steps = input_frames // 2  
        
    def forward(self, x):
        """
        Input: [Batch, 3137, 768] (Raw ViViT)
        Output: [Batch, target_frames, 768] (Clean Sequence)
        """
        b, tokens, c = x.shape
        
        ## Discard CLS token (index 0)
        x = x[:, 1:, :]  # [Batch, 3136, 768]
        x = x.view(b, self.vivit_internal_steps, -1, c)
        x = x.mean(dim=2)  # [Batch, 16, 768]
        x = x.permute(0, 2, 1) # [Batch, 768, 16]
            
        return x
    

#################################################################################
#                      Discriminator for GAN-like training                      #
#################################################################################


class MotionDiscriminator(nn.Module):
    def __init__(self, input_width=67, hidden_dim=512, down_t=2, stride_t=2, 
                 width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        
        # 1. Reuse the Encoder structure from your AE
        # This compresses [Batch, 67, 64] -> [Batch, 512, 16]
        self.backbone = Encoder(input_emb_width=input_width, 
                                output_emb_width=hidden_dim,
                                down_t=down_t,
                                stride_t=stride_t,
                                width=width,
                                depth=depth,
                                dilation_growth_rate=dilation_growth_rate,
                                activation=activation,
                                norm=norm)
        
        # 2. A simple classification head
        # self.head = nn.Sequential(nn.Linear(hidden_dim, 256),
        #                           nn.LeakyReLU(inplace=True),
        #                           nn.Linear(256, 128),
        #                           nn.LeakyReLU(inplace=True),
        #                           nn.Linear(128, 64),
        #                           nn.LeakyReLU(inplace=True),
        #                           nn.Linear(64, 1),
        #                           nn.Sigmoid()
        #                           )
        
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(128, 1),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Channels] (e.g. 64 frames, 67 joints)
        """
        # Permute for Conv1d: [Batch, Channels, Time]
        x = x.permute(0, 2, 1) 
        
        # Encode -> [Batch, 512, Compressed_Time]
        features = self.backbone(x)
        
        # Global Average Pooling: Squash time dimension to get one vector per clip
        # [Batch, 512, 16] -> [Batch, 512]
        features = features.mean(dim=2) 
        
        # Classify
        return self.head(features)

#################################################################################
#                              Full architecture                               #
#################################################################################


class VideoE(nn.Module):
    def __init__(self, video_dim=768, latent_dim=512, input_frames=32, 
                 window_size=64, down_t=1, stride_t=2, width=512, 
                 depth=3, num_style_classes=100):
        super().__init__()
        
        # --- Part A: The Adapter ---
        target_latent_frames = int(window_size / (2 ** 2)) 
        self.adapter = VideoAdapter(
            input_frames=input_frames, 
            target_frames=target_latent_frames, # e.g., 16
            spatial_grid_size=14
        )
        
        # --- Part B: The 1D Encoder ---
        self.encoder = Encoder(
            input_emb_width=video_dim,   # 768
            output_emb_width=latent_dim, # 512
            down_t=down_t,               # Usually 1 here, since Adapter handles resizing
            stride_t=stride_t,           # Usually 1
            width=width,
            depth=depth
        )

        self.classifier_head = nn.Linear(latent_dim, num_style_classes)
        
    def forward(self, x):
        """
        Args:
            x: ViViT raw output [Batch, 3137, 768]
        Returns:
            video_latent: [Batch, latent_dim, target_latent_steps]
        """
        # 1. Adapter
        x_adapted = self.adapter(x)              # [Batch, target_latent_steps, 768]
        
        # 2. Encoder
        x_in = x_adapted.permute(0, 2, 1)       # [Batch, 768, target_latent_steps]
        video_latent = self.encoder_1d(x_in)    # [Batch, latent_dim, target_latent_steps]
        
        logits = self.classifier_head(video_latent.mean(dim=2))
        return video_latent, logits

class DualAE(nn.Module):
    def __init__(self, 
                 ## Motion side
                 input_width=67, output_emb_width=512, down_t_m=1, down_t_v=0, stride_t=2, width=512, 
                 depth=3, dilation_growth_rate=3, activation='relu', norm=None, window_size=32,
                 ## Video side
                 video_dim=768,
                 is_classification=True,
                 num_style_classes=100):
        super().__init__()
        print(f"Setting up arch with downsampling in video by a factor of: {down_t_v} and motion by a factor of: {down_t_m}")
        self.output_emb_width = output_emb_width
        ## Motion Encoder (Standard) ---
        ## Compresses 64 frames -> 16 frames (if down_t=2)
        self.motion_encoder = Encoder(input_width, output_emb_width, down_t_m, stride_t, width, depth,
                                      dilation_growth_rate, activation=activation, norm=norm)

        ## Video Branch (Structural) ---
        ## Step A: Adapter. Converts ViViT tokens to 16 temporal steps to match Motion Latent
        self.video_adapter = VideoAdapter(input_frames=window_size)
        
        ## Step B: Video Encoder. 
        self.video_encoder = Encoder(video_dim, output_emb_width, down_t_v, stride_t, width, depth,
                                     dilation_growth_rate, activation=activation, norm=norm)
        
        if is_classification:
            projection_out_dim = output_emb_width
            self.classifier_head = nn.Linear(projection_out_dim, num_style_classes)

        # if is_classification:
        #     temporal_steps = 16
        #     projection_out_dim = output_emb_width * temporal_steps
        #     self.classifier_head = nn.Linear(projection_out_dim, num_style_classes)


        ## Shared Decoder
        ## Expands 16 frames -> 64 frames
        self.decoder = Decoder(input_width, output_emb_width, down_t_m, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def preprocess_motion(self, x):
        # Permute (Batch, Time, Channels) -> (Batch, Channels, Time)
        return x.permute(0, 2, 1).float()

    def forward(self, motion_x, video_x, force_mode=None):
        """
        Args:
            motion_x: Ground truth motion
            video_x: Video input
            force_mode: 'motion', 'video', or None (random switch)
        Returns:
            dict: Contains 'recon', 'motion_latent', 'video_latent', 'mode_used'
        """
        ## Encode Motion (Always)
        x_in = self.preprocess_motion(motion_x)
        motion_latent = self.motion_encoder(x_in)

        ## Encode Video (Always)
        vid_in = self.video_adapter(video_x)
        video_latent = self.video_encoder(vid_in)

        ## The 50/50 Logic
        if force_mode == 'video':
            use_video = True
        elif force_mode == 'motion':
            use_video = False
        else:
            # Default training behavior: 50/50 split
            use_video = (torch.rand(1).item() < 0.5)
            # print("=== Using randomized mixed training strategy: 50% w/ video projection, 50% w/ motion projection===")


        # 4. Decode the chosen latent
        if use_video:
            latent_to_decode = video_latent
            mode_used = 'video'
            # logits = self.classifier_head(video_latent.mean(dim=2))
            # logits = self.classifier_head(latent_to_decode)
        else:
            latent_to_decode = motion_latent
            mode_used = 'motion'
            logits = self.classifier_head(video_latent.mean(dim=2))
            # logits = self.classifier_head(latent_to_decode)

        recon_motion = self.decoder(latent_to_decode)
        logits = self.classifier_head(video_latent.mean(dim=2))


        # 5. Return a clean dictionary
        return {
            'recons': recon_motion,
            'motion_latent': motion_latent,
            'video_latent': video_latent,
            'style_logits': logits,
            'mode_used': mode_used
        }
    
    def forward_eval(self, motion_x, video_x):
        """
        Args:
            motion_x: Ground truth motion
            video_x: Video input
            force_mode: 'motion', 'video', or None (random switch)
        Returns:
            dict: Contains 'recon', 'motion_latent', 'video_latent', 'mode_used'
        """
        ## Encode Motion (Always)
        x_in = self.preprocess_motion(motion_x)
        motion_latent = self.motion_encoder(x_in)

        ## Encode Video (Always)
        vid_in = self.video_adapter(video_x)
        video_latent = self.video_encoder(vid_in)

        # ## The 50/50 Logic
        # if force_mode == 'video':
        #     use_video = True
        # elif force_mode == 'motion':
        #     use_video = False
        # else:
        #     # Default training behavior: 50/50 split
        #     use_video = (torch.rand(1).item() < 0.5)
        #     print("=== Using randomized mixed training strategy: 50% w/ video projection, 50% w/ motion projection===")


        # 4. Decode the chosen latent
        # logits = self.classifier_head(latent_to_decode.mean(dim=2))
        video_logits = self.classifier_head(video_latent.mean(dim=2))
        motion_logits = self.classifier_head(motion_latent.mean(dim=2))

        recon_video = self.decoder(video_latent)
        recon_motion = self.decoder(motion_latent)

        # 5. Return a clean dictionary
        return {
            'recons_motion': recon_motion,
            'recons_video': recon_video,
            'motion_latent': motion_latent,
            'video_latent': video_latent,
            'video_logits': video_logits,
            'motion_logits': motion_logits,
        }
    
    def encode_motion(self, motion_x):
        x_in = self.preprocess_motion(motion_x)
        return self.motion_encoder(x_in)
    
    def encode_video(self, video_x):
        vid_in = self.video_adapter(video_x)
        video_latent = self.video_encoder(vid_in)

        return video_latent

    def decode(self, latent):
        return self.decoder(latent)
       
    
    def encode(self, motion_x, video_x):
        x_in = self.preprocess_motion(motion_x)
        motion_latent = self.motion_encoder(x_in)

        vid_in = self.video_adapter(video_x)
        video_latent = self.video_encoder(vid_in)

        return motion_latent, video_latent


class DualVAE(nn.Module):
    def __init__(self, 
                 ## Motion side
                 input_width=67, output_emb_width=512, down_t_m=1, down_t_v=0, stride_t=2, width=512, 
                 depth=3, dilation_growth_rate=3, activation='relu', norm=None, window_size=32,
                 ## Video side
                 video_dim=768,
                 is_classification=True,
                 num_style_classes=100):
        super().__init__()
        print(f"Setting up variational arch with downsampling in video by a factor of: {down_t_v} and motion by a factor of: {down_t_m}")
        self.output_emb_width = output_emb_width
        ## Motion Encoder (Standard) ---
        ## Compresses 64 frames -> 16 frames (if down_t=2)
        self.motion_encoder = VariationalEncoder(input_width, output_emb_width, down_t_m, stride_t, width, depth,
                                      dilation_growth_rate, activation=activation, norm=norm)

        ## Video Branch (Structural) ---
        ## Step A: Adapter. Converts ViViT tokens to 16 temporal steps to match Motion Latent
        self.video_adapter = VideoAdapter(input_frames=window_size)
        
        ## Step B: Video Encoder. 
        self.video_encoder = VariationalEncoder(video_dim, output_emb_width, down_t_v, stride_t, width, depth,
                                     dilation_growth_rate, activation=activation, norm=norm)
        
        if is_classification:
            projection_out_dim = output_emb_width
            self.classifier_head = nn.Linear(projection_out_dim, num_style_classes)

        # if is_classification:
        #     temporal_steps = 16
        #     projection_out_dim = output_emb_width * temporal_steps
        #     self.classifier_head = nn.Linear(projection_out_dim, num_style_classes)


        ## Shared Decoder
        ## Expands 16 frames -> 64 frames
        self.decoder = Decoder(input_width, output_emb_width, down_t_m, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def reparameterize(self, mu, logvar , training=True):
        """
        The Reparameterization Trick: z = mu + sigma * epsilon
        """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu # Deterministic in eval mode
    
    def preprocess_motion(self, x):
        # Permute (Batch, Time, Channels) -> (Batch, Channels, Time)
        return x.permute(0, 2, 1).float()
    

    def forward(self, motion_x, video_x, force_mode=None):
        """
        Args:
            motion_x: Ground truth motion
            video_x: Video input
            force_mode: 'motion', 'video', or None (random switch)
        Returns:
            dict: Contains 'recon', 'motion_latent', 'video_latent', 'mode_used'
        """
        ## Encode Motion (Always)
        x_in = self.preprocess_motion(motion_x)
        mu_m, logvar_m = self.motion_encoder(x_in)
        z_motion = self.reparameterize(mu_m, logvar_m)

        ## Encode Video (Always)
        vid_in = self.video_adapter(video_x)
        mu_v, logvar_v = self.video_encoder(vid_in)
        z_video = self.reparameterize(mu_v, logvar_v)

        ## The 50/50 Logic
        if force_mode == 'video':
            use_video = True
        elif force_mode == 'motion':
            use_video = False
        else:
            # Default training behavior: 50/50 split
            use_video = (torch.rand(1).item() < 0.5)
            # print("=== Using randomized mixed training strategy: 50% w/ video projection, 50% w/ motion projection===")


        # 4. Decode the chosen latent
        if use_video:
            logits = self.classifier_head(z_video.mean(dim=2))
            recons = self.decoder(z_video)
        else:
            logits = self.classifier_head(z_motion.mean(dim=2))
            recons = self.decoder(z_motion)

        # logits = self.classifier_head(z_video.mean(dim=2))

        # 5. Return a clean dictionary
        return {
            'recons': recons,
            'video_mean': mu_v,
            'style_logits': logits,
            'motion_mean': mu_m,             # For KL Loss
            'motion_logvar': logvar_m        # For KL Loss
        }
    
    def forward_eval(self, motion_x, video_x):
        """
        Args:
            motion_x: Ground truth motion
            video_x: Video input
        Returns:
            dict: Contains reconstructions, latents (means), and logits for evaluation.
        """
        ## Encode Motion (Always)
        x_in = self.preprocess_motion(motion_x)
        mu_m, logvar_m = self.motion_encoder(x_in)
        z_motion = self.reparameterize(mu_m, logvar_m, training=False)

        ## Encode Video (Always)
        vid_in = self.video_adapter(video_x)
        mu_v, logvar_v = self.video_encoder(vid_in)
        z_video = self.reparameterize(mu_v, logvar_v, training=False) # Deterministic "best guess"

        ## Decode & Classify
        ## We classify the mean directly during eval
        video_logits = self.classifier_head(z_video.mean(dim=2))
        motion_logits = self.classifier_head(z_motion.mean(dim=2))

        recon_video = self.decoder(z_video)
        recon_motion = self.decoder(z_motion)

        return {
                'recons_motion': recon_motion,
                'recons_video': recon_video,
                'video_logits': video_logits,
                'motion_logits': motion_logits,
                'motion_latent': z_motion,    # This is mu_m (used for embedding alignment check)
                'video_latent': z_video,      # This is mu_v (used for embedding alignment check)
                'motion_logvar': logvar_m,    # Returned in case you want to compute Val KL
                'video_logvar': logvar_v
            }
    
    def encode_motion(self, motion_x):
        x_in = self.preprocess_motion(motion_x)
        mu, logvar = self.motion_encoder(x_in)
        return mu, logvar

    def decode(self, latent):
        return self.decoder(latent)

class AE(nn.Module):
    def __init__(self, input_width=67, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        self.output_emb_width = output_emb_width
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def preprocess(self, x):
        x = x.permute(0, 2, 1).float()
        return x

    def encode(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        return x_encoder

    def forward(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_out = self.decoder(x_encoder)
        return x_out

    def decode(self, x):
        x_out = self.decoder(x)
        return x_out

#################################################################################
#                                      AE Zoos                                  #
#################################################################################
MODEL_CONFIG = {
        'vivit': {"name": "google/vivit-b-16x2-kinetics400", "processor": "google/vivit-b-16x2-kinetics400", "num_frames": 32, "uses_cls": True},
        'timesformer': {"name": "facebook/timesformer-base-finetuned-k400", "processor": "MCG-NJU/videomae-base", "num_frames": 32, "uses_cls": True},
        'xclip': {"name": "microsoft/xclip-base-patch32", "processor": "microsoft/xclip-base-patch32", "num_frames": 32, "uses_cls": True},
        # 'magvit': {"name": "magvit", "processor": "", "num_frames": 16, "uses_cls": False},
    }

# def ae(**kwargs):
#     # return AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
#     #              dilation_growth_rate=3, activation='relu', norm=None, **kwargs)
#     # return AEProjector(num_style_classes=kwargs.pop('num_style_classes', 100), output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
#     #              dilation_growth_rate=3, activation='relu', norm=None, **kwargs)
#     return AEProjector_new(num_style_classes=kwargs.pop('num_style_classes', 100), output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
#                 dilation_growth_rate=3, activation='relu', norm=None, **kwargs)

def ae(**kwargs):
    return AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None, **kwargs)

def dae(**kwargs):
    return DualAE(output_emb_width=512, down_t_m=2, down_t_v=1, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None, **kwargs)

def dvae(**kwargs):
    return DualVAE(output_emb_width=512, down_t_m=2, down_t_v=1, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None, **kwargs)

def discriminator(**kwargs):
    return MotionDiscriminator(input_width=kwargs.get('input_width', 67), hidden_dim=512, down_t=2, activation='relu')

def ae_sample(**kwargs):
    return AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None, **kwargs)

def style_classifier(**kwargs):
    return VideoE(down_t=0, stride_t=2, width=512, depth=3, is_classification=True, 
                  num_style_classes=100, **kwargs)
    

# def scc(encoder_name="vivit", config=MODEL_CONFIG, **kwargs):
#     video_backbone = VideoEncoderBackbone(
#         model_name=config[encoder_name]["name"],
#         processor=config[encoder_name]["processor"],
#         uses_cls=config[encoder_name]["uses_cls"],
#         num_frames=config[encoder_name]["num_frames"],
#     )
#     return StyleClassifierConv(
#         backbone=video_backbone,
#         num_classes=kwargs.get('num_classes', 100),  # Default to 100 classes
#         projection_out_dim=kwargs.get('projection_out_dim', 512)
#     )

# AE_models = {
#     'AE_Model': ae,
#     'Style_Classifier': scc
# }

DAE_models_full = {
    'DAE_Model': dae,
    'Video arm': VideoE
}

DAE_models = {
    'DAE_Model': dae

}

DVAE_models = {
    'DVAE_Model': dvae

}

DAED_models = {
    'DAE_Model': dae,
    'Discriminator': discriminator

}

AE_eval = {
    'AE_Model': ae_sample
}

AE_models = {
    'AE_Model': ae
}


#################################################################################
#                                 Inner Architectures                           #
#################################################################################

class VariationalEncoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        
        # CHANGE 1: Output double the channels (chunk 1 = mean, chunk 2 = log_variance)
        self.model = nn.Sequential(*blocks)
        self.fc_mu = nn.Conv1d(width, output_emb_width, 3, 1, 1)
        self.fc_logvar = nn.Conv1d(width, output_emb_width, 3, 1, 1)

    def forward(self, x):
        out = self.model(x)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


class Encoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)

        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        ## Might need to comment these
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class nonlinearity(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.2):
        super(ResConv1DBlock, self).__init__()
        padding = dilation
        self.norm = norm

        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig
        return x