"""
Style Classification Training Script (67-dim motion input)
Combined model definition + training loop for Phase 1 compatibility
"""

import os
from os.path import join as pjoin
import math
import argparse
from statistics import mean
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Union
from utils.datasets import StyleMotionDataset
from tqdm import tqdm
import numpy as np

#################################################################################
#                              Utility Functions                                #
#################################################################################

def lengths_to_mask(lengths: List[int], device: torch.device) -> torch.Tensor:
    """
    Convert list of lengths to boolean mask.
    Args:
        lengths: List of sequence lengths [B]
        device: Target device
    Returns:
        mask: Boolean tensor [B, max_len], True for valid positions
    """
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


def conv_layer(kernel_size, in_channels, out_channels, pad_type='replicate'):
    """1D convolution with padding."""
    def zero_pad_1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = zero_pad_1d

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return nn.Sequential(pad((pad_l, pad_r)), nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))


#################################################################################
#                             Position Encoding                                 #
#################################################################################

class PositionalEncoding(nn.Module):
    """Learned or sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, mode: str = "learned"):
        super().__init__()
        self.mode = mode
        
        if mode == "learned":
            self.pe = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)
        else:
            # Sinusoidal
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
            self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [seq_len, batch, d_model]
        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(0)]


#################################################################################
#                           Transformer Components                              #
#################################################################################

class TransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer with pre/post norm option."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu if activation == "gelu" else F.relu
        self.normalize_before = normalize_before
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.normalize_before:
            src2 = self.norm1(src)
            src2, _ = self.self_attn(src2, src2, src2, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None):
        super().__init__()
        self.layers = nn.ModuleList([
            self._clone_layer(encoder_layer) for _ in range(num_layers)
        ])
        self.norm = norm
        self.num_layers = num_layers
    
    def _clone_layer(self, layer):
        """Create a new layer with same config."""
        return TransformerEncoderLayer(
            d_model=layer.self_attn.embed_dim,
            nhead=layer.self_attn.num_heads,
            dim_feedforward=layer.linear1.out_features,
            dropout=layer.dropout.p,
            activation="gelu" if layer.activation == F.gelu else "relu",
            normalize_before=layer.normalize_before,
        )
    
    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        is_intermediate: bool = False,
    ) -> Union[Tensor, tuple]:
        output = src
        intermediates = []
        
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
            if is_intermediate:
                intermediates.append(output)
        
        if self.norm is not None:
            output = self.norm(output)
        
        if is_intermediate:
            return output, torch.stack(intermediates)
        return output


#################################################################################
#                            Style Classification Model                          #
#################################################################################

class StyleClassification(nn.Module):
    """
    Transformer-based style classifier for motion sequences.
    
    Args:
        nclasses: Number of style classes
        input_dim: Input motion feature dimension (67 for joint positions, 263 for HumanML3D)
        latent_dim: [num_tokens, hidden_dim] - transformer configuration
        ff_size: Feedforward network size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        nclasses: int,
        input_dim: int = 67,  # Changed from 263 to 67
        latent_dim: list = [1, 256],
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        normalize_before: bool = False,
        activation: str = "gelu",
        position_embedding: str = "learned",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.style_num = nclasses
        self.input_dim = input_dim
        self.latent_dim = latent_dim[-1]  # Hidden dimension (256)
        self.latent_size = latent_dim[0]  # Number of global tokens (1)
        
        # Input projection: motion_dim -> latent_dim
        self.skel_embedding = nn.Linear(input_dim, self.latent_dim)
        
        # Learnable global motion token(s) for classification
        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.latent_dim)
        )
        
        # Positional encoding
        self.query_pos = PositionalEncoding(self.latent_dim, mode=position_embedding)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        
        # Classification head
        self.classifier = nn.Linear(self.latent_dim, self.style_num)
    
    def forward(
        self,
        features: Tensor,
        lengths: Optional[List[int]] = None,
        stage: str = "Classification",
    ) -> Union[Tensor, tuple]:
        """
        Forward pass with multiple output modes.
        
        Args:
            features: Motion tensor [B, T, input_dim]
            lengths: Optional list of sequence lengths
            stage: Output mode
                - "Classification": Return logits only [B, nclasses]
                - "Both": Return (logits, features) tuple
                - "Encode": Return global feature [B, latent_dim]
                - "Encode_all": Return all encoder outputs
        
        Returns:
            Depends on stage parameter
        """
        if lengths is None:
            lengths = [features.size(1)] * features.size(0)
        
        device = features.device
        bs, nframes, nfeats = features.shape
        
        # Create attention mask
        mask = lengths_to_mask(lengths, device)
        
        # Project input to latent dimension
        x = self.skel_embedding(features.float())  # [B, T, latent_dim]
        
        # Switch to [T, B, latent_dim] for transformer
        x = x.permute(1, 0, 2)
        
        # Prepend global tokens
        dist = self.global_motion_token[:, None, :].expand(-1, bs, -1)  # [2, B, latent_dim]
        
        # Extend mask for global tokens
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=bool, device=device)
        aug_mask = torch.cat((dist_masks, mask), dim=1)
        
        # Concatenate and add positional encoding
        xseq = torch.cat((dist, x), dim=0)  # [2+T, B, latent_dim]
        xseq = self.query_pos(xseq)
        
        # Transformer encoding
        encoded = self.encoder(xseq, src_key_padding_mask=~aug_mask)
        
        # Output based on stage
        if stage == "Encode":
            return encoded[0]  # First global token [B, latent_dim]
        
        elif stage == "Encode_all":
            return encoded
        
        elif stage == "Classification":
            feat = encoded[0]  # [B, latent_dim]
            output = self.classifier(feat)  # [B, nclasses]
            return output
        
        elif stage == "Both":
            feat = encoded[0]  # [B, latent_dim]
            output = self.classifier(feat)  # [B, nclasses]
            return output, feat
        
        elif stage == "intermediate":
            _, intermediate = self.encoder(xseq, src_key_padding_mask=~aug_mask, is_intermediate=True)
            style_features = []
            intermediate = intermediate[-2:]
            for i in range(intermediate.size(0)):
                sub_tensor = intermediate[i]
                mean = torch.mean(sub_tensor, dim=[0], keepdim=True)
                std = torch.std(sub_tensor, dim=[0], keepdim=True)
                style_features.append((mean, std))
            return style_features
        
        else:
            raise ValueError(f"Unknown stage: {stage}")

#################################################################################
#                               Training Loop                                    #
#################################################################################

def train(args):
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

    num_classes = len(args.styles)

    print(f"Training style classifier with {args.input_dim}-dim motion input")
    print(f"Classes: {num_classes}, Epochs: {args.epochs}")
    
    # Create datasets

    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    data_root = f'{args.dataset_dir}/100STYLE-SMPL/'
    dim_pose = 67
    
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')
    dict_file = pjoin(data_root, '100STYLE_name_dict_length.txt')
    train_split_file = pjoin(data_root, 'train_100STYLE_Full.txt')
    val_split_file = pjoin(data_root, 'test_100STYLE_Full.txt')



    print("Initializing Datasets...")
    train_dataset = StyleMotionDataset(stage='train', data_root=data_root, motion_dir= motion_dir, text_dir=text_dir, dict_file=dict_file, styles=args.styles, split_file=train_split_file, use_augmentation=True, dim_pose=dim_pose)
    val_dataset_full = StyleMotionDataset(stage='test', data_root=data_root, motion_dir= motion_dir, text_dir=text_dir, dict_file=dict_file, styles=args.styles, split_file=val_split_file, use_augmentation=True, dim_pose=dim_pose)    
    
    val_size = len(val_dataset_full)*2 // 3
    test_size = len(val_dataset_full) - val_size

    val_dataset, test_dataset = torch.utils.data.random_split(
        val_dataset_full, 
        [val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed) # for reproducibility
    )

    print(f"Dataet loaded - full val: {len(val_dataset_full)}, val: {len(val_dataset)}, test: {len(test_dataset)}, train: {len(train_dataset)}")
    

    train_loader = DataLoader(train_dataset,  batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"DataLoaders created - train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
    
    # Create model
    model = StyleClassification(nclasses=num_classes, input_dim=args.input_dim, latent_dim=[1, args.latent_dim], ff_size=args.ff_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout).cuda()
    model.to(device)

    # Print parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params / 1e6:.2f}M")
    
    # Optimizer and loss
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build style_to_idx once so it can be saved in every checkpoint
    style_to_idx = {s: i for i, s in enumerate(args.styles)}

    # Training loop
    best_acc = 0.0

    for epoch in tqdm(range(args.epochs)):
        # ===================== Training =====================
        model.train()
        train_loss = []
        train_correct = 0
        train_total = 0
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            motion = batch['motion'].to(device)  # [B, T, 67]
            labels = batch['label'].to(device)   # [B]
            
            # Forward pass
            logits, _ = model(motion, stage="Both")
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_loss.append(loss.item())
            _, predicted = torch.max(logits, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            
            # Logging
            avg_loss = mean(train_loss)
            acc = 100 * train_correct / train_total
            print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f} Acc: {acc:.2f}%")
        
        scheduler.step()
        
        # ===================== Evaluation =====================
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                motion = batch['motion'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(motion, stage="Classification")
                _, predicted = torch.max(logits, dim=1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        print(f"\n[Eval] Epoch [{epoch+1}/{args.epochs}] Test Accuracy: {test_acc:.2f}%\n")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.output_dir, "style_classifier_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'args': vars(args),
                'style_to_idx': style_to_idx,
            }, save_path)
            print(f"Saved best model with accuracy {test_acc:.2f}%")
    
        # ===================== Checkpointing =====================
        if (epoch + 1) % 50 == 0 or epoch == 0:
            save_path = os.path.join(args.output_dir, f"style_classifier_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'style_to_idx': style_to_idx,
            }, save_path)
            print(f"Checkpoint saved: {save_path}")
    
    # Final save
    save_path = os.path.join(args.output_dir, "style_classifier_final.pt")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
        'args': vars(args),
        'style_to_idx': style_to_idx,
    }, save_path)
    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
    print(f"Final model saved: {save_path}")


#################################################################################
#                                    Main                                        #
#################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Style Classifier (67-dim)")
    
    # Data
    # parser.add_argument('--dataset_dir', type=str, default='./datasets/100STYLE-SMPL/')  # code appends /100STYLE-SMPL/ itself
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--nclasses', type=int, default=47, help='UNUSED: the head size is len(--styles)')
    parser.add_argument('--input_dim', type=int, default=67, help='Motion feature dimension')
    parser.add_argument('--styles', type=str, nargs='+', default=["Aeroplane", "Chicken", "Robot", "Superman", "ArmsFolded", "Cat", "FlickLegs", "HandsBetweenLegs", "Neutral", "InTheDark", "FairySteps", "BeatChest", "BigSteps","Rocket", "Monk", "LegsApart", "CrowdAvoidance", "Flapping", "SpinClock", "Elated", "Balance"])
    
    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=512, help='Classifier hidden dimension')
    parser.add_argument('--ff_size', type=int, default=1024, help='Feedforward size')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1)


    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=2e-4)  # value stored in released checkpoint
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 4,
                        help='DataLoader workers (default 0 on Windows, where worker processes can deadlock)')
    
    # Logging and saving
    # parser.add_argument('--output_dir', type=str, default='./experiments/style_classifier')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/style_classifier')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', type=int, default=0, help='GPU device index')
    
    args = parser.parse_args()
    train(args)