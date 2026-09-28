import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import clip
import math
from functools import partial
from timm.models.vision_transformer import Mlp
from models.DiffMLPs import DiffMLPs_models
from utils.eval_utils import eval_decorator
from utils.train_utils import lengths_to_mask, uniform, get_mask_subset_prob, cosine_schedule

#################################################################################
#                                      MARDM                                    #
#################################################################################
class MARDM(nn.Module):
    def __init__(self, ae_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.2, clip_dim=512,
                 diffmlps_batch_mul=4, diffmlps_model='SiT-XL', cond_drop_prob=0.1,
                 clip_version='ViT-B/32', 
                 style_routing='diffmlp', # NEW: Architectural Toggle ('mart' or 'diffmlp')
                 style_dim=1024,          # NEW
                 **kargs):
        super(MARDM, self).__init__()

        self.ae_dim = ae_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        self.style_routing = style_routing
        self.style_dim = style_dim

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
            self.num_actions = kargs.get('num_actions', 1)
            self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
            
        # --------------------------------------------------------------------------
        # MAR Tranformer
        print('Loading MARTransformer...')
        self.input_process = InputProcess(self.ae_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        self.MARTransformer = nn.ModuleList([
            MARTransBlock(self.latent_dim, num_heads, mlp_size=ff_size, drop_out=self.dropout, style_dim=self.style_dim) for _ in range(num_layers)
        ])

        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")
        
        # NEW: The Style Projection Layer
        # Takes the concatenated [Mean (512) + Std (512)] and projects to 512
        self.style_proj = nn.Sequential(
            nn.Linear(512 * 2, self.clip_dim),
            nn.SiLU(),
            nn.Linear(self.clip_dim, self.clip_dim)
        )

        self.mask_latent = nn.Parameter(torch.zeros(1, 1, self.ae_dim))

        self.apply(self.__init_weights)
        for block in self.MARTransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            # NEW: Zero Init style branch
            nn.init.constant_(block.style_modulation[-1].weight, 0)
            nn.init.constant_(block.style_modulation[-1].bias, 0)

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        # --------------------------------------------------------------------------
        # DiffMLPs
        print('Loading DiffMLPs...')
        self.DiffMLPs = DiffMLPs_models[diffmlps_model](target_channels=self.ae_dim, z_channels=self.latent_dim)
        self.diffmlps_batch_mul = diffmlps_batch_mul

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)
        assert torch.cuda.is_available()
        clip.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    
    def encode_style(self, raw_style_latents):
        """
        Takes raw [B, 512, T] latents from DualAE, pools them, 
        and projects them to the standard 512-dim style token.
        """
        # 1. Calculate Mean and Std across the temporal dimension (T)
        style_mean = raw_style_latents.mean(dim=-1) # [B, 512]
        style_std = raw_style_latents.std(dim=-1)   # [B, 512]
        
        # 2. Concatenate
        style_concat = torch.cat([style_mean, style_std], dim=-1) # [B, 1024]
        
        # 3. Project back to 512
        style_token = self.style_proj(style_concat) # [B, 512]
        
        return style_token

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, latents, cond, padding_mask, force_mask=False, mask=None, c_style=None, style_weight_schedule=None): 
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.input_process(latents)
        cond = self.cond_emb(cond)
        x = self.position_enc(x)
        x = x.permute(1, 0, 2)
        if mask is not None: 
            sort_indices = torch.argsort(mask.to(torch.float), dim=1)
            x = torch.gather(x, dim=1, index=sort_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            inverse_indices = torch.argsort(sort_indices, dim=1)
            padding_mask = torch.gather(padding_mask, dim=1, index=sort_indices)

        for i, block in enumerate(self.MARTransformer):
            # Grab the weight for this specific transformer layer
            w = style_weight_schedule[i] if style_weight_schedule is not None else 1.0
            x = block(x, cond, padding_mask, c_style=c_style, style_weight=w)
            
        if mask is not None:
            x = torch.gather(x, dim=1, index=inverse_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return x

    # def forward_loss(self, latents, y, m_lens, style_cond=None, style_weight_schedule=None): 
    def forward_loss(self, latents, y, m_lens, raw_style_latents=None, style_weight_schedule=None):
        """
        Compute training loss for MARDM.
        
        Args:
            latents: Input motion latents [B, D, L]
            y: Text descriptions (list of strings)
            m_lens: Motion lengths [B]
            raw_style_latents: Raw style latents from DAE [B, 512, T] or None
            style_weight_schedule: Per-layer weights [num_blocks] or None
        
        Returns:
            loss: Scalar loss value
            full_pred: Predicted latents [B, D, L]
        """
        latents = latents.permute(0, 2, 1)
        b, l, d = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1), latents, torch.zeros_like(latents))

        target = latents.clone().detach()
        input = latents.clone()

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # Masking schedule (from MaskGIT)
        rand_time = uniform((b,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        # num_masked = (l * rand_mask_probs).round().clamp(min=1)
        num_masked = (m_lens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask
        
        # 10% random latents, 88% mask token, 2% unchanged
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        rand_latents = torch.randn_like(input)
        input = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input)
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        input = torch.where(mask_mlatents.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), input)

        # ==========================================
        # STYLE ENCODING
        # ==========================================
        style_cond = None
        if raw_style_latents is not None:
            style_cond = self.encode_style(raw_style_latents)

        # ==========================================
        # ROUTING LOGIC
        # ==========================================
        if self.style_routing == 'mart':
            # MART mode: style goes through MARTransformer
            z = self.forward(input, cond_vector, ~non_pad_mask, force_mask, 
                            c_style=style_cond, style_weight_schedule=style_weight_schedule)
        else:
            # DiffMLP mode: MARTransformer sees no style
            z = self.forward(input, cond_vector, ~non_pad_mask, force_mask, 
                            c_style=None, style_weight_schedule=None)

        # Reshape for DiffMLPs
        target = target.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        z = z.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        
        # Create repeated mask
        repeated_mask = mask.reshape(b * l).repeat(self.diffmlps_batch_mul)
        
        # Filter by mask
        target = target[repeated_mask]
        z = z[repeated_mask]

        # ==========================================
        # DIFFMLP FORWARD
        # ==========================================
        if self.style_routing == 'diffmlp' and style_cond is not None:
            # Expand style_cond to match sequence dimensions, then filter by mask
            c_style_expanded = style_cond.unsqueeze(1).expand(-1, l, -1).reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
            c_style_filtered = c_style_expanded[repeated_mask]
            
            # Pass schedule directly - no filtering needed!
            # The schedule is per-layer, not per-token
            loss, pred = self.DiffMLPs(z=z, target=target, 
                                    c_style=c_style_filtered, 
                                    style_weight_schedule=style_weight_schedule)
        else:
            # No style in DiffMLPs (either mart mode or no style provided)
            loss, pred = self.DiffMLPs(z=z, target=target, 
                                    c_style=None, 
                                    style_weight_schedule=None)

        # Reconstruct full prediction
        chunk_size = mask.sum()
        pred_xstart_single = pred[:chunk_size]
        
        full_pred = latents.clone().detach()
        full_pred[mask] = pred_xstart_single

        return loss, full_pred.permute(0, 2, 1), mask

    # =========================================================================
    # CYCLE CONSISTENCY LOSS FUNCTIONS (Phase 3)
    # =========================================================================
    
    def forward_cycle_loss_masked_only(self, z_content, y, m_lens, style_latents, 
                                        style_weight_schedule=None, detach_pass2=True):
        """
        Cycle consistency loss: z + style - style ≈ z
        Loss computed ONLY on masked (cycled) positions.
        
        Uses SAME mask for both passes to ensure fair comparison.
        
        Args:
            z_content: Content latents [B, D, L] (e.g., from HumanML3D)
            y: Text descriptions (list of strings)
            m_lens: Motion lengths [B]
            style_latents: Raw style latents from DAE [B, 512, T]
            style_weight_schedule: Per-layer weights [num_blocks] or None
            detach_pass2: Whether to detach pred_styled before Pass 2 (default: True)
        
        Returns:
            cycle_loss: Scalar MSE loss on masked positions only
            pred_styled: Styled prediction [B, D, L]
            pred_recovered: De-styled prediction [B, D, L]
            mask: Boolean mask [B, L] indicating which positions were cycled
        """
        # === Setup ===
        z_content = z_content.permute(0, 2, 1)  # [B, D, L] -> [B, L, D]
        b, l, d = z_content.shape
        device = z_content.device
        
        # Handle padding
        non_pad_mask = lengths_to_mask(m_lens, l)
        z_content = torch.where(non_pad_mask.unsqueeze(-1), z_content, torch.zeros_like(z_content))
        target = z_content.clone().detach()
        
        # === Generate mask ONCE (shared by both passes) ===
        rand_time = uniform((b,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        # num_masked = (l * rand_mask_probs).round().clamp(min=1)
        num_masked = (m_lens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask
        
        # === Prepare input with standard masking strategy ===
        # 10% random latents, 88% mask token, 2% unchanged (same as forward_loss)
        input_pass1 = z_content.clone()
        
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        rand_latents = torch.randn_like(input_pass1)
        input_pass1 = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input_pass1)
        
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        input_pass1 = torch.where(mask_mlatents.unsqueeze(-1), 
                                   self.mask_latent.repeat(b, l, 1), input_pass1)
        
        # === Encode conditioning ===
        with torch.no_grad():
            cond_vector = self.encode_text(y)
        style_cond = self.encode_style(style_latents)
        
        # === Prepare DiffMLPs inputs (shared setup) ===
        target_flat = target.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        repeated_mask = mask.reshape(b * l).repeat(self.diffmlps_batch_mul)
        chunk_size = mask.sum()
        
        # =========================================================================
        # PASS 1: WITH STYLE
        # =========================================================================
        if self.style_routing == 'mart':
            z1 = self.forward(input_pass1, cond_vector, ~non_pad_mask, mask=mask,
                              c_style=style_cond, style_weight_schedule=style_weight_schedule)
            diffmlp_style = None
            diffmlp_schedule = None
        else:
            z1 = self.forward(input_pass1, cond_vector, ~non_pad_mask, mask=mask,
                              c_style=None, style_weight_schedule=None)
            diffmlp_style = style_cond
            diffmlp_schedule = style_weight_schedule
        
        # DiffMLPs Pass 1
        z1_flat = z1.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        
        if diffmlp_style is not None:
            c_style_expanded = diffmlp_style.unsqueeze(1).expand(-1, l, -1)
            c_style_flat = c_style_expanded.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
            c_style_filtered = c_style_flat[repeated_mask]
        else:
            c_style_filtered = None
        
        _, pred_styled_flat = self.DiffMLPs(
            z=z1_flat[repeated_mask], 
            target=target_flat[repeated_mask],
            c_style=c_style_filtered,
            style_weight_schedule=diffmlp_schedule
        )
        
        # Reconstruct full pred_styled [B, L, D]
        pred_styled = z_content.clone().detach()
        pred_styled[mask] = pred_styled_flat[:chunk_size]
        
        # =========================================================================
        # PASS 2: WITHOUT STYLE (remove style)
        # =========================================================================
        if detach_pass2:
            input_pass2 = pred_styled.clone().detach()
        else:
            input_pass2 = pred_styled.clone()
        
        # Re-apply SAME masking pattern to pred_styled
        input_pass2 = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input_pass2)
        input_pass2 = torch.where(mask_mlatents.unsqueeze(-1),
                                   self.mask_latent.repeat(b, l, 1), input_pass2)
        
        # MARTransformer Pass 2 (NO style for either routing mode)
        z2 = self.forward(input_pass2, cond_vector, ~non_pad_mask, mask=mask,
                          c_style=None, style_weight_schedule=None)
        
        # DiffMLPs Pass 2 (NO style)
        z2_flat = z2.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        
        _, pred_recovered_flat = self.DiffMLPs(
            z=z2_flat[repeated_mask],
            target=target_flat[repeated_mask],
            c_style=None,
            style_weight_schedule=None
        )
        
        # Reconstruct full pred_recovered [B, L, D]
        pred_recovered = pred_styled.clone().detach()
        pred_recovered[mask] = pred_recovered_flat[:chunk_size]
        
        # =========================================================================
        # CYCLE LOSS: Only on masked positions
        # =========================================================================
        pred_recovered_chunk = pred_recovered_flat[:chunk_size]
        target_chunk = target_flat[repeated_mask][:chunk_size]
        
        cycle_loss = F.mse_loss(pred_recovered_chunk, target_chunk)
        
        return (
            cycle_loss, 
            pred_styled.permute(0, 2, 1),
            pred_recovered.permute(0, 2, 1),
            mask
        )


    def forward_cycle_loss_weighted(self, z_content, y, m_lens, style_latents, 
                                     style_weight_schedule=None, detach_pass2=True,
                                     focus_ratio=0.9):
        """
        Cycle consistency loss: z + style - style ≈ z
        Weighted loss on ALL positions:
        - Higher weight (focus_ratio) on masked/cycled positions
        - Lower weight (1-focus_ratio) on copied positions (sanity check)
        
        Uses SAME mask for both passes to ensure fair comparison.
        
        Args:
            z_content: Content latents [B, D, L] (e.g., from HumanML3D)
            y: Text descriptions (list of strings)
            m_lens: Motion lengths [B]
            style_latents: Raw style latents from DAE [B, 512, T]
            style_weight_schedule: Per-layer weights [num_blocks] or None
            detach_pass2: Whether to detach pred_styled before Pass 2 (default: True)
            focus_ratio: Weight for masked positions (default: 0.9)
        
        Returns:
            cycle_loss: Weighted MSE loss over all positions
            pred_styled: Styled prediction [B, D, L]
            pred_recovered: De-styled prediction [B, D, L]
            mask: Boolean mask [B, L]
        """
        # === Setup ===
        z_content = z_content.permute(0, 2, 1)
        b, l, d = z_content.shape
        device = z_content.device
        
        non_pad_mask = lengths_to_mask(m_lens, l)
        z_content = torch.where(non_pad_mask.unsqueeze(-1), z_content, torch.zeros_like(z_content))
        target = z_content.clone().detach()
        
        # === Generate mask ONCE ===
        rand_time = uniform((b,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        # num_masked = (l * rand_mask_probs).round().clamp(min=1)
        num_masked = (m_lens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask
        
        # === Prepare input ===
        input_pass1 = z_content.clone()
        
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        rand_latents = torch.randn_like(input_pass1)
        input_pass1 = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input_pass1)
        
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        input_pass1 = torch.where(mask_mlatents.unsqueeze(-1), 
                                   self.mask_latent.repeat(b, l, 1), input_pass1)
        
        # === Encode conditioning ===
        with torch.no_grad():
            cond_vector = self.encode_text(y)
        style_cond = self.encode_style(style_latents)
        
        # === Prepare DiffMLPs inputs ===
        target_flat = target.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        repeated_mask = mask.reshape(b * l).repeat(self.diffmlps_batch_mul)
        chunk_size = mask.sum()
        
        # =========================================================================
        # PASS 1: WITH STYLE
        # =========================================================================
        if self.style_routing == 'mart':
            z1 = self.forward(input_pass1, cond_vector, ~non_pad_mask, mask=mask,
                              c_style=style_cond, style_weight_schedule=style_weight_schedule)
            diffmlp_style = None
            diffmlp_schedule = None
        else:
            z1 = self.forward(input_pass1, cond_vector, ~non_pad_mask, mask=mask,
                              c_style=None, style_weight_schedule=None)
            diffmlp_style = style_cond
            diffmlp_schedule = style_weight_schedule
        
        z1_flat = z1.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        
        if diffmlp_style is not None:
            c_style_expanded = diffmlp_style.unsqueeze(1).expand(-1, l, -1)
            c_style_flat = c_style_expanded.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
            c_style_filtered = c_style_flat[repeated_mask]
        else:
            c_style_filtered = None
        
        _, pred_styled_flat = self.DiffMLPs(
            z=z1_flat[repeated_mask], 
            target=target_flat[repeated_mask],
            c_style=c_style_filtered,
            style_weight_schedule=diffmlp_schedule
        )
        
        pred_styled = z_content.clone().detach()
        pred_styled[mask] = pred_styled_flat[:chunk_size]
        
        # =========================================================================
        # PASS 2: WITHOUT STYLE
        # =========================================================================
        if detach_pass2:
            input_pass2 = pred_styled.clone().detach()
        else:
            input_pass2 = pred_styled.clone()
        
        input_pass2 = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input_pass2)
        input_pass2 = torch.where(mask_mlatents.unsqueeze(-1),
                                   self.mask_latent.repeat(b, l, 1), input_pass2)
        
        z2 = self.forward(input_pass2, cond_vector, ~non_pad_mask, mask=mask,
                          c_style=None, style_weight_schedule=None)
        
        z2_flat = z2.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        
        _, pred_recovered_flat = self.DiffMLPs(
            z=z2_flat[repeated_mask],
            target=target_flat[repeated_mask],
            c_style=None,
            style_weight_schedule=None
        )
        
        pred_recovered = pred_styled.clone().detach()
        pred_recovered[mask] = pred_recovered_flat[:chunk_size]
        
        # =========================================================================
        # WEIGHTED CYCLE LOSS
        # =========================================================================
        mask_expanded = mask.unsqueeze(-1).expand_as(target)
        non_pad_expanded = non_pad_mask.unsqueeze(-1).expand_as(target)
        
        weights = torch.zeros_like(target)
        weights[mask_expanded & non_pad_expanded] = focus_ratio
        weights[~mask_expanded & non_pad_expanded] = 1.0 - focus_ratio
        
        mse_per_element = (pred_recovered - target) ** 2
        weighted_mse = mse_per_element * weights
        cycle_loss = weighted_mse.sum() / weights.sum()
        
        return (
            cycle_loss, 
            pred_styled.permute(0, 2, 1),
            pred_recovered.permute(0, 2, 1),
            mask
        )

    def forward_with_CFG(self, latents, cond_vector, padding_mask,
                         cfg=3, cfg_mode='2way', cfg_text=None, cfg_style=None,
                         mask=None, force_mask=False, hard_pseudo_reorder=False,
                         raw_style_latents=None, style_weight_schedule=None):
        """
        Classifier-Free Guidance dispatcher supporting 2-way and 3-way CFG.

        cfg_mode:
            '2way'              — baseline: uncond + cfg * (cond - uncond)
            '3way_additive'     — uncond + cfg_text*(text_only - uncond)
                                          + cfg_style*(style_only - uncond)
            '3way_style_first'  — uncond + cfg_style*(style_only - uncond)
                                          + cfg_text*(full - style_only)

        If a 3-way mode is requested but `raw_style_latents is None` (no style
        available), silently fall back to 2-way with `cfg`.
        """
        if hard_pseudo_reorder:
            reorder_mask = mask.clone()
        else:
            reorder_mask = None

        # Style encoding
        style_cond = None
        if raw_style_latents is not None:
            style_cond = self.encode_style(raw_style_latents)

        if force_mask:
            c_style_pass = torch.zeros_like(style_cond) if style_cond is not None else None
            return self.forward(latents, cond_vector, padding_mask, force_mask=True,
                                mask=reorder_mask, c_style=c_style_pass)

        # Resolve effective CFG mode: 3-way needs style; otherwise fall back to 2-way
        use_3way = cfg_mode in ('3way_additive', '3way_style_first') and style_cond is not None

        # Style routed into MART only in 'mart' routing mode
        c_style_mart = style_cond if self.style_routing == 'mart' else None

        # Conditional MART pass (text present; style present if MART routing)
        z_cond = self.forward(latents, cond_vector, padding_mask, mask=reorder_mask,
                              c_style=c_style_mart, style_weight_schedule=style_weight_schedule)

        # ==========================================
        # Build n-branch z stack
        # ==========================================
        if use_3way:
            c_style_mart_uncond = torch.zeros_like(c_style_mart) if c_style_mart is not None else None
            z_uncond = self.forward(latents, cond_vector, padding_mask, force_mask=True,
                                    mask=reorder_mask, c_style=c_style_mart_uncond,
                                    style_weight_schedule=style_weight_schedule)
            if cfg_mode == '3way_additive':
                # B1=(no_text, no_style), B2=(text, no_style), B3=(no_text, style)
                mixed_logits = torch.cat([z_uncond, z_cond, z_uncond], dim=0)
            else:  # 3way_style_first
                # B1=(no_text, no_style), B2=(no_text, style), B3=(text, style)
                mixed_logits = torch.cat([z_uncond, z_uncond, z_cond], dim=0)
            n_branches = 3
        elif cfg != 1:
            c_style_mart_uncond = torch.zeros_like(c_style_mart) if c_style_mart is not None else None
            z_uncond = self.forward(latents, cond_vector, padding_mask, force_mask=True,
                                    mask=reorder_mask, c_style=c_style_mart_uncond,
                                    style_weight_schedule=style_weight_schedule)
            # 2-way convention: [cond, uncond]
            mixed_logits = torch.cat([z_cond, z_uncond], dim=0)
            n_branches = 2
        else:
            mixed_logits = z_cond
            n_branches = 1

        b, l, d = mixed_logits.size()  # b = n_branches * B

        # ==========================================
        # Flatten + mask-filter
        # ==========================================
        if mask is not None:
            if n_branches > 1:
                mask2 = torch.cat([mask] * n_branches, dim=0).reshape(b * l)
            else:
                mask2 = mask.reshape(b * l)
            mixed_logits_filtered = mixed_logits.reshape(b * l, d)[mask2]
        else:
            mask2 = None
            mixed_logits_filtered = mixed_logits.reshape(b * l, d)

        # ==========================================
        # Build n-branch style stack for DiffMLP routing
        # ==========================================
        c_style_filtered = None
        diffmlp_schedule = None
        if self.style_routing == 'diffmlp' and style_cond is not None:
            zero_style = torch.zeros_like(style_cond)
            if use_3way:
                if cfg_mode == '3way_additive':
                    stacked_style = torch.cat([zero_style, zero_style, style_cond], dim=0)
                else:  # 3way_style_first
                    stacked_style = torch.cat([zero_style, style_cond, style_cond], dim=0)
            elif cfg != 1:
                # 2-way: match [z_cond, z_uncond] order → [style, zero]
                stacked_style = torch.cat([style_cond, zero_style], dim=0)
            else:
                stacked_style = style_cond

            c_style_expanded = stacked_style.unsqueeze(1).expand(-1, l, -1).reshape(b * l, -1)
            c_style_filtered = c_style_expanded[mask2] if mask is not None else c_style_expanded
            diffmlp_schedule = style_weight_schedule

        # ==========================================
        # Dispatch to DiffMLPs with appropriate CFG mode
        # ==========================================
        if use_3way:
            output = self.DiffMLPs.sample(
                mixed_logits_filtered, temperature=1,
                cfg_mode=cfg_mode, cfg_text=cfg_text, cfg_style=cfg_style,
                c_style=c_style_filtered, style_weight_schedule=diffmlp_schedule,
            )
        else:
            output = self.DiffMLPs.sample(
                mixed_logits_filtered, temperature=1, cfg=cfg,
                c_style=c_style_filtered, style_weight_schedule=diffmlp_schedule,
            )

        # All n_branches chunks are identical after CFG; take the first
        if n_branches > 1:
            scaled_logits = output.chunk(n_branches, dim=0)[0]
        else:
            scaled_logits = output

        # ==========================================
        # Reshape back to [B, L, D]
        # ==========================================
        if mask is not None:
            flat_len = (b // n_branches) * l
            latents = latents.reshape(flat_len, self.ae_dim)
            latents[mask.reshape(flat_len)] = scaled_logits
            scaled_logits = latents.reshape(b // n_branches, l, self.ae_dim)

        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self, conds, m_lens, timesteps=18, cond_scale=3, temperature=1, hard_pseudo_reorder=True,
                 raw_style_latents=None, style_weight_schedule=None, force_mask=False,
                 cfg_mode='2way', cfg_text=None, cfg_style=None):
        device = next(self.parameters()).device
        b = len(conds)
        l = max(m_lens) if isinstance(m_lens, list) else m_lens.max().item()

        padding_mask = ~lengths_to_mask(m_lens, l)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # mask_latent init for non-padding; zeros for padding
        latents = torch.where(padding_mask.unsqueeze(-1),
                              torch.zeros(b, l, self.ae_dim, device=device),
                              self.mask_latent.repeat(b, l, 1))
        masked_rand_schedule = torch.where(padding_mask, 1e5,
                                           torch.rand_like(padding_mask, dtype=torch.float))

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = cosine_schedule(timestep)  # starts ~1.0, decays toward 0
            num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents)
            logits = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                           cfg=cond_scale, cfg_mode=cfg_mode,
                                           cfg_text=cfg_text, cfg_style=cfg_style,
                                           mask=is_mask, force_mask=force_mask,
                                           hard_pseudo_reorder=hard_pseudo_reorder,
                                           raw_style_latents=raw_style_latents,
                                           style_weight_schedule=style_weight_schedule)
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)
            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        return latents.permute(0, 2, 1)

    @torch.no_grad()
    @eval_decorator
    def edit(self, conds, latents, m_lens, timesteps=18, cond_scale=3, temperature=1,
             force_mask=False, edit_mask=None, padding_mask=None, hard_pseudo_reorder=True,
             raw_style_latents=None, style_weight_schedule=None,
             cfg_mode='2way', cfg_text=None, cfg_style=None):
        """Region editing / inpainting. Replaces only edit_mask positions; preserves the rest."""
        device = next(self.parameters()).device
        b = len(conds)
        l = latents.shape[-1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask is None:
            padding_mask = ~lengths_to_mask(m_lens, l)

        # Permute input latents from [B, D, L] to [B, L, D]
        latents = latents.permute(0, 2, 1)
        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)

        # Constrain edit_mask to non-padding positions
        edit_mask = edit_mask & ~padding_mask
        edit_len = edit_mask.sum(dim=-1)

        # Fill edit region with mask tokens
        latents = torch.where(edit_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents)
        masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents)
            logits = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                           cfg=cond_scale, cfg_mode=cfg_mode,
                                           cfg_text=cfg_text, cfg_style=cfg_style,
                                           mask=is_mask, force_mask=force_mask,
                                           hard_pseudo_reorder=hard_pseudo_reorder,
                                           raw_style_latents=raw_style_latents,          # FIX: was style_cond=style_cond
                                           style_weight_schedule=style_weight_schedule)
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)
            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        return latents.permute(0, 2, 1)

#################################################################################
#                                     MARDM Zoos                                #
#################################################################################
def mardm_ddpm_xl(**kwargs):
    return MARDM(latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16, dropout=0.2, clip_dim=512,
                 diffmlps_model="DDPM-XL", diffmlps_batch_mul=4, cond_drop_prob=0.1, **kwargs)
def mardm_sit_xl(**kwargs):
    return MARDM(latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16, dropout=0.2, clip_dim=512,
                 diffmlps_model="SiT-XL", diffmlps_batch_mul=4, cond_drop_prob=0.1, **kwargs)

MARDM_models = {
    'MARDM-DDPM-XL': mardm_ddpm_xl, 'MARDM-SiT-XL': mardm_sit_xl,
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def modulate_here(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.2):
        super().__init__()
        assert embed_dim % 8 == 0
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, mask):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            mask = mask[:, None, None, :]
            att = att.masked_fill(mask != 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class MARTransBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size=1024, drop_out=0.2, style_dim=1024): # NEW
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, drop_out_rate=drop_out)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = mlp_size
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # NEW: Parallel Dual-AdaLN for Style
        self.style_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, padding_mask=None, c_style=None, style_weight=1.0): # NEW ARGUMENT
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # NEW: The Residual Dual-AdaLN Injection
        if c_style is not None:
            shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s = self.style_modulation(c_style).chunk(6, dim=1)
            
            # # Additive
            # shift_msa = shift_msa + (style_weight * shift_msa_s)
            # scale_msa = scale_msa * (1 + (style_weight * scale_msa_s))
            # gate_msa = gate_msa + (style_weight * gate_msa_s)
            
            # shift_mlp = shift_mlp + (style_weight * shift_mlp_s)
            # scale_mlp = scale_mlp * (1 + (style_weight * scale_mlp_s))
            # gate_mlp = gate_mlp + (style_weight * gate_mlp_s)

            # Additive 
            shift_msa = shift_msa + (style_weight * shift_msa_s)
            gate_msa = gate_msa + (style_weight * gate_msa_s)
            shift_mlp = shift_mlp + (style_weight * shift_mlp_s)
            gate_mlp = gate_mlp + (style_weight * gate_mlp_s)
            
            # Multiplicative with sigmoid 
            scale_msa = scale_msa * (2.0 * torch.sigmoid(style_weight * scale_msa_s))
            scale_mlp = scale_mlp * (2.0 * torch.sigmoid(style_weight * scale_mlp_s))

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate_here(self.norm1(x), shift_msa, scale_msa), mask=padding_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate_here(self.norm2(x), shift_mlp, scale_mlp))
        
        return x