import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from diffusions.diffusion import create_diffusion
from diffusions.transport import create_transport, Sampler

#################################################################################
#                                     DiffMLPs                                  #
#################################################################################
class DiffMLPs_DDPM(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, learn_sigma=False):
        super(DiffMLPs_DDPM, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2 if learn_sigma else target_channels,
            z_channels=z_channels,
            style_dim=512, # NEW: Added Style Dim
            num_res_blocks=depth,
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, c_style=None, mask=None, style_weight_schedule=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        
        # ADDED TO KWARGS HERE
        model_kwargs = dict(c=z, c_style=c_style, style_weight_schedule=style_weight_schedule) 
        
        # This function passes the kwargs directly into SimpleMLPAdaLN.forward()
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        pred = loss_dict["pred_xstart"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean(), pred

    def sample(self, z, c_style=None, temperature=1.0, cfg=1.0,
               cfg_mode='2way', cfg_text=None, cfg_style=None,
               style_weight_schedule=None):
        device = z.device
        if cfg_mode in ('3way_additive', '3way_style_first'):
            # z is pre-stacked as 3 branches along batch dim by the caller.
            noise = torch.randn(z.shape[0] // 3, self.in_channels).to(device)
            noise = torch.cat([noise, noise, noise], dim=0)

            mode_tag = cfg_mode.replace('3way_', '')
            model_kwargs = dict(
                c=z, cfg_text=cfg_text, cfg_style=cfg_style, mode=mode_tag,
                c_style=c_style, style_weight_schedule=style_weight_schedule,
            )
            sample_fn = self.net.forward_with_cfg_3way
        elif not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(device)
            noise = torch.cat([noise, noise], dim=0)

            # ADDED TO KWARGS HERE
            model_kwargs = dict(c=z, cfg_scale=cfg, c_style=c_style, style_weight_schedule=style_weight_schedule)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(device)

            # ADDED TO KWARGS HERE
            model_kwargs = dict(c=z, c_style=c_style, style_weight_schedule=style_weight_schedule)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent


    def get_total_blocks(self):
        """Return total number of blocks for schedule sizing."""
        return self.net.get_total_blocks()

class DiffMLPs_SiT(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width):
        super(DiffMLPs_SiT, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            style_dim=512, # NEW: Added Style Dim
            num_res_blocks=depth,
        )

        self.train_diffusion = create_transport() 
        self.gen_diffusion = Sampler(self.train_diffusion)

    def forward(self, target, z, c_style=None, mask=None, style_weight_schedule=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        
        # ADDED TO KWARGS HERE
        model_kwargs = dict(c=z, c_style=c_style, style_weight_schedule=style_weight_schedule) 
        
        # This function passes the kwargs directly into SimpleMLPAdaLN.forward()
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        pred = loss_dict["pred_xstart"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean(), pred

    def sample(self, z, c_style=None, temperature=1.0, cfg=1.0,
               cfg_mode='2way', cfg_text=None, cfg_style=None,
               style_weight_schedule=None):
        device = z.device
        if cfg_mode in ('3way_additive', '3way_style_first'):
            noise = torch.randn(z.shape[0] // 3, self.in_channels).to(device)
            noise = torch.cat([noise, noise, noise], dim=0)

            mode_tag = cfg_mode.replace('3way_', '')
            model_kwargs = dict(
                c=z, cfg_text=cfg_text, cfg_style=cfg_style, mode=mode_tag,
                c_style=c_style, style_weight_schedule=style_weight_schedule,
            )
            sample_fn = self.net.forward_with_cfg_3way
        elif not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(device)
            noise = torch.cat([noise, noise], dim=0)

            # ADDED TO KWARGS HERE
            model_kwargs = dict(c=z, cfg_scale=cfg, c_style=c_style, style_weight_schedule=style_weight_schedule)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(device)

            # ADDED TO KWARGS HERE
            model_kwargs = dict(c=z, c_style=c_style, style_weight_schedule=style_weight_schedule)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent

    def get_total_blocks(self):
        """Return total number of blocks for schedule sizing."""
        return self.net.get_total_blocks()

#################################################################################
#                                  DiffMLPs Zoos                                #
#################################################################################
def diffmlps_ddpm_xl(**kwargs):
    return DiffMLPs_DDPM(depth=16, width=1792, num_sampling_steps="50", learn_sigma=False, **kwargs)
def diffmlps_sit_xl(**kwargs):
    return DiffMLPs_SiT(depth=16, width=1792, **kwargs)

DiffMLPs_models = {
    'DDPM-XL': diffmlps_ddpm_xl, 'SiT-XL': diffmlps_sit_xl,
}

#################################################################################
#                                Inner Architectures                            #
#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        style_dim=1024 # NEW
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )
        
        # NEW: Parallel Dual-AdaLN for Style
        self.style_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim, 3 * channels, bias=True)
        )

    def forward(self, x, y, c_style=None, style_weight=1.0): 
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        
        if c_style is not None:
            shift_s, scale_s, gate_s = self.style_modulation(c_style).chunk(3, dim=-1)
            
            # APPLIED THE SCHEDULE MULTIPLIER
            shift_mlp = shift_mlp + (style_weight * shift_s)
            # scale_mlp = scale_mlp * (1 + (style_weight * scale_s))
            gate_mlp = gate_mlp + (style_weight * gate_s) 

            # Multiplicative with sigmoid (NEW)
            scale_mult = 2.0 * torch.sigmoid(style_weight * scale_s)
            scale_mlp = scale_mlp * scale_mult

        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels, style_dim=1024): # NEW
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )
        # NEW: Parallel Dual-AdaLN for Style (No gate here)
        self.style_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim, 2 * model_channels, bias=True)
        )

    def forward(self, x, c, c_style=None, style_weight=1.0):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        
        if c_style is not None:
            shift_s, scale_s = self.style_modulation(c_style).chunk(2, dim=-1)
            shift = shift + (style_weight * shift_s)
            # scale = scale * (1 + (style_weight * scale_s))

            # Sigmoid-based scaling
            scale_mult = 2.0 * torch.sigmoid(style_weight * scale_s)
            scale = scale * scale_mult


        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        style_dim=512,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_style_blocks = num_res_blocks // 2
        self.grad_checkpointing = grad_checkpointing
        self.style_dim = style_dim

        # Embeddings
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        # Input projection
        self.input_proj = nn.Linear(in_channels, model_channels)

        # 1. Original content blocks (e.g., 16)
        self.res_blocks = nn.ModuleList([
            ResBlock(model_channels, style_dim=style_dim) 
            for _ in range(self.num_res_blocks)
        ])

        # 2. Additional style-focused blocks (e.g., 8)
        self.style_blocks = nn.ModuleList([
            ResBlock(model_channels, style_dim=style_dim) 
            for _ in range(self.num_style_blocks)
        ])

        # 3. Final layer
        self.final_layer = FinalLayer(model_channels, out_channels, style_dim=style_dim)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize all weights with proper zero-init for modulation layers."""
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Time embedding
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-init for res_blocks
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.style_modulation[-1].weight, 0)
            nn.init.constant_(block.style_modulation[-1].bias, 0)

        # Zero-init for style_blocks
        for block in self.style_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.style_modulation[-1].weight, 0)
            nn.init.constant_(block.style_modulation[-1].bias, 0)

        # Final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.style_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.style_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_total_blocks(self):
        """Return total number of blocks for schedule sizing."""
        return self.num_res_blocks + self.num_style_blocks  # e.g., 16 + 8 = 24

    def forward(self, x, t, c, c_style=None, style_weight_schedule=None):
        """
        Forward pass through all blocks.
        
        Args:
            x: Input tensor [B, in_channels]
            t: Timestep tensor [B]
            c: Conditioning from MARTransformer [B, z_channels]
            c_style: Style conditioning [B, style_dim] or None
            style_weight_schedule: List of length (num_res_blocks + num_style_blocks)
                                   e.g., [0.0, 0.04, ..., 1.0] for 24 blocks
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        # ============================================================
        # Phase 1: Content blocks (res_blocks) - indices 0 to 15
        # ============================================================
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for i, block in enumerate(self.res_blocks):
                w = style_weight_schedule[i] if style_weight_schedule is not None else 1.0
                x = checkpoint(
                    lambda x_in, y_in, c_style_in, w_in: block(x_in, y_in, c_style=c_style_in, style_weight=w_in),
                    x, y, c_style, w,
                    use_reentrant=False
                )
        else:
            for i, block in enumerate(self.res_blocks):
                w = style_weight_schedule[i] if style_weight_schedule is not None else 1.0
                x = block(x, y, c_style=c_style, style_weight=w)

        # ============================================================
        # Phase 2: Style blocks (style_blocks) - indices 16 to 23
        # ============================================================
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for i, block in enumerate(self.style_blocks):
                idx = self.num_res_blocks + i  # Offset: 16, 17, 18, ...
                w = style_weight_schedule[idx] if style_weight_schedule is not None else 1.0
                x = checkpoint(
                    lambda x_in, y_in, c_style_in, w_in: block(x_in, y_in, c_style=c_style_in, style_weight=w_in),
                    x, y, c_style, w,
                    use_reentrant=False
                )
        else:
            for i, block in enumerate(self.style_blocks):
                idx = self.num_res_blocks + i  # Offset: 16, 17, 18, ...
                w = style_weight_schedule[idx] if style_weight_schedule is not None else 1.0
                x = block(x, y, c_style=c_style, style_weight=w)

        # ============================================================
        # Final Layer - uses last weight in schedule
        # ============================================================
        final_w = style_weight_schedule[-1] if style_weight_schedule is not None else 1.0
        return self.final_layer(x, y, c_style=c_style, style_weight=final_w)

    def forward_with_cfg(self, x, t, c, cfg_scale, c_style=None, style_weight_schedule=None):
        """
        Forward pass with Classifier-Free Guidance.

        The batch is structured as [conditioned, unconditioned].
        CFG formula: output = uncond + cfg_scale * (cond - uncond)
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        model_out = self.forward(
            combined, t, c,
            c_style=c_style,
            style_weight_schedule=style_weight_schedule
        )

        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        return torch.cat([eps, rest], dim=1)

    def forward_with_cfg_3way(self, x, t, c, cfg_text, cfg_style, mode,
                              c_style=None, style_weight_schedule=None):
        """
        Forward pass with 3-way Classifier-Free Guidance.

        Batch is structured as three equal thirds [B1, B2, B3]:
            additive:    B1=(no_text, no_style), B2=(text, no_style), B3=(no_text, style)
            style_first: B1=(no_text, no_style), B2=(no_text, style),  B3=(text, style)

        Formulas:
            additive:    out = B1 + cfg_text  * (B2 - B1) + cfg_style * (B3 - B1)
            style_first: out = B1 + cfg_style * (B2 - B1) + cfg_text  * (B3 - B2)

        The three-branch conditioning is supplied via `c` (pre-stacked by the caller)
        and `c_style` (pre-stacked by the caller). Noise is identical across the three
        thirds, matching the 2-way convention.
        """
        third_len = len(x) // 3
        third = x[:third_len]
        combined = torch.cat([third, third, third], dim=0)

        model_out = self.forward(
            combined, t, c,
            c_style=c_style,
            style_weight_schedule=style_weight_schedule
        )

        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps_b1, eps_b2, eps_b3 = torch.split(eps, len(eps) // 3, dim=0)

        if mode == 'additive':
            guided = eps_b1 + cfg_text * (eps_b2 - eps_b1) + cfg_style * (eps_b3 - eps_b1)
        elif mode == 'style_first':
            guided = eps_b1 + cfg_style * (eps_b2 - eps_b1) + cfg_text * (eps_b3 - eps_b2)
        else:
            raise ValueError(f"Unknown 3-way CFG mode: {mode}")

        eps = torch.cat([guided, guided, guided], dim=0)
        return torch.cat([eps, rest], dim=1)