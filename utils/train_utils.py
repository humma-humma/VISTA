import torch
import math
import time

#################################################################################
#                                  Util Functions                               #
#################################################################################
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)


def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def update_ema(model, ema_model, ema_decay):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=(1 - ema_decay))


def ema_decay_warmup(it, warmup_iters, max_decay=0.9999):
    if it >= warmup_iters:
        return max_decay
    return min(max_decay, (1 + it) / (10 + it))


def build_training_schedule(total_epochs, steps_per_epoch):
    total_iters = total_epochs * steps_per_epoch
    return {
        'total_iters':         total_iters,
        'warm_up_iter':        max(500, int(0.05 * total_iters)),
        'lr_milestones':       [int(0.8 * total_iters)],
        'ema_warmup_iters':    max(1000, int(0.02 * total_iters)),
        'ema_max_decay':       0.9999,
    }


STYLE_PARAM_SUBSTRINGS = ('style_blocks', 'style_proj', 'style_modulation')


def reset_ema_style_params(ema_model, model):
    """Overwrite EMA slots for style-only params with the online weights.

    Preserves EMA averaging on the base pathway while wiping out the
    random-init contamination on the fast-moving style modules.
    """
    with torch.no_grad():
        n_reset = 0
        for (name, ema_p), (_, p) in zip(
            ema_model.named_parameters(), model.named_parameters()
        ):
            if any(tag in name for tag in STYLE_PARAM_SUBSTRINGS):
                ema_p.data.copy_(p.data)
                n_reset += 1
    return n_reset

#################################################################################
#                                Logging Functions                              #
#################################################################################
def def_value():
    return 0.0


def update_lr_warm_up(nb_iter, warm_up_iter, optimizer, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return current_lr

def save_lora(file_name, ep, model, optimizer, scheduler, total_it, name, ema_mardm=None, save_lora_only=True):
    """
    Save model checkpoint.
    
    Args:
        file_name: Path to save the checkpoint
        ep: Current epoch
        model: The model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        total_it: Total iterations
        name: Name key for the model in state dict
        ema_mardm: Optional EMA model
        save_lora_only: If True, only save LoRA + new trainable params (much smaller file)
    """
    
    def filter_state_dict(state_dict, lora_only=False):
        """Filter state dict to remove CLIP weights and optionally keep only LoRA params."""
        filtered = {}
        for k, v in state_dict.items():
            # Always skip CLIP weights
            if k.startswith('clip_model.'):
                continue
            
            if lora_only:
                # Keep only LoRA params + new trainable layers + mask_latent
                if any(pattern in k.lower() for pattern in ['lora_module', 'lora.']):
                    filtered[k] = v
                elif 'adaLN_style' in k or 'video_proj' in k:
                    filtered[k] = v
                elif k == 'mask_latent':
                    filtered[k] = v
            else:
                # Keep everything except CLIP
                filtered[k] = v
        
        return filtered
    
    # Get state dicts
    model_state_dict = model.state_dict()
    
    if save_lora_only:
        # Save only LoRA and new trainable parameters
        filtered_model_state = filter_state_dict(model_state_dict, lora_only=True)
        
        state = {
            name: filtered_model_state,
            f"opt_{name}": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'lora_only': True,  # Flag to indicate this is a LoRA-only checkpoint
            'lora_rank': model.lora_rank if hasattr(model, 'lora_rank') else None,
        }
        
        if ema_mardm is not None:
            ema_state_dict = ema_mardm.state_dict()
            filtered_ema_state = filter_state_dict(ema_state_dict, lora_only=True)
            state["ema_mardm"] = filtered_ema_state
        
        print(f"Saving LoRA-only checkpoint: {len(filtered_model_state)} params")
        
    else:
        # Save full model (original behavior)
        filtered_model_state = filter_state_dict(model_state_dict, lora_only=False)
        
        state = {
            name: filtered_model_state,
            f"opt_{name}": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'lora_only': False,
        }
        
        if ema_mardm is not None:
            ema_state_dict = ema_mardm.state_dict()
            filtered_ema_state = filter_state_dict(ema_state_dict, lora_only=False)
            state["ema_mardm"] = filtered_ema_state
    
    torch.save(state, file_name)


def save(file_name, ep, model, optimizer, scheduler, total_it, name, ema_mardm=None):
    state = {
        name: model.state_dict(),
        f"opt_{name}": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        'ep': ep,
        'total_it': total_it,
    }
    if ema_mardm is not None:
        mardm_state_dict = model.state_dict()
        ema_mardm_state_dict = ema_mardm.state_dict()
        clip_weights = [e for e in mardm_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del mardm_state_dict[e]
            del ema_mardm_state_dict[e]
        state[name] = mardm_state_dict
        state["ema_mardm"] = ema_mardm_state_dict
    torch.save(state, file_name)

def savediff(file_name, ep, model, optimizer, scheduler, total_it, name, args, ema_mardm=None, weight_schedule=None):
    state = {
        name: model.state_dict(),
        f"opt_{name}": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        'ep': ep,
        'total_it': total_it,
        'style_routing': args.style_routing,
        'weight_schedule': weight_schedule,
    }
    
    if ema_mardm is not None:
        mardm_state_dict = model.state_dict()
        ema_mardm_state_dict = ema_mardm.state_dict()
        clip_weights = [e for e in mardm_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del mardm_state_dict[e]
            del ema_mardm_state_dict[e]
        state[name] = mardm_state_dict
        state["ema_mardm"] = ema_mardm_state_dict
    torch.save(state, file_name)

def save_mld(file_name, ep, model, optimizer, scheduler, total_it, name, ema_mld=None):
    state = {
        name: model.state_dict(),
        f"opt_{name}": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        'ep': ep,
        'total_it': total_it,
    }
    if ema_mld is not None:
        mld_state_dict = model.state_dict()
        ema_mld_state_dict = ema_mld.state_dict()
        clip_weights = [e for e in mld_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del mld_state_dict[e]
            del ema_mld_state_dict[e]
        state[name] = mld_state_dict
        state["ema_mld"] = ema_mld_state_dict
    torch.save(state, file_name)



def save_upd(file_name, ep, model, optimizer, scheduler, total_it, name, style_classifier=None, ema_mardm=None):
    state = {
        name: model.state_dict(),
        f"opt_{name}": optimizer.state_dict(), # This now saves optimizer state for BOTH if you combined them
        "scheduler": scheduler.state_dict(),
        'ep': ep,
        'total_it': total_it,
    }
    
    # --- UPDATE START: Save the Style Classifier ---
    if style_classifier is not None:
        state['style_classifier'] = style_classifier.state_dict()
    # --- UPDATE END ---

    if ema_mardm is not None:
        mardm_state_dict = model.state_dict()
        ema_mardm_state_dict = ema_mardm.state_dict()
        clip_weights = [e for e in mardm_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del mardm_state_dict[e]
            del ema_mardm_state_dict[e]
        state[name] = mardm_state_dict
        state["ema_mardm"] = ema_mardm_state_dict
        
    torch.save(state, file_name)


def save_disc_upd(file_name, ep, model, optimizer, scheduler, total_it, name, style_classifier=None, ema_mardm=None, discriminator=None, opt_disc=None):
    
    state = {
        name: model.state_dict(),
        f"opt_{name}": optimizer.state_dict(), 
        "scheduler": scheduler.state_dict(),
        'ep': ep,
        'total_it': total_it,
    }
    
    # --- Save Style Classifier (Existing) ---
    if style_classifier is not None:
        state['style_classifier'] = style_classifier.state_dict()

    # --- UPDATE START: Save Discriminator & Its Optimizer ---
    if discriminator is not None:
        state['discriminator'] = discriminator.state_dict()
        
    if opt_disc is not None:
        state['opt_disc'] = opt_disc.state_dict()
    # --- UPDATE END ---

    # --- Save EMA Model (Existing) ---
    if ema_mardm is not None:
        mardm_state_dict = model.state_dict()
        ema_mardm_state_dict = ema_mardm.state_dict()
        # Remove CLIP weights if they exist (optimization)
        clip_weights = [e for e in mardm_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del mardm_state_dict[e]
            del ema_mardm_state_dict[e]
        state[name] = mardm_state_dict
        state["ema_mardm"] = ema_mardm_state_dict
        
    torch.save(state, file_name)


def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):
    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
    if epoch is not None:
        print('ep/it:%2d-%4d niter:%6d' % (epoch, inner_iter, niter_state), end=" ")
    message = ' %s completed:%3d%%)' % (time_since(start_time, niter_state / total_niters), niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)