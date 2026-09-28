import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path
import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import join as pjoin

# Assuming your model definitions are accessible
from models.AE import AE_models 

def preencode_latents_sliced(
    data_dir, text_dir, split_dir, output_dir,
    output_latent_dir, output_text_dir, output_split_dir,
    checkpoint_path, mean_path, std_path, fps=20.0, device="cuda"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_latent_dir, exist_ok=True)
    os.makedirs(output_text_dir, exist_ok=True)
    os.makedirs(output_split_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. Load HumanML3D AE
    model = AE_models['AE_Model'](input_width=67).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['ae'])
    model.eval()

    mean = np.load(mean_path)[:67]
    std = np.load(std_path)[:67]

    motion_files = glob(pjoin(data_dir, "*.npy"))
    print(f"Found {len(motion_files)} files. Slicing and Encoding...")

    # Keep track of which new slices belong to which original motion
    original_to_slices = {}

    # with open(pjoin(output_latent_dir, "all_lengths.txt"), "w", encoding='utf-8') as len_file:
    # train/eval read <output_split_dir>/all_lengths.txt
    with open(pjoin(output_split_dir, "all_lengths.txt"), "w", encoding='utf-8') as len_file:
        with torch.no_grad():
            for fpath in tqdm(motion_files):
                name = os.path.basename(fpath).replace(".npy", "")
                text_path = pjoin(text_dir, f"{name}.txt")
                
                if not os.path.exists(text_path):
                    continue

                # Load raw motion
                motion = np.load(fpath)[:, :67]
                real_len = motion.shape[0]

                # Parse text file
                with open(text_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                slice_names = []
                for idx, line in enumerate(lines):
                    parts = line.strip().split('#')
                    caption = parts[0]
                    
                    start_frame = 0
                    end_frame = real_len

                    # HumanML3D standard format: caption#tokens#start_time#end_time
                    if len(parts) >= 4:
                        try:
                            f_tag = float(parts[2])
                            to_tag = float(parts[3])
                            
                            # Handle NaNs and 0.0s just like your snippet
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            
                            if f_tag != 0.0 or to_tag != 0.0:
                                # If values are in seconds, convert to frames (HumanML3D is 20fps)
                                start_frame = int(f_tag * fps) if f_tag < 100 else int(f_tag)
                                end_frame = int(to_tag * fps) if to_tag < 100 else int(to_tag)
                        except ValueError:
                            # Fallback if there's a weird parsing error on a specific line
                            pass
                            
                    # Safe boundaries
                    safe_start = max(0, min(start_frame, real_len - 1))
                    safe_end = max(safe_start + 1, min(end_frame, real_len))
                    
                    # Calculate segment length
                    seg_len = safe_end - safe_start

                    
                    # Skip slices that are too short for the Autoencoder or too long
                    if seg_len < 40 or seg_len > 200:
                        continue


                    # 1. Slice Raw Motion
                    sliced_motion = motion[safe_start:safe_end]
                    slice_name = f"{name}_{idx}"
                    np.save(pjoin(output_dir, f"{slice_name}.npy"), sliced_motion)
                    
                    # 2. Z-Normalize
                    sliced_motion = (sliced_motion - mean) / std
                    
                    # 3. Encode to Latent
                    motion_tensor = torch.from_numpy(sliced_motion).float().unsqueeze(0).to(device)
                    latent = model.encode(motion_tensor)
                    
                    # 4. Save Latent
                    latent_np = latent.squeeze(0).cpu().numpy()
                    slice_name = f"{name}_{idx}"
                    slice_names.append(slice_name)
                    np.save(pjoin(output_latent_dir, f"{slice_name}.npy"), latent_np)
                    
                    # 5. Save single text caption
                    with open(pjoin(output_text_dir, f"{slice_name}.txt"), "w", encoding='utf-8') as out_txt:
                        out_txt.write(caption)
                        
                    # 6. Track length (Number of latent frames, NOT raw frames)
                    len_file.write(f"{slice_name} {latent_np.shape[-1]}\n")
                
                original_to_slices[name] = slice_names

    # ==========================================
    # 7. Update Split Files (train.txt / val.txt)
    # ==========================================
    print("Updating split files to match new sliced names...")
    for split_name in ['train.txt', 'val.txt', 'test.txt']:
        split_path = pjoin(split_dir, split_name)
        if not os.path.exists(split_path):
            continue
            
        with open(split_path, 'r', encoding='utf-8') as f:
            original_names = [line.strip() for line in f.readlines()]
            
        new_split_path = pjoin(output_split_dir, split_name)
        with open(new_split_path, 'w', encoding='utf-8') as f:
            for orig_name in original_names:
                if orig_name in original_to_slices:
                    for slice_name in original_to_slices[orig_name]:
                        f.write(f"{slice_name}\n")
                        
    print("Finished successfully! Your data is now fully pre-sliced and modular.")

if __name__ == "__main__":
    # Example Usage for HumanML3D
    preencode_latents_sliced(
        data_dir="./datasets/HumanML3D/new_joint_vecs",
        text_dir="./datasets/HumanML3D/texts",
        split_dir="./datasets/HumanML3D/",
        
        # New Output Directories to prevent overwriting your old data
        output_dir='./datasets/HumanML3D/sliced_joint_vecs',
        output_latent_dir="./datasets/HumanML3D/latent_vecs",
        # output_text_dir="./datasets/HumanML3D/texts_sliced",
        output_text_dir="./datasets/HumanML3D/splits_sliced/texts_sliced",  # path read by train/eval
        output_split_dir="./datasets/HumanML3D/splits_sliced",
        
        checkpoint_path="./checkpoints/t2m/AE/model/latest.tar",
        mean_path="./datasets/HumanML3D/Mean.npy",
        std_path="./datasets/HumanML3D/Std.npy",
    )