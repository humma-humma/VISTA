from os.path import join as pjoin
import torch
from torch.utils import data
from rich.progress import track
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
import random
import codecs as cs
from utils.glove import GloVe
import os
from decord import VideoReader, cpu
import glob
from utils.profiling import timer
from typing import Dict, List, Tuple, Optional
import string

#################################################################################
#                                  Collate Functions                            #
#################################################################################
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def video_collate_fn(batch):
    # Separate the components of the batch
    motions, videos, labels = zip(*batch)

    # Collate motions and labels using the default behavior
    motion_tensor = default_collate(motions)
    label_tensor = default_collate(labels)

    # Keep videos as a list of lists of frames (don't collate them into a tensor)
    # The `videos` variable is already in the correct format: a tuple of video clips,
    # where each clip is a numpy array of frames. We just convert it to a list.
    video_list = list(videos)

    return motion_tensor, video_list, label_tensor

def t2m_collate_fn(batch):
    # 1. Sort by Motion Length (Index 2), not Index 3
    # Index 3 is now the video list, which is heavy and unsortable.
    batch.sort(key=lambda x: x[2], reverse=True)
    
    # 2. Unpack the 4 components
    texts, motions, m_lengths, styles, videos = zip(*batch)

    # 3. Stack Tensors
    # 'default_collate' handles stacking numpy arrays into tensors automatically
    motion_tensor = default_collate(motions)
    m_length_tensor = default_collate(m_lengths)

    # 4. Handle Lists
    # Keep videos as a list of lists (don't stack, as per your previous logic)
    video_list = list(videos)
    # Keep texts as a simple tuple or list (default_collate can handle strings, but explict list is safer)
    text_list = list(texts)
    style_list = list(styles)

    return text_list, motion_tensor, m_length_tensor, style_list, video_list

def t2m_collate_fn_v2(batch):
    """
    Collate function for Text2MotionVideoDataset_style00styles_v3.
    
    Input batch format (per sample):
        (text_data, neutral_motion, styled_motion, m_length, style_name, video_snippet)
    
    Output format:
        (text_list, neutral_motion_tensor, styled_motion_tensor, m_length_tensor, style_list, video_list)
    """
    # 1. Sort by motion length (index 3), descending
    batch.sort(key=lambda x: x[3], reverse=True)
    
    # 2. Unpack the 6 components
    texts, neutral_motions, styled_motions, m_lengths, styles, videos = zip(*batch)
    
    # 3. Stack motion tensors
    neutral_motion_tensor = default_collate(neutral_motions)  # [B, max_len, dim_pose]
    styled_motion_tensor = default_collate(styled_motions)    # [B, max_len, dim_pose]
    m_length_tensor = default_collate(m_lengths)              # [B]
    
    # 4. Handle lists (don't stack)
    text_list = list(texts)
    style_list = list(styles)
    video_list = list(videos)
    
    return text_list, neutral_motion_tensor, styled_motion_tensor, m_length_tensor, style_list, video_list

def eval_collate_fn(batch):
    # Sort by Motion Length (Index 5) 
    # (Or Index 3 'sent_len' if using a text encoder that needs packed sequences)
    batch.sort(key=lambda x: x[5], reverse=True)
    
    # Unpack all 8 items
    word_embs, pos_ohs, texts, sent_lens, motions, m_lens, tokens, styles, videos = zip(*batch)

    # Stack the mathematical components
    word_emb_tensor = default_collate(word_embs)
    pos_oh_tensor = default_collate(pos_ohs)
    sent_len_tensor = default_collate(sent_lens)
    motion_tensor = default_collate(motions)
    m_len_tensor = default_collate(m_lens)

    # Keep complex structures as lists
    text_list = list(texts)
    token_list = list(tokens)
    video_list = list(videos)
    style_list = list(styles)

    return word_emb_tensor, pos_oh_tensor, text_list, sent_len_tensor, motion_tensor, m_len_tensor, token_list, style_list, video_list


def collate_tensors(batch):
    """
    Pads a list of variable-length tensors to the maximum length in the batch.
    """
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def mld_collate_paired(batch):
    """
    Collate function for Paired Data.
    Expects a dictionary from Text2MotionDatasetCombined_v4 or v5.
    """
    notnone_batches = [b for b in batch if b is not None]
    
    if not notnone_batches:
        return {}
    
    # Sort by the length of the 100Styles motion
    notnone_batches.sort(key=lambda x: x['length_styled'], reverse=True)

    raw_latents = [torch.tensor(b['latent_humanml']).float() for b in notnone_batches]
    
    # # Debug: shapes before padding
    # print("latent_humanml shapes before padding:", [lat.shape for lat in raw_latents])

    # Find the maximum temporal length in this batch (the last dimension)
    max_t = max([lat.shape[-1] for lat in raw_latents])
    
    # Pad the last dimension (T) with zeros on the right side
    padded_latents = [F.pad(lat, (0, max_t - lat.shape[-1])) for lat in raw_latents]

    # Pad Raw HumanML Motions (Shape: [T, dim_pose])
    raw_motions_humanml = [torch.tensor(b['motion_humanml']).float() for b in notnone_batches]
    
    # Find max T for raw motions
    max_t_motion = max([m.shape[0] for m in raw_motions_humanml])
    
    # F.pad for 2D tensors reads as (pad_left, pad_right, pad_top, pad_bottom).
    # We want to pad the bottom of the T dimension.
    padded_motions_humanml = [F.pad(m, (0, 0, 0, max_t_motion - m.shape[0])) for m in raw_motions_humanml]


    # # Debug: shapes after padding
    # print("latent_humanml shapes after padding:", [lat.shape for lat in padded_latents])


    adapted_batch = {
        # --- Pair 1 (100STYLES: Raw Motion) ---
        "text_styled":   [b['text_styled'] for b in notnone_batches],
        "motion_styled": torch.stack([torch.tensor(b['motion_styled']).float() for b in notnone_batches]),
        "length_styled": torch.tensor([b['length_styled'] for b in notnone_batches], dtype=torch.long),
        "style_name":    [b['style_name'] for b in notnone_batches],
        "video_styled":  [b['video_styled'] for b in notnone_batches], # Kept as list of lists for HF ViViT

        # --- Pair 2 (HumanML3D: Pre-Encoded Latents) ---
        "text_humanml":   [b['text_humanml'] for b in notnone_batches],
        "motion_humanml": torch.stack(padded_motions_humanml),  # Padded raw motions
        "latent_humanml": torch.stack(padded_latents),
        "length_humanml": torch.tensor([b['length_humanml'] for b in notnone_batches], dtype=torch.long),
    }

    return adapted_batch


def mld_collate_async(batch):
    """
    Dynamic Collate function for Decoupled Data (v5).
    Safely checks which yield_mode is active and only collates available keys.
    """
    notnone_batches = [b for b in batch if b is not None]
    
    if not notnone_batches:
        return {}
    
    adapted_batch = {}
    
    # Peek at the first sample to see which yield_mode we are dealing with
    sample_keys = notnone_batches[0].keys()

    raw_latents = [torch.tensor(b['latent_humanml']).float() for b in notnone_batches]
    
    # Find the maximum temporal length in this batch (the last dimension)
    max_t = max([lat.shape[-1] for lat in raw_latents])
    
    # Pad the last dimension (T) with zeros on the right side
    padded_latents = [F.pad(lat, (0, max_t - lat.shape[-1])) for lat in raw_latents]


    # ==========================================
    # ROUTE A: 100STYLES Batch
    # ==========================================
    if 'motion_styled' in sample_keys:
        # Sort by the length of the 100Styles motion
        notnone_batches.sort(key=lambda x: x['length_styled'], reverse=True)
        
        adapted_batch.update({
            "text_styled":   [b['text_styled'] for b in notnone_batches],
            "motion_styled": torch.stack([torch.tensor(b['motion_styled']).float() for b in notnone_batches]),
            "length_styled": torch.tensor([b['length_styled'] for b in notnone_batches], dtype=torch.long),
            "style_name":    [b['style_name'] for b in notnone_batches],
            "video_styled":  [b['video_styled'] for b in notnone_batches], # List of lists for HF ViViT
        })

    # ==========================================
    # ROUTE B: HumanML3D Batch
    # ==========================================
    if 'latent_humanml' in sample_keys:
        # Sort by the length of the HumanML3D motion
        notnone_batches.sort(key=lambda x: x['length_humanml'], reverse=True)
        
        adapted_batch.update({
            "text_humanml":   [b['text_humanml'] for b in notnone_batches],
            "latent_humanml": torch.stack([torch.tensor(b['latent_humanml']).float() for b in notnone_batches]),
            "length_humanml": torch.stack(padded_latents),
        })

    return adapted_batch

def mld_collate_paired_v2(batch, style_method=None):
    """
    Collate function for Paired Data with Video and Style.
    Expected Tuple from Dataset:
      0: caption_style (Text 1)
      1: motion_style (Motion 1)
      2: m_length_style (Length 1)
      3: style_name_style (Style Class)
      4: video_snippet_style (Video)
      5: caption_humanml3d (Text 2)
      6: motion_humanml3d (Motion 2)
      7: m_length_humanml3d (Length 2)
    """
    notnone_batches = [b for b in batch if b is not None]
    
    # Sort by length_common (descending) - using DICTIONARY KEY, not index
    
    

    if style_method == 3:
        notnone_batches.sort(key=lambda x: x['length_common'], reverse=True)

        return {

            "text_styled":   [b['text_styled'] for b in notnone_batches],
            "motion_styled":  collate_tensors([torch.tensor(b['motion_styled']).float() for b in notnone_batches]),
            "length_styled":   [b['length_styled'] for b in notnone_batches],
            "style_name":     [b['style_name'] for b in notnone_batches],
            "video_styled":    [b['video_styled'] for b in notnone_batches],
            
            "text_humanml":   [b['text_humanml'] for b in notnone_batches],
            "motion_humanml": collate_tensors([torch.tensor(b['motion_humanml']).float() for b in notnone_batches]),
            "length_humanml": [b['length_humanml'] for b in notnone_batches],
            
            "text_neutral":    [b['text_neutral'] for b in notnone_batches],
            "motion_neutral": collate_tensors([torch.tensor(b['motion_neutral']).float() for b in notnone_batches]),
            "length_neutral":  [b['length_neutral'] for b in notnone_batches],
            "video_neutral":   [b['video_neutral'] for b in notnone_batches],
        }
    
    else:
        
        notnone_batches.sort(key=lambda x: x['length_styled'], reverse=True)
        return {
            # --- Pair 1 (100Styles) ---
            "text_styled":   [b['text_styled'] for b in notnone_batches],
            "motion_styled": collate_tensors([torch.tensor(b['motion_styled']).float() for b in notnone_batches]),
            "length_styled": [b['length_styled'] for b in notnone_batches],
            "style_name":  [b['style_name'] for b in notnone_batches],  # List of style strings
            "video_styled":  [b['video_styled'] for b in notnone_batches],  # List of video frame lists (heavy, do not stack)

            # --- Pair 2 (HumanML3D) ---
            "text_humanml":   [b['text_humanml'] for b in notnone_batches],
            "motion_humanml": collate_tensors([torch.tensor(b['motion_humanml']).float() for b in notnone_batches]),
            "length_humanml": [b['length_humanml'] for b in notnone_batches],
        }



#################################################################################
#                               Helper functions                                #
#################################################################################

def random_zero_out(data, percentage=0.4, probability=0.6,noise_probability=0.8,noise_level=0.05):
    if random.random() < probability:
        # Calculate the total number of sequences to zero out
        num_sequences = data.shape[0]

        percentage = np.random.rand() * 0.5
        num_to_zero_out = int(num_sequences * percentage)

        # Randomly choose sequence indices to zero out
        indices_to_zero_out = np.random.choice(num_sequences, num_to_zero_out, replace=False)

        # Zero out the chosen sequences
        data[indices_to_zero_out, :] = 0

    # data = shuffle_segments_numpy(data,16)

    if random.random() < noise_probability:
        noise = np.random.normal(0, noise_level * np.ptp(data), data.shape)
        data += noise

    return data


def build_dict_from_txt(filename):
    '''
    Build a dictionary from a metadata file.
    The metadata file should contain lines in the format:
    <motion_id> <BVH filename> <style class label>

    '''
    result_dict = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")  
            if len(parts) >= 3:
                key = parts[0]                                      ## motion id eg: 030001
                value = parts[2].split("_")[0]                      ## style class label in numbers: 0...99       
                value2 = parts[1].split("_")[0]                     ## style class label in words; eg: Aeroplane  
                value3 = parts[1].split("_")[1]                     ## motion type: eg: BR, BW, FR, FW, ID, SR, SW       
                value4 = parts[1].split("_")[2].split('.')[0]       ## sequence cut index, i.e point from where motion should be considered. values range: 00...09
                value5 = parts[3]                                   ## Sequence length in frames: eg: 120, 240

            result_dict[key] = value, value2, value3, value4, int(value5)
                
    return result_dict


def build_dict_from_txt2(filename):
    '''
    Build a dictionary from a metadata file.
    For the new file_lengths.txt, the format is:
    <motion_id>,<length>
    '''
    result_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            # supports comma-separated or space-separated
            parts = line.strip().replace(",", " ").split()
            if len(parts) >= 2:
                key = parts[0]              # motion id (e.g., 030001)
                length = int(parts[1])      # sequence length in frames
                result_dict[key] = length
    return result_dict

def build_dict_from_txt_mirror(filename):
    """
    Build a dictionary from a metadata file.
    
    Input format expected:
    <motion_id> <BVH filename> <style class label> <frame_count>
    Example: 
    030001 Aeroplane_BR_00.bvh 0 263
    M030001 Aeroplane_BR_00.bvh 0 263
    
    Returns dict:
    { key: (style_id, style_name, motion_type, motion_idx, is_mirrored, length) }
    """
    result_dict = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            
            # Ensure we have enough parts (ID, Filename, StyleID, Length)
            if len(parts) >= 4:
                key = parts[0]                                      # e.g., M030001
                
                # 1. Determine Mirroring
                is_mirrored = key.startswith("M")
                
                # 2. Parse Filename (Aeroplane_BR_00.bvh)
                filename_parts = parts[1].split("_")
                style_name = filename_parts[0]                      # Aeroplane
                motion_type = filename_parts[1]                     # BR
                motion_idx = filename_parts[2].split('.')[0]        # 00
                
                # 3. Get Style ID and Length
                style_id = parts[2]                                 # 0
                length = int(parts[3])                              # 263
                
                # Return tuple with ALL components
                result_dict[key] = (style_id, style_name, motion_type, motion_idx, is_mirrored, length)
                
    return result_dict

def read_video_decord(video_path, start, num_frames):
    with timer("data/video_loading"):
        try:
            container = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
            total_frames = len(container)
            frame_indices = np.arange(start, start + num_frames, dtype=int)
            frames = container.get_batch(frame_indices).asnumpy()                       # (T, H, W, C)
        except Exception as e:
            print(f"Warning: Could not load video {video_path}. Returning a dummy tensor. Error: {e}")
            frames = (np.random.rand(num_frames, 224, 224, 3) * 255).astype(np.uint8)
    
        video_frames = list(frames)    
    
        return video_frames


def load_and_split_data(root_path="./datasets/100STYLE-SMPL/",
                        split_file='train_100STYLE_Full.txt',
                        dict_file='100STYLE_name_dict.txt',
                        video_subpath="videos",
                        motion_subpath="new_joint_vecs",
                        style_classes=["Aeroplane", "Chicken", "Robot", "Superman", "ArmsFolded"],
                        video_filename_suffix="_FV.mp4",
                        dim_pose=67) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, list]]:
    """
    Scans for video and motion files, filters them based on metadata, and splits them
    into training, validation, and testing sets.

    Returns:
        A tuple of three dictionaries: (train_data, val_data, test_data)
        Each dictionary has keys "paths" and "labels".
    """
    video_root = os.path.join(root_path, video_subpath)
    motion_root = os.path.join(root_path, motion_subpath)
    dict_path = os.path.join(root_path, dict_file)
    split_path = os.path.join(root_path, split_file)

    # --- Load metadata and split information ---
    motion_to_meta = build_dict_from_txt(dict_path)
    style_to_label = {style: i for i, style in enumerate(style_classes)}

    with cs.open(split_path, "r") as f:
        train_ids = {line.strip() for line in f.readlines()}

    # --- Populate train and initial test lists ---
    train_video_paths, train_motion_data , train_labels = [], [], []
    initial_test_video_paths, initial_test_motion_data, initial_test_labels = [], [], []

    all_video_files = glob.glob(os.path.join(video_root, "*" + video_filename_suffix))
    # all_motion_files = glob.glob(os.path.join(motion_root, ".npy"))

    for video_path in all_video_files:
        basename = os.path.basename(video_path)
        motion_id = basename.split('_')[0]

        if motion_id in motion_to_meta:
            _, style_name, motion_type, _ = motion_to_meta[motion_id]

            if style_name not in style_classes or motion_type == "TR":
                continue

            if motion_id in train_ids:
                train_video_paths.append(video_path)
                train_labels.append(style_to_label[style_name])
                train_motion_path = pjoin(motion_root, motion_id + ".npy")
                train_motion = np.load(train_motion_path)
                train_motion_data.append(train_motion[:, :dim_pose])
            else:
                initial_test_video_paths.append(video_path)
                initial_test_labels.append(style_to_label[style_name])
                initial_test_motion_path = pjoin(motion_root, motion_id + ".npy")
                initial_test_motion = np.load(initial_test_motion_path)
                initial_test_motion_data.append(initial_test_motion[:, :dim_pose])

    # --- Split the initial test set into validation and final test sets (50/50) ---
    # Shuffle the initial test set to ensure a random split
    temp_test_data = list(zip(initial_test_video_paths, initial_test_motion_data, initial_test_labels))
    random.shuffle(temp_test_data)
    if temp_test_data:
        initial_test_video_paths, initial_test_motion_data, initial_test_labels = zip(*temp_test_data)
    
    split_idx = len(initial_test_video_paths) // 2

    val_video_paths = list(initial_test_video_paths[:split_idx])
    val_motion_data = list(initial_test_motion_data[:split_idx])
    val_labels = list(initial_test_labels[:split_idx])
    
    test_video_paths = list(initial_test_video_paths[split_idx:])
    test_motion_data = list(initial_test_motion_data[split_idx:])
    test_labels = list(initial_test_labels[split_idx:])

    print(f"Found {len(train_video_paths)} training, {len(val_video_paths)} validation, and {len(test_video_paths)} testing videos.")

    train_data = {"video_paths": train_video_paths, "motion": train_motion_data ,"labels": train_labels}
    val_data = {"video_paths": val_video_paths, "motion": val_motion_data ,"labels": val_labels}
    test_data = {"video_paths": test_video_paths, "motion": test_motion_data ,"labels": test_labels}

    return train_data, val_data, test_data

PUNCT_TRANSLATOR = str.maketrans('', '', string.punctuation)

def normalize_caption(caption: str) -> str:
    return caption.translate(PUNCT_TRANSLATOR).strip()


#################################################################################
#                                      Datasets                                 #
#################################################################################
class AEDataset(data.Dataset):
    def __init__(self, mean, std, motion_dir, window_size, split_file, dim_pose):
        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                motion = motion[:, :dim_pose]
                if motion.shape[0] < window_size:
                    continue
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except Exception as e:
                pass
        self.cumsum = np.cumsum([0] + self.lengths)
        self.window_size = window_size ## 32

        self.mean = mean[:dim_pose]
        self.std = std[:dim_pose]

        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):        
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.window_size]
        "Z Normalization"
        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        return motion

class AEVideoDataset_100styles_v1(data.Dataset):
    ## Contiguous chunks from each motion sequence are grouped together in each epoch
    def __init__(self, mean, std, motion_dir, video_dir, styles, window_size, split_file, dim_pose, dict_file, snippets_per_sequence=1):
        with timer("data/dataset_initialization"):
            self.styles = styles
            self.window_size = window_size
            self.mean = mean[:dim_pose]
            self.std = std[:dim_pose]
            # snippets_per_sequence is now used to build the index map, not directly in __len__ or __getitem__
            
            self.data = []
            self.video_paths = []
            self.motion_lengths = []
            
            style_labels = []
            id_list = []
            with open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())

            metadata_dict = build_dict_from_txt(dict_file)

            print("Loading dataset...")
            for name in tqdm(id_list):
                try:
                    # ... (rest of the file loading logic is the same) ...
                    video_path = pjoin(video_dir, name + '_FV.mp4')
                    motion_path = pjoin(motion_dir, name + '.npy')
                    
                    if not os.path.exists(motion_path) or not os.path.exists(video_path):
                        continue
                    
                    motion = np.load(motion_path)
                    
                    if motion.shape[0] < self.window_size:
                        continue
                    if metadata_dict[name][1] not in self.styles:
                        continue
                    if metadata_dict[name][2].startswith("TR"):  # Exclude transitions
                        continue

                    style_label = int(metadata_dict[name][0])
                    
                    self.motion_lengths.append(motion.shape[0])
                    self.data.append(motion[:, :dim_pose])
                    self.video_paths.append(video_path)
                    style_labels.append(style_label)

                except Exception as e:
                    print(f"Warning: Skipping file {name} due to error: {e}")
                    pass
            
            unique_labels = sorted(list(set(style_labels)))
            mapping = {label: i for i, label in enumerate(unique_labels)}
            self.labels = [mapping[label] for label in style_labels]

            print(f"Total number of motions loaded: {len(self.data)}")
            print(f"Video files list length: {len(self.video_paths)}")

            # --- New Shuffling Logic ---
            # 1. Create a flat list where each motion_id is repeated `snippets_per_sequence` times.
            self.motion_id_map = []
            for i in range(len(self.data)):
                self.motion_id_map.extend([i] * snippets_per_sequence)
            
            # 2. Shuffle this map. This is the key to breaking the sequential grouping.
            random.shuffle(self.motion_id_map)
            
            print(f"Total number of snippets per epoch: {self.__len__()} with {snippets_per_sequence} snippets per sequence.")

    def __len__(self):
        # The length is the size of our new shuffled map.
        return len(self.motion_id_map)

    def __getitem__(self, item):
        with timer("data/getitem_call"):
            # The `item` index now directly looks up the pre-shuffled motion_id.
            motion_id = self.motion_id_map[item]
            
            motion_data = self.data[motion_id]
            motion_length = self.motion_lengths[motion_id]
            label = self.labels[motion_id]

            # 1. Choose a random start index for the snippet
            max_start_idx = motion_length - self.window_size
            idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
            
            # 2. Slice the motion data (from RAM - fast)
            motion_snippet = motion_data[idx : idx + self.window_size]

            # 3. Randomly select video view and read the corresponding snippet (one I/O call)
            # if random.choice([True, False]):
            #     video_path = self.video_paths_fv[motion_id]
            # else:
            #     video_path = self.video_paths_lv[motion_id]
            
            # video_snippet = read_video_decord(video_path, start=idx, num_frames=self.window_size)

            # video_snippet = read_video_decord(self.video_paths_lv[motion_id], start=idx, num_frames=self.window_size)
            video_snippet = read_video_decord(self.video_paths[motion_id], start=idx, num_frames=self.window_size)
            
            # vr = VideoReader(self.video_paths_fv[motion_id], ctx=cpu(0), width=224, height=224)
            # total_frames = len(vr)
            # frame_indices = np.linspace(0, total_frames - 1, num=self.window_size, dtype=int) 
            # frames = vr.get_batch(frame_indices).asnumpy()
            # video_snippet = list(frames)



            # 4. Normalize the motion snippet
            motion_snippet = (motion_snippet - self.mean) / self.std

            return motion_snippet, video_snippet, label
        
    def get_style_names(self):
        if hasattr(self, 'style_classes') and self.style_classes:
            return self.style_classes
        
        # Fallback: just return numbers as strings
        num_classes = len(self.style_to_idx) if hasattr(self, 'style_to_idx') else 100
        return [str(i) for i in range(num_classes)]

class AEVideoDataset_100styles_v2(data.Dataset):
    ## Unifom sampling throughout the entire sequence for each snippet, but with randomized start points
    def __init__(self, mean, std, motion_dir, video_dir, styles, window_size, split_file, dim_pose, dict_file):
        with timer("data/dataset_initialization"):
            self.styles = styles
            self.window_size = window_size
            self.mean = mean[:dim_pose]
            self.std = std[:dim_pose]
            

            self.motion_data = [] 
            self.video_paths = []
            self.motion_lengths = []
            
            style_labels = []
            id_list = []
            try:
                with open(split_file, 'r') as f:
                    for line in f.readlines():
                        id_list.append(line.strip())
            except FileNotFoundError:
                print(f"Warning: Split file not found at {split_file}. Dataset will be empty.")
                id_list = []

            metadata_dict = build_dict_from_txt(dict_file)

            print("Loading dataset...")
            for name in tqdm(id_list):
                try:
                    video_path = pjoin(video_dir, name + '_FV.mp4')
                    motion_path = pjoin(motion_dir, name + '.npy')
                    
                    if not os.path.exists(motion_path) or not os.path.exists(video_path):
                        continue
                    
                    motion = np.load(motion_path)
                    
                    if motion.shape[0] < self.window_size:
                        continue
                    if name in metadata_dict:
                        if metadata_dict[name][1] not in self.styles:
                            continue
                        if metadata_dict[name][2].startswith("TR"):  # Exclude transitions
                            continue
                        style_label = int(metadata_dict[name][0])
                    else:
                        # Fallback if metadata is missing for an ID
                        print(f"Warning: No metadata found for {name}. Skipping.")
                        continue
                        
                    self.motion_lengths.append(motion.shape[0])
                    self.motion_data.append(motion[:, :dim_pose]) 
                    self.video_paths.append(video_path)          
                    style_labels.append(style_label)

                except Exception as e:
                    print(f"Warning: Skipping file {name} due to error: {e}")
                    pass
            
            unique_labels = sorted(list(set(style_labels)))
            mapping = {label: i for i, label in enumerate(unique_labels)}
            self.labels = [mapping[label] for label in style_labels]

            print(f"Total number of motions loaded: {len(self.motion_data)}")
            print(f"Video files list length: {len(self.video_paths)}")


    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, item):
        with timer("data/getitem_call"):
            
            motion_data = self.motion_data[item]
            motion_length = self.motion_lengths[item]
            label = self.labels[item]
            video_path = self.video_paths[item]

            try:
                                
                ## Get video length to find the true minimum
                vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
                video_length = len(vr)
                
                ## Use the *minimum* length for synchronization
                max_valid_frames = min(motion_length, video_length)
                
                ## Find a random start index
                max_start_idx = max_valid_frames - self.window_size
                idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
                
                ## Uniformly sample from the start index to the whole video sequence
                frame_indices = np.linspace(idx, max_valid_frames - 1, self.window_size, dtype=int)
                
                ## Sample motion and video
                motion_snippet = motion_data[frame_indices]
                frames = vr.get_batch(frame_indices).asnumpy()
                video_snippet = list(frames)
                
            except Exception as e:
                # Fallback for any loading error
                print(f"Warning: Could not load snippet from {video_path} (Index {item}). Returning dummy. Error: {e}")
                frames = (np.random.rand(self.window_size, 224, 224, 3) * 255).astype(np.uint8)
                video_snippet = list(frames)
                # Return a simple slice from the start for motion
                motion_snippet = self.motion_data[item][0 : self.window_size] 

            motion_snippet = (motion_snippet - self.mean) / self.std

            return motion_snippet, video_snippet, label
        
    def get_style_names(self):
        if hasattr(self, 'style_classes') and self.style_classes:
            return self.style_classes
        
        # Fallback: just return numbers as strings
        num_classes = len(self.style_to_idx) if hasattr(self, 'style_to_idx') else 100
        return [str(i) for i in range(num_classes)]

class AEVideoDataset_100styles_v3(data.Dataset):
    ## Unifom sampling between randomized start and stop points
    def __init__(self, mean, std, motion_dir, video_dir, styles, window_size, split_file, dim_pose, dict_file, snippets_per_sequence=1):
        self.styles = styles
        self.window_size = window_size
        self.mean = mean[:dim_pose]
        self.std = std[:dim_pose]
        
        self.data = []
        # self.video_paths_fv = []
        # self.video_paths_lv = []
        self.video_paths = []
        self.motion_lengths = []
        self.labels = []
        
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        metadata_dict = build_dict_from_txt(dict_file)
        style_to_label = {style: i for i, style in enumerate(self.styles)}


        print("Loading dataset...")
        for name in tqdm(id_list):
            try:
                # video_path_fv = pjoin(video_dir, name + '_FV.mp4')
                # video_path_lv = pjoin(video_dir, name + '_LV.mp4')
                _, style_name, motion_type, _, _ = metadata_dict[name]
                video_path = pjoin(video_dir, name +  '_FV.mp4')
                motion_path = pjoin(motion_dir, name + '.npy')
                
                # if not os.path.exists(motion_path) or not os.path.exists(video_path_fv) or not os.path.exists(video_path_lv):
                if not os.path.exists(motion_path) or not os.path.exists(video_path):
                    continue
                
                motion = np.load(motion_path)
                
                if motion.shape[0] < self.window_size:
                    # print(f"Skipping {name} due to insufficient length: {motion.shape[0]}")
                    continue
                if style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):  # Exclude transitions
                    continue
                
                self.motion_lengths.append(motion.shape[0])
                self.data.append(motion[:, :dim_pose])
                # self.video_paths_fv.append(video_path_fv)
                # self.video_paths_lv.append(video_path_lv)
                self.video_paths.append(video_path)
                self.labels.append(style_to_label[style_name])

            except Exception as e:
                print(f"Warning: Skipping file {name} due to error: {e}")
                pass

        print(f"Total number of motions loaded: {len(self.data)}")
        # print(f"Video files list length: {len(self.video_paths_fv)}, {len(self.video_paths_lv)}")
        print(f"Video files list length: {len(self.video_paths)}")


        self.motion_id_map = []
        for i in range(len(self.data)):
            self.motion_id_map.extend([i] * snippets_per_sequence)
        
        random.shuffle(self.motion_id_map)
        
        print(f"Total number of snippets per epoch: {self.__len__()} with {snippets_per_sequence} snippets per sequence.")

    def __len__(self):
        # The length is the size of our new shuffled map.
        return len(self.motion_id_map)

    def __getitem__(self, item):
            motion_id = self.motion_id_map[item]
            
            motion_data = self.data[motion_id]
            motion_length = self.motion_lengths[motion_id]
            label = self.labels[motion_id]

            ## Randomly select which video view to use
            # if random.choice([True, False]):
            #     video_path = self.video_paths_fv[motion_id]
            # else:
            #     video_path = self.video_paths_lv[motion_id]

            video_path = self.video_paths[motion_id]

        
            ## Open video to get its true length
            vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
            video_length = len(vr)

            ## Find the shortest sequence length to prevent errors
            max_valid_frames = min(motion_length, video_length)

            ## Define a random sequence span to sample *from*. This span must be at least 'window_size' long.
            min_span = self.window_size
            max_span = max_valid_frames
            
            if min_span > max_span:
                # This should not happen if __init__ check is correct, but handles edge case
                span = max_span
            else:
                span = random.randint(min_span, max_span)

            ## Find a random start index for this span
            max_start_idx = max_valid_frames - span
            start = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
            stop = start + span - 1

            ## Generate 1 set of indices using linspace on this *random span*
            if self.window_size > 32:
                motion_frame_indices = np.linspace(start, stop, num=self.window_size, dtype=int)
                video_frame_indices =  np.linspace(start, stop, num=32, dtype=int)  # Keep them the same for synchronization
                # 7. Sample both modalities with the different indices from the same span
                motion_snippet = motion_data[motion_frame_indices]  # Sample from RAM (fast)
                frames = vr.get_batch(video_frame_indices).asnumpy() # Sample from Disk (slow)
                video_snippet = list(frames)
            elif self.window_size == 32:
                frame_indices = np.linspace(start, stop, num=self.window_size, dtype=int)
                # 7. Sample both modalities with the *same* indices
                motion_snippet = motion_data[frame_indices]  # Sample from RAM (fast)
                frames = vr.get_batch(frame_indices).asnumpy() # Sample from Disk (slow)
                video_snippet = list(frames)
                
            # --- END OF KEY CHANGE ---

            # 8. Normalize the motion snippet
            motion_snippet = (motion_snippet - self.mean) / self.std

            return motion_snippet, video_snippet, label
    
    def get_style_names(self):
        if hasattr(self, 'style_classes') and self.style_classes:
            return self.style_classes
        
        # Fallback: just return numbers as strings
        num_classes = len(self.style_to_idx) if hasattr(self, 'style_to_idx') else 100
        return [str(i) for i in range(num_classes)]

class AEVideoStyleDataset(data.Dataset):
    """
    A dataset that loads and transforms video frames from a pre-defined list of paths.
    """
    def __init__(self, data_dict: Dict[str, list], mean, std, dim_pose=67, num_frames_to_extract: int = 16):
        self.video_paths = data_dict["video_paths"]
        self.motions = data_dict["motion"]
        self.labels = data_dict["labels"]
        self.mean = mean[:dim_pose]
        self.std = std[:dim_pose]
        self.num_frames = num_frames_to_extract

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        motion = self.motions[idx]
        label = self.labels[idx]

        try:
            # Load and sample frames using decord
            # Resize directly during loading for efficiency
            vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
            # total_frames = len(vr)
            seq_len = min(len(vr), len(motion))
            frame_indices = np.linspace(0, seq_len - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(frame_indices).asnumpy() # (T, H, W, C)
            # motion_data = motion[frame_indices]
        except Exception as e:
            print(f"Warning: Could not load video {video_path}. Returning a dummy tensor. Error: {e}")
            frames = (np.random.rand(self.num_frames, 224, 224, 3) * 255).astype(np.uint8)
        
        # The processor expects a list of numpy arrays (frames)
        video_frames = list(frames)
        motion_data = motion[frame_indices]
        motion_data = (motion_data - self.mean) / self.std

        return motion_data, video_frames, label
    
class AEVideoDataset_100styles(data.Dataset):
    ## Sliding window snippets from each motion sequence, very high redundancy
    def __init__(self, mean, std, motion_dir, video_dir, styles, window_size, split_file, dim_pose, dict_file):
        self.styles = styles
        self.data = []
        # self.video_paths = []
        # self.video_paths_fv = [] # Store front-view video paths
        # self.video_paths_lv = [] # Store left-view video paths
        self.video_paths = []
        self.lengths = []
        self.labels = []  # Added to store style labels

        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # Build the metadata dictionary
        metadata_dict = build_dict_from_txt(dict_file)
        style_to_label = {style: i for i, style in enumerate(self.styles)}


        for name in tqdm(id_list):
            try:
                _, style_name, motion_type, _, _ = metadata_dict[name]
                # video_path_fv = pjoin(video_dir, name + '_FV.mp4')
                # video_path_lv = pjoin(video_dir, name + '_LV.mp4')
                
                video_path = pjoin(video_dir, name +  '_FV.mp4')
                motion_path = pjoin(motion_dir, name + '.npy')
                
                # if not os.path.exists(motion_path) or not os.path.exists(video_path_fv) or not os.path.exists(video_path_lv):
                if not os.path.exists(motion_path) or not os.path.exists(video_path):
                    continue                

                motion = np.load(motion_path)
                motion = motion[:, :dim_pose]

                if motion.shape[0] < window_size:
                    continue
                if style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):  # Exclude transitions
                    continue
                
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
                # self.video_paths.append(video_path)
                # self.video_paths_fv.append(video_path_fv)
                # self.video_paths_lv.append(video_path_lv)
                self.video_paths.append(video_path)
                self.labels.append(style_to_label[style_name])


            except Exception as e:
                print(f"Warning: Skipping file {name} due to error: {e}")
                pass
        
        self.cumsum = np.cumsum([0] + self.lengths)
        self.window_size = window_size
        self.mean = mean[:dim_pose]
        self.std = std[:dim_pose]

        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))
        print("Average snippets per sequence: {:.2f}".format(self.cumsum[-1] / len(self.data)))

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        
        motion = self.data[motion_id][idx:idx + self.window_size]
        label = self.labels[motion_id]
        
        # if random.choice([True, False]):
        #     video_path = self.video_paths_fv[motion_id]
        # else:
        #     video_path = self.video_paths_lv[motion_id]

        video_path = self.video_paths[motion_id]

        try:
            vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
            video_length = len(vr)
            if video_length == 0:
                raise ValueError(f"Video {video_path} has no frames.")
            raw_indices = np.arange(idx, idx + self.window_size, dtype=int)
            raw_indices = np.clip(raw_indices, 0, max(video_length - 1, 0))
            if self.window_size > 32:
                sub_count = 32
                sample_positions = np.linspace(0, self.window_size - 1, num=sub_count, dtype=float)
                sample_positions = np.round(sample_positions).astype(int)
                sample_positions = np.clip(sample_positions, 0, self.window_size - 1)
                frame_indices = raw_indices[sample_positions]
                print(f"Number of video frames selected (inner): {len(frame_indices)}")
            else:
                frame_indices = raw_indices           
            frame_indices = frame_indices.astype(int)
            print(f"Number of video frames selected (outer): {len(frame_indices)}")
            frames = vr.get_batch(frame_indices).asnumpy()
            video = list(frames)
        except Exception as e:
            print(f"Warning: Could not load video snippet from {video_path}. Returning a dummy tensor. Error: {e}")
            video = list((np.random.rand(max(32, self.window_size), 224, 224, 3) * 255).astype(np.uint8))

        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        return motion, video, label

class Text2MotionDataset(data.Dataset):
    def __init__(self, mean, std, split_file, dataset_name, motion_dir, text_dir, 
                 unit_length, dim_pose, max_motion_length,
                 max_text_length, max_vid_length=32, evaluation=False):
        self.evaluation = evaluation
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.unit_length = unit_length
        self.max_vid_length = max_vid_length
        min_motion_len = 40 if dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []

        print(f"Loading {dataset_name} dataset...")
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        else:
            name_list, length_list = new_name_list, length_list
        self.mean = mean[:dim_pose]
        self.std = std[:dim_pose]
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        if self.evaluation:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.evaluation:
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        elif m_length > self.max_motion_length:
            if not self.evaluation:
                idx = random.randint(0, self.max_motion_length - m_length)
                motion = motion[idx:idx + self.max_motion_length]

        # Create dummy video for mixed training
        # video_snippet = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.max_vid_length)]

        if self.evaluation:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        else:
            # return caption, motion, m_length, video_snippet
            return caption, motion, m_length
        
## Standard matched data loading        
class Text2MotionVideoDataset_100styles(data.Dataset):
    def __init__(self, mean, std, split_file, dataset_name, video_dir, motion_dir, text_dir, dict_file, unit_length, max_motion_length, styles, dim_pose, max_vid_length=32, evaluation=False):
        
        assert styles is not None, "Styles must be provided for Text2MotionVideoDataset"

        self.evaluation = evaluation
        self.styles = styles
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # self.max_text_len = max_text_length
        self.max_vid_length = max_vid_length
        self.unit_length = unit_length
        min_motion_len = 40 if dataset_name in ['t2m','100styles'] else 24

        self.data = []
        self.motion_lengths = []
 

        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        metadata_dict = build_dict_from_txt(dict_file)
        style_to_label = {style: i for i, style in enumerate(self.styles)}

        new_name_list = []
        length_list = []

        print(f"Loading {dataset_name} dataset...")
        for name in tqdm(id_list):
            try:
                
                ################## Load video ##################
                # video_path_fv = pjoin(video_dir, name + '_FV.mp4')
                # video_path_lv = pjoin(video_dir, name + '_LV.mp4')
                video_path = pjoin(video_dir, name +  '_FV.mp4')
                 ################## Load video ##################

                _, style_name, motion_type, _, _ = metadata_dict[name]

                ################## Load motion ##################
                motion_path = pjoin(motion_dir, name + '.npy')

                # if not os.path.exists(motion_path) or not os.path.exists(video_path_fv) or not os.path.exists(video_path_lv):
                if not os.path.exists(motion_path) or not os.path.exists(video_path):
                    continue

                motion = np.load(motion_path)


                if (len(motion)) < min_motion_len or (len(motion) >= 400):
                    continue
                if style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):  # Exclude transitions
                    continue

                ################## Load and preprocess text ##################
                text_data = []
                text_path = pjoin(text_dir, name + ".txt")                                  
                assert os.path.exists(text_path)
                seen_captions = set()
                with cs.open(text_path) as f:
                    for line in f.readlines():                                                                  ## load the text data
                        # text_dict_2 = {}
                        line_split = line.strip().split("#")                                                    ## split the line to get caption, since there is no '#', it takes the whole line, eg: "a person holds their arms up and turns from left to right while waving their arms"
                        caption = line_split[0]
                        caption = normalize_caption(caption)
                        if not caption:
                            continue
                        # text_dict_2["caption"] = caption                                                        ## creates a dictionary for the caption, eg: {"caption": "a person holds their arms up and turns from left to right while waving their arms"}
                        canonical_caption = caption.lower()
                        if canonical_caption in seen_captions:
                            continue
                        seen_captions.add(canonical_caption)
                        text_data.append(caption)                                                           ## appends the dictionary to the text data list
                ################## Load and preprocess text ##################

                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'style': style_to_label[style_name],
                                   'style_name': style_name,
                                   'text': text_data,
                                #    'video_fv': video_path_fv,
                                #    'video_lv': video_path_lv
                                   'video': video_path
                                    }
                
                new_name_list.append(name)
                length_list.append(len(motion))
                
            except:
                pass
    
        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        else:
            name_list, length_list = new_name_list, length_list

        self.mean = mean[:dim_pose]
        self.std = std[:dim_pose]
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        if self.evaluation:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * self.std + mean
        

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        # motion, m_length, style_label, text_list, video_path_fv, video_path_lv = data['motion'], data['length'], data['style'], data['text'], data['video_fv'], data['video_lv']
        motion, m_length, style_label, style_name, text_list, video_path = data['motion'], data['length'], data['style'], data['style_name'], data['text'], data['video']


        # Randomly select a caption
        text_data = random.choice(text_list)

        if self.evaluation:
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        start_idx = random.randint(0, len(motion) - m_length)
        motion = motion[start_idx:start_idx+m_length]

        ## Load video snippets
        # if random.choice([True, False]):
        #     video_path = video_path_fv
        # else:
        #     video_path = video_path_lv

        # video_path = video_path
        try:
            vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
            video_length = len(vr)
            if video_length > 0:
                window_start = min(start_idx, max(video_length - 1, 0))
                window_end = min(start_idx + m_length-1, video_length-1)
                if window_end <= window_start:
                    window_end = window_start
                frame_indices = np.linspace(window_start, window_end, num=self.max_vid_length, dtype=int)
                frame_indices = frame_indices.tolist()
                while len(frame_indices) < self.max_vid_length:
                    frame_indices.append(frame_indices[-1])
                frame_indices = np.array(frame_indices[:self.max_vid_length], dtype=int)
                frames = vr.get_batch(frame_indices).asnumpy()
                video_snippet = list(frames)
            else:
                print(f"Warning: Video {video_path} has no frames. Check path and file integrity.")
        except Exception as e:
            print(f"Warning: Could not load video snippet from {video_path} due to error: {e}")

        "Z Normalization"
        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        elif m_length > self.max_motion_length:
            if not self.evaluation:
                idx = random.randint(0, m_length - self.max_motion_length)
                motion = motion[idx:idx + self.max_motion_length]

        if self.evaluation:
            return word_embeddings, pos_one_hots, text_data, sent_len, motion, m_length, '_'.join(tokens), style_name,  video_snippet
        else:
            return text_data, motion, m_length, style_name, video_snippet
        

## Stochastic matching: Pairs a styled video with any random neutral motion of the same category
class Text2MotionVideoDataset_100styles_v2(data.Dataset):
    """
    Dataset that pairs:
    - Motion & Text: from NEUTRAL style (content)
    - Video: from TARGET style (style transfer source)
    
    Matching is done by motion_type (FW, FR, BW, etc.)
    """
    
    def __init__(self, mean, std, split_file, dataset_name, video_dir, motion_dir, text_dir, 
                 dict_file, unit_length, max_motion_length, styles, dim_pose, 
                 max_vid_length=32, evaluation=False, neutral_style="Neutral"):
        
        assert styles is not None, "Styles must be provided"
        
        self.evaluation = evaluation
        self.styles = styles
        self.neutral_style = neutral_style
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_vid_length = max_vid_length
        self.unit_length = unit_length
        min_motion_len = 40 if dataset_name in ['t2m', '100styles'] else 24

        # Build style label mapping
        style_to_label = {style: i for i, style in enumerate(self.styles)}
        
        # Load metadata
        metadata_dict = build_dict_from_txt(dict_file)
        
        # Load split IDs
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # =====================================================================
        # STEP 1: Build NEUTRAL motion dictionary (indexed by motion_type)
        # =====================================================================
        self.neutral_motions = {}  # {motion_type: [list of neutral samples]}
        
        print(f"Loading NEUTRAL style motions...")
        for name in tqdm(id_list):
            try:
                _, style_name, motion_type, _, _ = metadata_dict[name]
                
                # Only process neutral style
                if style_name != self.neutral_style:
                    continue
                if motion_type.startswith("TR"):  # Exclude transitions
                    continue
                    
                motion_path = pjoin(motion_dir, name + '.npy')
                text_path = pjoin(text_dir, name + ".txt")
                
                if not os.path.exists(motion_path) or not os.path.exists(text_path):
                    continue
                    
                motion = np.load(motion_path)
                
                if len(motion) < min_motion_len or len(motion) >= 400:
                    continue
                
                # Load text
                text_data = []
                seen_captions = set()
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        caption = normalize_caption(caption)
                        if not caption:
                            continue
                        canonical_caption = caption.lower()
                        if canonical_caption in seen_captions:
                            continue
                        seen_captions.add(canonical_caption)
                        text_data.append(caption)
                
                if not text_data:
                    continue
                
                # Store neutral sample
                if motion_type not in self.neutral_motions:
                    self.neutral_motions[motion_type] = []
                    
                self.neutral_motions[motion_type].append({
                    'name': name,
                    'motion': motion,
                    'length': len(motion),
                    'text': text_data,
                    'motion_type': motion_type
                })
                
            except Exception as e:
                pass
        
        print(f"Loaded neutral motions for {len(self.neutral_motions)} motion types:")
        for mt, samples in self.neutral_motions.items():
            print(f"  {mt}: {len(samples)} samples")

        # =====================================================================
        # STEP 2: Build STYLED video dictionary (excluding neutral)
        # =====================================================================
        self.styled_data = {}  # Main data dict for styled samples
        new_name_list = []
        length_list = []
        
        print(f"\nLoading STYLED video samples...")
        for name in tqdm(id_list):
            try:
                _, style_name, motion_type, _, _ = metadata_dict[name]
                
                # Skip neutral (we use it for motion, not video)
                if style_name == self.neutral_style:
                    continue
                if style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):  # Exclude transitions
                    continue
                    
                # Check if we have matching neutral motion for this motion_type
                if motion_type not in self.neutral_motions:
                    continue
                if len(self.neutral_motions[motion_type]) == 0:
                    continue
                
                video_path = pjoin(video_dir, name + '_FV.mp4')
                
                if not os.path.exists(video_path):
                    continue
                
                # We don't need to load styled motion, just video
                # But we need motion length for video frame extraction
                motion_path = pjoin(motion_dir, name + '.npy')
                if os.path.exists(motion_path):
                    styled_motion = np.load(motion_path)
                    styled_length = len(styled_motion)
                else:
                    styled_length = 100  # Default fallback
                
                if styled_length < min_motion_len or styled_length >= 400:
                    continue
                
                self.styled_data[name] = {
                    'style_label': style_to_label[style_name],
                    'style_name': style_name,
                    'motion_type': motion_type,
                    'video': video_path,
                    'video_length_hint': styled_length  # For frame extraction
                }
                
                new_name_list.append(name)
                length_list.append(styled_length)
                
            except Exception as e:
                pass
        
        print(f"Loaded {len(self.styled_data)} styled video samples")

        # =====================================================================
        # Setup standard attributes
        # =====================================================================
        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        else:
            name_list, length_list = new_name_list, length_list

        self.mean = mean[:dim_pose]
        self.std = std[:dim_pose]
        self.length_arr = np.array(length_list)
        self.name_list = list(name_list)
        
        if self.evaluation:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        styled_name = self.name_list[idx]
        styled_data = self.styled_data[styled_name]
        
        style_label = styled_data['style_label']
        style_name = styled_data['style_name']
        motion_type = styled_data['motion_type']
        video_path = styled_data['video']
        video_length_hint = styled_data['video_length_hint']
        
        # =====================================================================
        # Get NEUTRAL motion & text (matched by motion_type)
        # =====================================================================
        neutral_samples = self.neutral_motions[motion_type]
        neutral_sample = random.choice(neutral_samples)
        
        motion = neutral_sample['motion'].copy()
        m_length = neutral_sample['length']
        text_data = random.choice(neutral_sample['text'])
        
        # =====================================================================
        # Process motion length (same as original)
        # =====================================================================
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        
        start_idx = random.randint(0, len(motion) - m_length)
        motion = motion[start_idx:start_idx + m_length]

        # =====================================================================
        # Load STYLED video
        # =====================================================================
        try:
            vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
            video_length = len(vr)
            
            if video_length > 0:
                # Use video_length_hint to determine frame window
                # (since styled video may have different length than neutral motion)
                window_start = 0
                window_end = min(video_length - 1, video_length_hint - 1)
                
                frame_indices = np.linspace(window_start, window_end, 
                                           num=self.max_vid_length, dtype=int)
                frame_indices = frame_indices.tolist()
                
                while len(frame_indices) < self.max_vid_length:
                    frame_indices.append(frame_indices[-1])
                    
                frame_indices = np.array(frame_indices[:self.max_vid_length], dtype=int)
                frames = vr.get_batch(frame_indices).asnumpy()
                video_snippet = list(frames)
            else:
                print(f"Warning: Video {video_path} has no frames.")
                video_snippet = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length
                
        except Exception as e:
            print(f"Warning: Could not load video from {video_path}: {e}")
            video_snippet = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length

        # =====================================================================
        # Normalize motion
        # =====================================================================
        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        # Pad/crop motion
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)
        elif m_length > self.max_motion_length:
            if not self.evaluation:
                crop_idx = random.randint(0, m_length - self.max_motion_length)
                motion = motion[crop_idx:crop_idx + self.max_motion_length]
                m_length = self.max_motion_length

        # =====================================================================
        # Return
        # =====================================================================
        if self.evaluation:
            # Add evaluation-specific processing here if needed
            return text_data, motion, m_length, style_name, video_snippet
        else:
            return text_data, motion, m_length, style_name, video_snippet
        

## Strict matching: Pairs a styled video only with the exact specific recording of the neutral motion
class Text2MotionVideoDataset_100styles_v3(data.Dataset):
    """
    Dataset that pairs:
    - Motion & Text: from NEUTRAL style (content)
    - Video: from TARGET style (style transfer source)
    
    Matching is done by EXACT motion identifier (e.g., BR_02, M_BR_02)
    """
    
    def __init__(self, mean, std, split_file, dataset_name, video_dir, motion_dir, text_dir, 
                 dict_file, unit_length, max_motion_length, styles, dim_pose, 
                 max_vid_length=32, evaluation=False, neutral_style="Neutral"):
        
        assert styles is not None, "Styles must be provided"
        
        self.evaluation = evaluation
        self.styles = styles
        self.neutral_style = neutral_style
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_vid_length = max_vid_length
        self.unit_length = unit_length
        self.dim_pose = dim_pose
        min_motion_len = 40 if dataset_name in ['t2m', '100styles'] else 24

        # Build style label mapping
        style_to_label = {style: i for i, style in enumerate(self.styles)}
        
        # Load metadata
        metadata_dict = build_dict_from_txt_mirror(dict_file)
        
        # Load split IDs
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # =====================================================================
        # STEP 1: Build NEUTRAL motion dictionary (indexed by motion_key)
        # motion_key = "BR_02" or "M_BR_02" (includes mirror prefix)
        # =====================================================================
        neutral_by_key = {}  # {"BR_02": {motion, text, length, ...}, "M_BR_02": {...}}
        
        print(f"Loading NEUTRAL style motions...")
        for name in tqdm(id_list):
            try:
                _, style_name, motion_type, motion_idx, _, is_mirrored = metadata_dict[name]
                
                if style_name != self.neutral_style:
                    continue
                if motion_type.startswith("TR"):
                    continue
                
                motion_path = pjoin(motion_dir, name + '.npy')
                text_path = pjoin(text_dir, name + ".txt")
                
                if not os.path.exists(motion_path) or not os.path.exists(text_path):
                    continue
                    
                motion = np.load(motion_path)
                
                if len(motion) < min_motion_len or len(motion) >= 400:
                    continue
                
                # Load text
                text_data = []
                seen_captions = set()
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        caption = normalize_caption(caption)
                        if not caption:
                            continue
                        canonical_caption = caption.lower()
                        if canonical_caption in seen_captions:
                            continue
                        seen_captions.add(canonical_caption)
                        text_data.append(caption)
                
                if not text_data:
                    continue
                
                # =============================================================
                # Create motion_key: "BR_02" or "M_BR_02"
                # =============================================================
                motion_key = f"{motion_type}_{motion_idx}"
                if is_mirrored:
                    motion_key = f"M_{motion_key}"
                
                neutral_by_key[motion_key] = {
                    'name': name,
                    'motion': motion,
                    'length': len(motion),
                    'text': text_data,
                }
                
            except Exception as e:
                pass
        
        print(f"Loaded {len(neutral_by_key)} neutral motion keys")
        # Print some examples
        example_keys = list(neutral_by_key.keys())[:10]
        print(f"Example keys: {example_keys}")

        # =====================================================================
        # STEP 2: Build PRE-COUPLED dataset (styled video + matched neutral)
        # =====================================================================
        self.data_list = []
        skipped_no_match = 0
        
        print(f"\nCreating coupled (neutral_motion, styled_video) pairs...")
        for name in tqdm(id_list):
            try:
                _, style_name, motion_type, motion_idx, _, is_mirrored = metadata_dict[name]
                
                # Skip neutral (we use it for motion, not video)
                if style_name == self.neutral_style:
                    continue
                if style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):
                    continue
                
                # =============================================================
                # Create motion_key to find matching neutral
                # =============================================================
                motion_key = f"{motion_type}_{motion_idx}"
                if is_mirrored:
                    motion_key = f"M_{motion_key}"
                
                # Must have matching neutral motion
                if motion_key not in neutral_by_key:
                    skipped_no_match += 1
                    continue
                
                video_path = pjoin(video_dir, name + '_FV.mp4')
                if not os.path.exists(video_path):
                    continue
                
                # Get styled motion length (for video frame extraction)
                motion_path = pjoin(motion_dir, name + '.npy')
                if os.path.exists(motion_path):
                    styled_motion_len = len(np.load(motion_path))
                else:
                    styled_motion_len = 100
                
                if styled_motion_len < min_motion_len or styled_motion_len >= 400:
                    continue
                
                # =============================================================
                # Get the EXACT matching neutral motion
                # =============================================================
                neutral_sample = neutral_by_key[motion_key]
                
                coupled_sample = {
                    # Neutral motion & text (content)
                    'motion': neutral_sample['motion'].copy(),
                    'motion_length': neutral_sample['length'],
                    'text_list': neutral_sample['text'],
                    'neutral_name': neutral_sample['name'],
                    
                    # Styled video (style source)
                    'video_path': video_path,
                    'video_length_hint': styled_motion_len,
                    'style_label': style_to_label[style_name],
                    'style_name': style_name,
                    'styled_name': name,
                    
                    # Matching key
                    'motion_key': motion_key,
                }
                
                self.data_list.append(coupled_sample)
                
            except Exception as e:
                pass
        
        print(f"\nCreated {len(self.data_list)} coupled pairs")
        print(f"Skipped {skipped_no_match} styled samples (no matching neutral)")
        
        # Print distribution
        style_counts = {}
        type_counts = {}
        for sample in self.data_list:
            style_counts[sample['style_name']] = style_counts.get(sample['style_name'], 0) + 1
            # Extract motion_type from motion_key (remove M_ prefix and _XX suffix)
            key = sample['motion_key']
            mt = key.lstrip('M_').rsplit('_', 1)[0]
            type_counts[mt] = type_counts.get(mt, 0) + 1
        
        print(f"\nDistribution by style:")
        for style, count in sorted(style_counts.items()):
            print(f"  {style}: {count}")
        print(f"\nDistribution by motion type:")
        for mt, count in sorted(type_counts.items()):
            print(f"  {mt}: {count}")

        # =====================================================================
        # Sort by motion length for evaluation
        # =====================================================================
        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')
            self.data_list = sorted(self.data_list, key=lambda x: x['motion_length'])
        
        self.length_arr = np.array([s['motion_length'] for s in self.data_list])
        self.mean = mean[:dim_pose]
        self.std = std[:dim_pose]
        
        if self.evaluation:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.data_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        sample = self.data_list[idx]
        
        # =====================================================================
        # Extract pre-coupled data (EXACT match by motion_key)
        # =====================================================================
        motion = sample['motion'].copy()
        m_length = sample['motion_length']
        text_data = random.choice(sample['text_list'])  # Random caption from list
        
        style_label = sample['style_label']
        style_name = sample['style_name']
        video_path = sample['video_path']
        video_length_hint = sample['video_length_hint']
        
        # =====================================================================
        # Process motion length
        # =====================================================================
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        
        m_length = max(m_length, self.unit_length)
        start_idx = random.randint(0, max(0, len(motion) - m_length))
        motion = motion[start_idx:start_idx + m_length]

        # =====================================================================
        # Load styled video
        # =====================================================================
        try:
            vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
            video_length = len(vr)
            
            if video_length > 0:
                window_end = min(video_length - 1, video_length_hint - 1)
                frame_indices = np.linspace(0, window_end, num=self.max_vid_length, dtype=int)
                frame_indices = np.clip(frame_indices, 0, video_length - 1)
                frames = vr.get_batch(frame_indices.tolist()).asnumpy()
                video_snippet = list(frames)
            else:
                video_snippet = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length
                
        except Exception as e:
            print(f"Warning: Could not load video from {video_path}: {e}")
            video_snippet = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length

        # =====================================================================
        # Normalize and pad motion
        # =====================================================================
        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)
        elif m_length > self.max_motion_length:
            if not self.evaluation:
                crop_idx = random.randint(0, m_length - self.max_motion_length)
                motion = motion[crop_idx:crop_idx + self.max_motion_length]
                m_length = self.max_motion_length

        return text_data, motion, m_length, style_name, video_snippet


## Strict matching: Pairs a styled video only with the exact specific recording of the neutral motion
class Text2MotionVideoDataset_100styles_v4(data.Dataset):
    """
    Dataset that pairs:
    - Motion & Text: from NEUTRAL style (content)
    - Video & Styled Motion: from TARGET style (style transfer source)
    
    Matching is done by EXACT motion identifier (e.g., BR_02, M_BR_02)
    """
    
    def __init__(self, mean, std, split_file, dataset_name, video_dir, motion_dir, text_dir, 
                 dict_file, unit_length, max_motion_length, styles, dim_pose, 
                 max_vid_length=32, evaluation=False, neutral_style="Neutral"):
        
        assert styles is not None, "Styles must be provided"
        
        self.evaluation = evaluation
        self.styles = styles
        self.neutral_style = neutral_style
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_vid_length = max_vid_length
        self.unit_length = unit_length
        self.dim_pose = dim_pose
        min_motion_len = 40 if dataset_name in ['t2m', '100styles'] else 24

        # Build style label mapping
        style_to_label = {style: i for i, style in enumerate(self.styles)}
        
        # Load metadata
        metadata_dict = build_dict_from_txt_mirror(dict_file)
        
        # Load split IDs
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # =====================================================================
        # STEP 1: Build NEUTRAL motion dictionary (indexed by motion_key)
        # motion_key = "BR_02" or "M_BR_02" (includes mirror prefix)
        # =====================================================================
        neutral_by_key = {}  # {"BR_02": {motion, text, length, ...}, "M_BR_02": {...}}
        
        print(f"Loading NEUTRAL style motions...")
        for name in tqdm(id_list):
            try:
                _, style_name, motion_type, motion_idx, _, is_mirrored = metadata_dict[name]
                
                if style_name != self.neutral_style:
                    continue
                if motion_type.startswith("TR"):
                    continue
                
                motion_path = pjoin(motion_dir, name + '.npy')
                text_path = pjoin(text_dir, name + ".txt")
                
                if not os.path.exists(motion_path) or not os.path.exists(text_path):
                    continue
                    
                motion = np.load(motion_path)
                
                if len(motion) < min_motion_len or len(motion) >= 400:
                    continue
                
                # Load text
                text_data = []
                seen_captions = set()
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        caption = normalize_caption(caption)
                        if not caption:
                            continue
                        canonical_caption = caption.lower()
                        if canonical_caption in seen_captions:
                            continue
                        seen_captions.add(canonical_caption)
                        text_data.append(caption)
                
                if not text_data:
                    continue
                
                # =============================================================
                # Create motion_key: "BR_02" or "M_BR_02"
                # =============================================================
                motion_key = f"{motion_type}_{motion_idx}"
                if is_mirrored:
                    motion_key = f"M_{motion_key}"
                
                neutral_by_key[motion_key] = {
                    'name': name,
                    'motion': motion,
                    'length': len(motion),
                    'text': text_data,
                }
                
            except Exception as e:
                pass
        
        print(f"Loaded {len(neutral_by_key)} neutral motion keys")
        # Print some examples
        example_keys = list(neutral_by_key.keys())[:10]
        print(f"Example keys: {example_keys}")

        # =====================================================================
        # STEP 2: Build PRE-COUPLED dataset (styled video + matched neutral)
        # =====================================================================
        self.data_list = []
        skipped_no_match = 0
        skipped_no_styled_motion = 0
        
        print(f"\nCreating coupled (neutral_motion, styled_video, styled_motion) pairs...")
        for name in tqdm(id_list):
            try:
                _, style_name, motion_type, motion_idx, _, is_mirrored = metadata_dict[name]
                
                # Skip neutral (we use it for motion, not video)
                if style_name == self.neutral_style:
                    continue
                if style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):
                    continue
                
                # =============================================================
                # Create motion_key to find matching neutral
                # =============================================================
                motion_key = f"{motion_type}_{motion_idx}"
                if is_mirrored:
                    motion_key = f"M_{motion_key}"
                
                # Must have matching neutral motion
                if motion_key not in neutral_by_key:
                    skipped_no_match += 1
                    continue
                
                video_path = pjoin(video_dir, name + '_FV.mp4')
                if not os.path.exists(video_path):
                    continue
                
                # =============================================================
                # Load styled motion (motion corresponding to the video)
                # =============================================================
                styled_motion_path = pjoin(motion_dir, name + '.npy')
                if not os.path.exists(styled_motion_path):
                    skipped_no_styled_motion += 1
                    continue
                
                styled_motion = np.load(styled_motion_path)
                styled_motion_len = len(styled_motion)
                
                if styled_motion_len < min_motion_len or styled_motion_len >= 400:
                    continue
                
                # =============================================================
                # Get the EXACT matching neutral motion
                # =============================================================
                neutral_sample = neutral_by_key[motion_key]
                
                coupled_sample = {
                    # Neutral motion & text (content)
                    'neutral_motion': neutral_sample['motion'].copy(),
                    'neutral_length': neutral_sample['length'],
                    'text_list': neutral_sample['text'],
                    'neutral_name': neutral_sample['name'],
                    
                    # Styled motion (ground truth for style transfer)
                    'styled_motion': styled_motion.copy(),
                    'styled_length': styled_motion_len,
                    
                    # Styled video (style source)
                    'video_path': video_path,
                    'style_label': style_to_label[style_name],
                    'style_name': style_name,
                    'styled_name': name,
                    
                    # Matching key
                    'motion_key': motion_key,
                }
                
                self.data_list.append(coupled_sample)
                
            except Exception as e:
                pass
        
        print(f"\nCreated {len(self.data_list)} coupled pairs")
        print(f"Skipped {skipped_no_match} styled samples (no matching neutral)")
        print(f"Skipped {skipped_no_styled_motion} styled samples (no styled motion file)")
        
        # Print distribution
        style_counts = {}
        type_counts = {}
        for sample in self.data_list:
            style_counts[sample['style_name']] = style_counts.get(sample['style_name'], 0) + 1
            # Extract motion_type from motion_key (remove M_ prefix and _XX suffix)
            key = sample['motion_key']
            mt = key.lstrip('M_').rsplit('_', 1)[0]
            type_counts[mt] = type_counts.get(mt, 0) + 1
        
        print(f"\nDistribution by style:")
        for style, count in sorted(style_counts.items()):
            print(f"  {style}: {count}")
        print(f"\nDistribution by motion type:")
        for mt, count in sorted(type_counts.items()):
            print(f"  {mt}: {count}")

        # =====================================================================
        # Sort by motion length for evaluation (use styled_length for consistency)
        # =====================================================================
        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')
            self.data_list = sorted(self.data_list, key=lambda x: x['styled_length'])
        
        self.length_arr = np.array([s['styled_length'] for s in self.data_list])
        self.mean = mean[:dim_pose]
        self.std = std[:dim_pose]
        
        if self.evaluation:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.data_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        sample = self.data_list[idx]
        
        # =====================================================================
        # Extract pre-coupled data (EXACT match by motion_key)
        # =====================================================================
        neutral_motion = sample['neutral_motion'].copy()
        neutral_length = sample['neutral_length']
        
        styled_motion = sample['styled_motion'].copy()
        styled_length = sample['styled_length']
        
        text_data = random.choice(sample['text_list'])  # Random caption from list
        
        style_label = sample['style_label']
        style_name = sample['style_name']
        video_path = sample['video_path']
        
        # =====================================================================
        # Determine common length for both motions (use styled as reference)
        # =====================================================================
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        # Use the minimum of both lengths to ensure valid cropping
        min_length = min(neutral_length, styled_length)
        
        if coin2 == 'double':
            m_length = (min_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (min_length // self.unit_length) * self.unit_length
        
        m_length = max(m_length, self.unit_length)
        
        # Use same start_idx for both motions to keep them aligned
        start_idx = random.randint(0, max(0, min_length - m_length))
        
        neutral_motion = neutral_motion[start_idx:start_idx + m_length]
        styled_motion = styled_motion[start_idx:start_idx + m_length]

        # =====================================================================
        # Load styled video
        # =====================================================================
        try:
            vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
            video_length = len(vr)
            
            if video_length > 0:
                # Align video frames with the motion crop
                video_start = int(start_idx * video_length / styled_length)
                video_end = int((start_idx + m_length) * video_length / styled_length)
                video_end = min(video_end, video_length - 1)
                
                frame_indices = np.linspace(video_start, video_end, num=self.max_vid_length, dtype=int)
                frame_indices = np.clip(frame_indices, 0, video_length - 1)
                frames = vr.get_batch(frame_indices.tolist()).asnumpy()
                video_snippet = list(frames)
            else:
                video_snippet = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length
                
        except Exception as e:
            print(f"Warning: Could not load video from {video_path}: {e}")
            video_snippet = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length

        # =====================================================================
        # Normalize and pad motions
        # =====================================================================
        neutral_motion = neutral_motion[:, :self.mean.shape[0]]
        neutral_motion = (neutral_motion - self.mean) / self.std
        
        styled_motion = styled_motion[:, :self.mean.shape[0]]
        styled_motion = (styled_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            padding = np.zeros((self.max_motion_length - m_length, neutral_motion.shape[1]))
            neutral_motion = np.concatenate([neutral_motion, padding], axis=0)
            styled_motion = np.concatenate([styled_motion, padding.copy()], axis=0)
            
        elif m_length > self.max_motion_length:
            if not self.evaluation:
                crop_idx = random.randint(0, m_length - self.max_motion_length)
                neutral_motion = neutral_motion[crop_idx:crop_idx + self.max_motion_length]
                styled_motion = styled_motion[crop_idx:crop_idx + self.max_motion_length]
                m_length = self.max_motion_length

        # =====================================================================
        # Return: text, neutral_motion, styled_motion, m_length, style_name, video
        # =====================================================================
        return text_data, neutral_motion, styled_motion, m_length, style_name, video_snippet
    


class Text2MotionDatasetCombined(data.Dataset):
    def __init__(
        self,
        # 100STYLES Params
        style_mean, style_std, style_split_file, style_motion_dir, style_text_dir, style_video_dir, style_dict_file,
        # HumanML3D Params
        humanml_mean, humanml_std, humanml_split_file, humanml_motion_dir, humanml_text_dir, humanml_dict_file,  # FIXED: typo
        # Common Params
        dim_pose, unit_length, max_motion_length, min_motion_length=24, max_text_len=20, max_vid_length=32,
        styles=['Aeroplane','ArmsFolded','Chicken','Robot','Superman'], tiny=True, debug=False, progress_bar=True, evaluation=False,
        **kwargs):

        # 100STYLE Params
        self.style_mean = style_mean[:dim_pose]
        self.style_std = style_std[:dim_pose]
        self.styles = styles
        # HumanML3D Params
        self.humanml_mean = humanml_mean[:dim_pose]
        self.humanml_std = humanml_std[:dim_pose]
        
        # FIXED: Added missing instance variables
        self.dim_pose = dim_pose
        self.humanml_motion_dir = humanml_motion_dir

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.max_vid_length = max_vid_length
        self.evaluation = evaluation

        # --- Data Containers ---
        # Data_dict_1 = 100STYLES
        self.data_dict_style = {}
        self.id_list_style = []
        
        # Data_dict_2 = HumanML3D
        self.data_dict_humanml = {}
        self.id_list_humanml = []

        # --- Load Split Files ---
        # Load IDs for 100STYLES
        if os.path.exists(style_split_file):
            with cs.open(style_split_file, "r") as f:
                for line in f.readlines():
                    self.id_list_style.append(line.strip())
        else:
            raise FileNotFoundError(f"100STYLES split file not found: {style_split_file}")

        # Load IDs for HumanML3D
        if os.path.exists(humanml_split_file):
            with cs.open(humanml_split_file, "r") as f:
                for line in f.readlines():
                    self.id_list_humanml.append(line.strip())
        else:
            raise FileNotFoundError(f"HumanML3D split file not found: {humanml_split_file}")

        # Setup limits for debugging
        if tiny or debug:
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        # =========================================================
        # LOAD DATASET 1: 100STYLES (With Video & Style Filter)
        # =========================================================
        metadata_dict = build_dict_from_txt(style_dict_file)
        style_to_label = {style: i for i, style in enumerate(self.styles)} if self.styles else {}

        new_name_list_style = []
        length_list_style = []
        count = 0
        
        print(f"Loading 100STYLES data from {style_split_file}...")
        iterator_style = tqdm(self.id_list_style) if progress_bar else self.id_list_style
        
        for name in iterator_style:
            if count > maxdata: break
            try:
                # 1. Load Metadata & Filter
                _, style_name, motion_type, _, length = metadata_dict[name]
                
                if self.styles and style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):
                    continue

                # 2. Check Paths
                video_path = pjoin(style_video_dir, name + '_FV.mp4')
                motion_path = pjoin(style_motion_dir, name + '.npy')
                text_path = pjoin(style_text_dir, name + ".txt")

                if not (os.path.exists(motion_path) and os.path.exists(video_path) and os.path.exists(text_path)):
                    continue

                # 3. Length check
                if length < self.min_motion_length or length >= 400:
                    continue

                # 4. Load Text
                text_data_style = []
                seen_captions = set()
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        if not caption: continue
                        
                        canonical_caption = caption.lower()
                        if canonical_caption in seen_captions: continue
                        seen_captions.add(canonical_caption)
                        
                        text_data_style.append({'caption': caption, 'tokens': []})

                if len(text_data_style) > 0:
                    self.data_dict_style[name] = {
                        'motion': motion_path,
                        'length': length,
                        'style': style_to_label.get(style_name, 0),
                        'style_name': style_name,
                        'text': text_data_style,
                        'video': video_path
                    }
                    
                    new_name_list_style.append(name)
                    length_list_style.append(length)
                    count += 1
            except Exception as e:
                pass

        # Sort Dataset 1
        self.name_list_style, self.length_list_style = zip(*sorted(zip(new_name_list_style, length_list_style), key=lambda x: x[1]))
        self.length_arr_style = np.array(self.length_list_style)

        # =========================================================
        # LOAD DATASET 2: HumanML3D (With Timestamps & Splitting)
        # =========================================================
        metadata_dict_humanml = build_dict_from_txt2(humanml_dict_file)
        
        new_name_list_humanml = []
        length_list_humanml = []
        count = 0

        print(f"Loading HumanML3D data from {humanml_split_file}...")
        iterator_humanml = tqdm(self.id_list_humanml) if progress_bar else self.id_list_humanml

        for name in iterator_humanml:
            if count > maxdata: break
            try:
                total_length = metadata_dict_humanml[name]

                if total_length < self.min_motion_length or total_length >= 200:
                    continue
                
                text_data_humanml = []
                flag = False
                
                with cs.open(pjoin(humanml_text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data_humanml.append(text_dict)
                        else:
                            # Handle sub-segments
                            try:
                                start_idx = int(f_tag * 20)
                                end_idx = int(to_tag * 20)
                                seg_len = end_idx - start_idx
                                if seg_len < self.min_motion_length or seg_len >= 200:
                                    continue
                                
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in self.data_dict_humanml:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                
                                self.data_dict_humanml[new_name] = {
                                    'source_name': name,
                                    'start': start_idx,
                                    'end': end_idx,
                                    'length': seg_len,
                                    'text': [text_dict]
                                }
                                new_name_list_humanml.append(new_name)
                                length_list_humanml.append(seg_len)  # FIXED: removed len()
                            except:
                                pass

                if flag:
                    self.data_dict_humanml[name] = {
                        'source_name': name,
                        'start': 0,
                        'end': total_length,    
                        'length': total_length,
                        'text': text_data_humanml
                    }
                    new_name_list_humanml.append(name)
                    length_list_humanml.append(total_length)  # FIXED: removed len()
                    count += 1
            except Exception as e:
                pass

        # Sort Dataset 2
        self.name_list_humanml, self.length_list_humanml = zip(*sorted(zip(new_name_list_humanml, length_list_humanml), key=lambda x: x[1]))
        self.length_arr_humanml = np.array(self.length_list_humanml)

        # Final setup
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer_style = np.searchsorted(self.length_arr_style, length)
        self.max_length = length

    def inv_transform(self, data, mean, std):
        return data * std + mean

    def transform(self, data, mean, std):
        return (data - mean) / std

    def __len__(self):
        return len(self.data_dict_style) - self.pointer_style

    def __getitem__(self, item):
        try:
            return self._safe_getitem(item)
        except Exception as e:
            print(f"Critical Error in __getitem__ for index {item}: {e}")
            return self._safe_getitem(0)

    def _safe_getitem(self, item):
        # ==============================
        # 1. Get 100STYLES Sample (Indexed)
        # ==============================
        idx_style = self.pointer_style + item
        name_style = self.name_list_style[idx_style]
        data_style = self.data_dict_style[name_style]
        
        motion_style_path = data_style['motion']
        m_length_style = data_style['length']
        text_list_style = data_style['text']
        style_name_style = data_style['style_name']
        video_path_style = data_style['video']

        # Random caption
        text_item = random.choice(text_list_style)
        caption_style = text_item['caption']

        # ==============================
        # 2. Get HumanML3D Sample (Random)
        # ==============================
        idx_humanml = random.randint(0, len(self.name_list_humanml) - 1)
        name_humanml = self.name_list_humanml[idx_humanml]
        meta_humanml = self.data_dict_humanml[name_humanml]

        source_name = meta_humanml['source_name']
        start = meta_humanml['start']
        end = meta_humanml['end']
        m_length_humanml = meta_humanml['length']  # FIXED: consistent naming
        text_list_humanml = meta_humanml['text']

        motion_path = pjoin(self.humanml_motion_dir, source_name + '.npy')
        full_motion = np.load(motion_path)

        # Apply the slice
        real_len = len(full_motion)
        safe_start = min(start, real_len - 1)
        safe_end = min(end, real_len)
        
        motion_humanml = full_motion[safe_start:safe_end]
        motion_humanml = motion_humanml[:, :self.dim_pose]

        # Random caption
        text_data_humanml = random.choice(text_list_humanml)
        caption_humanml = text_data_humanml['caption']  # FIXED: consistent naming

        # ==============================
        # 3. Process Motions (Cropping & Normalization)
        # ==============================
        motion_style = np.load(motion_style_path)[:, :self.dim_pose]

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length_style = (m_length_style // self.unit_length - 1) * self.unit_length
            m_length_humanml = (m_length_humanml // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length_style = (m_length_style // self.unit_length) * self.unit_length
            m_length_humanml = (m_length_humanml // self.unit_length) * self.unit_length

        # Apply Max Length Cap
        m_length_style = min(self.max_motion_length, m_length_style)
        m_length_humanml = min(self.max_motion_length, m_length_humanml)

        # Crop 100STYLES - FIXED: added bounds checking
        max_start_style = max(0, len(motion_style) - m_length_style)
        start_idx_style = random.randint(0, max_start_style)
        motion_style = motion_style[start_idx_style : start_idx_style + m_length_style]
        
        # Crop HumanML3D - FIXED: added bounds checking
        max_start_humanml = max(0, len(motion_humanml) - m_length_humanml)
        start_idx_humanml = random.randint(0, max_start_humanml)
        motion_humanml = motion_humanml[start_idx_humanml : start_idx_humanml + m_length_humanml]

        # Z-Normalization
        motion_style = (motion_style - self.style_mean) / self.style_std
        motion_humanml = (motion_humanml - self.humanml_mean) / self.humanml_std

        # ==============================
        # 4. Load Video (100STYLES Only)
        # ==============================
        try:
            vr = VideoReader(video_path_style, ctx=cpu(0), width=224, height=224)
            video_length = len(vr)
            if video_length > 0:
                window_start = min(start_idx_style, max(video_length - 1, 0))
                window_end = min(start_idx_style + m_length_style - 1, video_length - 1)
                
                if window_end <= window_start:
                    window_end = window_start
                
                frame_indices = np.linspace(window_start, window_end, num=self.max_vid_length, dtype=int)
                frame_indices = frame_indices.tolist()
                
                while len(frame_indices) < self.max_vid_length:
                    frame_indices.append(frame_indices[-1])
                
                frame_indices = np.array(frame_indices[:self.max_vid_length], dtype=int)
                frames = vr.get_batch(frame_indices).asnumpy()
                video_snippet_style = list(frames)
        except Exception as e:
            raise ValueError(f"Failed to load video {video_path_style}: {e}")

        # ==============================
        # 5. Pad to max_motion_length
        # ==============================
        if m_length_style < self.max_motion_length:
            padding_len = self.max_motion_length - len(motion_style)
            padding = np.zeros((padding_len, motion_style.shape[1]))
            motion_style = np.concatenate([motion_style, padding], axis=0)
        
        if len(motion_humanml) < self.max_motion_length:
            padding_len = self.max_motion_length - len(motion_humanml)
            padding = np.zeros((padding_len, motion_humanml.shape[1]))
            motion_humanml = np.concatenate([motion_humanml, padding], axis=0)
                
        # ==============================
        # 6. Check for NaNs
        # ==============================
        if np.any(np.isnan(motion_style)) or np.any(np.isnan(motion_humanml)):
            print(f"⚠️ Warning: NaN found in sample '{style_name_style}'. Retrying with new random index...")
            new_item = random.randint(0, len(self) - 1)
            return self.__getitem__(new_item)

        # ==============================
        # 7. Return Combined Data - FIXED: consistent variable names
        # ==============================
        return {
            'text_styled': caption_style,
            'motion_styled': motion_style,
            'length_styled': m_length_style,
            'style_name': style_name_style,
            'video_styled': video_snippet_style,
            
            'text_humanml': caption_humanml,  # FIXED: was caption_humanml3d
            'motion_humanml': motion_humanml,  # FIXED: was motion_humanml3d
            'length_humanml': m_length_humanml  # FIXED: was m_length_humanml3d
        }


class Text2MotionDatasetCombined_v2(data.Dataset):
    """
    Combined dataset for Option 7 (Hybrid) training.
    
    Provides:
    1. 100STYLES: Paired (neutral_motion, styled_motion, neutral_video, styled_video) for cycle consistency
    2. HumanML3D: (text, motion) for text-to-content learning with zero style
    
    Key Features:
    - Exact matching between neutral and styled motions by motion_key
    - Same crop window applied to both neutral and styled
    - BOTH neutral and styled videos loaded and aligned with motion crop
    - HumanML3D sampled randomly (not paired with 100STYLES)
    """
    
    def __init__(
        self,
        # 100STYLES Params
        style_mean, style_std, style_split_file, style_motion_dir, style_text_dir, style_video_dir, style_dict_file,
        # HumanML3D Params
        humanml_mean, humanml_std, humanml_split_file, humanml_motion_dir, humanml_text_dir, humanml_dict_file,  # FIXED: typo
        # Common Params
        dim_pose, unit_length, max_motion_length, 
        min_motion_length=40, max_text_len=20, max_vid_length=32,
        styles=['Aeroplane', 'ArmsFolded', 'Chicken', 'Robot', 'Superman'],
        neutral_style='Neutral',
        tiny=True, debug=False, progress_bar=True, evaluation=False,
        **kwargs
    ):
        
        # =====================================================================
        # Store parameters
        # =====================================================================
        self.style_mean = style_mean[:dim_pose]
        self.style_std = style_std[:dim_pose]
        self.humanml_mean = humanml_mean[:dim_pose]
        self.humanml_std = humanml_std[:dim_pose]
        self.dim_pose = dim_pose
        
        self.styles = styles
        self.neutral_style = neutral_style
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.max_vid_length = max_vid_length
        self.evaluation = evaluation
        self.video_dir = style_video_dir
        
        # FIXED: Added missing instance variable
        self.humanml_motion_dir = humanml_motion_dir
        
        self.max_length = 20
        self.pointer = 0
        
        # =====================================================================
        # Load split files
        # =====================================================================
        id_list_style = []
        if os.path.exists(style_split_file):
            with cs.open(style_split_file, "r") as f:
                for line in f.readlines():
                    id_list_style.append(line.strip())
        else:
            raise FileNotFoundError(f"100STYLES split file not found: {style_split_file}")
        
        id_list_humanml = []
        if os.path.exists(humanml_split_file):
            with cs.open(humanml_split_file, "r") as f:
                for line in f.readlines():
                    id_list_humanml.append(line.strip())
        else:
            raise FileNotFoundError(f"HumanML3D split file not found: {humanml_split_file}")
        
        # Debug limits
        if tiny:
            maxdata = 10
        elif debug:
            maxdata = 100
        else:
            maxdata = float('inf')
        
        # =====================================================================
        # STEP 1: Build NEUTRAL motion dictionary
        # =====================================================================
        metadata_dict = build_dict_from_txt_mirror(style_dict_file)
        style_to_label = {style: i for i, style in enumerate(self.styles)}
        
        neutral_by_key = {}
        
        print(f"Loading NEUTRAL style motions...")
        iterator = tqdm(id_list_style) if progress_bar else id_list_style
        
        for name in iterator:
            try:
                _, style_name, motion_type, motion_idx, is_mirrored, length = metadata_dict[name]
                
                if style_name != self.neutral_style:
                    continue
                if motion_type.startswith("TR"):
                    continue
                
                motion_path = pjoin(style_motion_dir, name + '.npy')
                text_path = pjoin(style_text_dir, name + '.txt')
                video_path = pjoin(style_video_dir, name + '_FV.mp4')
                
                if not os.path.exists(motion_path) or not os.path.exists(text_path):
                    continue
                
                has_video = os.path.exists(video_path)
                
                if length < self.min_motion_length or length >= 400:
                    continue
                
                # Load text
                text_data = []
                seen_captions = set()
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        caption = normalize_caption(caption)
                        if not caption:
                            continue
                        canonical = caption.lower()
                        if canonical in seen_captions:
                            continue
                        seen_captions.add(canonical)
                        text_data.append(caption)
                
                if not text_data:
                    continue
                
                motion_key = f"{motion_type}_{motion_idx}"
                if is_mirrored:
                    motion_key = f"M_{motion_key}"
                
                neutral_by_key[motion_key] = {
                    'name': name,
                    'motion': motion_path,
                    'length': length,
                    'text': text_data,
                    'video_path': video_path if has_video else None,
                    'has_video': has_video,
                }
                
            except Exception as e:
                pass
        
        print(f"Loaded {len(neutral_by_key)} neutral motion keys")
        neutral_with_video = sum(1 for v in neutral_by_key.values() if v['has_video'])
        print(f"  {neutral_with_video} have videos, {len(neutral_by_key) - neutral_with_video} without")
        
        # =====================================================================
        # STEP 2: Build PAIRED dataset
        # =====================================================================
        self.data_list_paired = []
        skipped_no_match = 0
        skipped_no_styled_video = 0
        skipped_no_neutral_video = 0
        count = 0
        
        print(f"\nCreating paired (neutral, styled, videos) samples...")
        iterator = tqdm(id_list_style) if progress_bar else id_list_style
        
        for name in iterator:
            if count >= maxdata:
                break
            
            try:
                _, style_name, motion_type, motion_idx, style_length, is_mirrored = metadata_dict[name]
                
                if style_name == self.neutral_style:
                    continue
                if style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):
                    continue
                
                motion_key = f"{motion_type}_{motion_idx}"
                if is_mirrored:
                    motion_key = f"M_{motion_key}"
                
                if motion_key not in neutral_by_key:
                    skipped_no_match += 1
                    continue
                
                neutral_sample = neutral_by_key[motion_key]
                
                styled_video_path = pjoin(style_video_dir, name + '_FV.mp4')
                styled_motion_path = pjoin(style_motion_dir, name + '.npy')
                
                if not os.path.exists(styled_video_path):
                    skipped_no_styled_video += 1
                    continue
                if not os.path.exists(styled_motion_path):
                    continue
                
                neutral_video_path = neutral_sample['video_path']
                has_neutral_video = neutral_sample['has_video']
                
                if not has_neutral_video:
                    skipped_no_neutral_video += 1
                
                if style_length < self.min_motion_length or style_length >= 400:
                    continue

                # Load styled text
                styled_text_path = pjoin(style_text_dir, name + '.txt')
                styled_text_list = []
                if os.path.exists(styled_text_path):
                    seen_styled = set()
                    with cs.open(styled_text_path) as f:
                        for line in f.readlines():
                            caption = line.strip().split("#")[0]
                            caption = normalize_caption(caption)
                            if not caption:
                                continue
                            canonical = caption.lower()
                            if canonical in seen_styled:
                                continue
                            seen_styled.add(canonical)
                            styled_text_list.append(caption)
                
                if not styled_text_list:
                    styled_text_list = neutral_sample['text']
                
                paired_sample = {
                    'neutral_motion': neutral_sample['motion'],
                    'neutral_length': neutral_sample['length'],
                    'neutral_text_list': neutral_sample['text'],
                    'neutral_video_path': neutral_video_path,
                    'neutral_name': neutral_sample['name'],
                    'has_neutral_video': has_neutral_video,
                    
                    'styled_motion': styled_motion_path,
                    'styled_length': style_length,
                    'styled_text_list': styled_text_list,
                    'styled_video_path': styled_video_path,
                    'styled_name': name,
                    
                    'style_label': style_to_label.get(style_name, 0),
                    'style_name': style_name,
                    
                    'motion_key': motion_key,
                }
                
                self.data_list_paired.append(paired_sample)
                count += 1
                
            except Exception as e:
                pass
        
        print(f"\nCreated {len(self.data_list_paired)} paired samples")
        print(f"Skipped {skipped_no_match} (no matching neutral)")
        print(f"Skipped {skipped_no_styled_video} (no styled video)")
        print(f"Note: {skipped_no_neutral_video} samples missing neutral video (will use zero tensor)")
        
        # Print distribution
        style_counts = {}
        for sample in self.data_list_paired:
            s = sample['style_name']
            style_counts[s] = style_counts.get(s, 0) + 1
        print(f"\nPaired samples by style:")
        for style, cnt in sorted(style_counts.items()):
            print(f"  {style}: {cnt}")
        
        # =====================================================================
        # STEP 3: Load HumanML3D dataset
        # =====================================================================
        metadata_dict_humanml = build_dict_from_txt2(humanml_dict_file)

        self.data_dict_humanml = {}
        new_name_list_humanml = []
        length_list_humanml = []
        count = 0
        
        print(f"\nLoading HumanML3D data...")
        iterator = tqdm(id_list_humanml) if progress_bar else id_list_humanml
        
        for name in iterator:
            if count >= maxdata:
                break
            
            try:
                _, total_length = metadata_dict_humanml[name]
                
                if total_length < self.min_motion_length or total_length >= 200:
                    continue
                
                text_data = []
                flag = False
                
                with cs.open(pjoin(humanml_text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ') if len(line_split) > 1 else []
                        f_tag = float(line_split[2]) if len(line_split) > 2 else 0.0
                        to_tag = float(line_split[3]) if len(line_split) > 3 else 0.0
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                start_idx = int(f_tag * 20)
                                end_idx = int(to_tag * 20)
                                seg_len = end_idx - start_idx
                                if seg_len < self.min_motion_length or seg_len >= 200:
                                    continue
                                
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in self.data_dict_humanml:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                
                                self.data_dict_humanml[new_name] = {
                                    'source_name': name,
                                    'start': start_idx,
                                    'end': end_idx,
                                    'length': seg_len,
                                    'text': [text_dict]
                                }
                                new_name_list_humanml.append(new_name)
                                length_list_humanml.append(seg_len)  # FIXED: removed len()
                            except:
                                pass
                
                if flag:
                    self.data_dict_humanml[name] = {
                        'source_name': name,
                        'start': 0,
                        'end': total_length,    
                        'length': total_length,
                        'text': text_data
                    }
                    new_name_list_humanml.append(name)
                    length_list_humanml.append(total_length)  # FIXED: removed len()
                    count += 1   
            except:
                pass
        
        # Sort HumanML3D by length
        if len(new_name_list_humanml) > 0:
            self.name_list_humanml, self.length_list_humanml = zip(
                *sorted(zip(new_name_list_humanml, length_list_humanml), key=lambda x: x[1])
            )
            self.length_arr_humanml = np.array(self.length_list_humanml)
        else:
            self.name_list_humanml = []
            self.length_arr_humanml = np.array([])
        
        print(f"Loaded {len(self.data_dict_humanml)} HumanML3D samples")
        
        # =====================================================================
        # Sort paired data by styled_length
        # =====================================================================
        if self.evaluation:
            self.data_list_paired = sorted(self.data_list_paired, key=lambda x: x['styled_length'])
        
        self.length_arr_paired = np.array([s['styled_length'] for s in self.data_list_paired])
        
        if self.evaluation:
            self.reset_max_len(self.max_length)
        
        print(f"\n{'='*60}")
        print(f"Dataset Summary:")
        print(f"  100STYLES paired: {len(self.data_list_paired)}")
        print(f"  HumanML3D: {len(self.data_dict_humanml)}")
        print(f"{'='*60}")
    
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr_paired, length)
        self.max_length = length
        print(f"Pointer set to {self.pointer}")
    
    def __len__(self):
        return len(self.data_list_paired) - self.pointer

    def __getitem__(self, item):
        try:
            return self._safe_getitem(item)
        except Exception as e:
            print(f"Critical Error in __getitem__ for index {item}: {e}")
            return self._safe_getitem(0)
    
    def _safe_getitem(self, item):
        # =====================================================================
        # 1. Get PAIRED 100STYLES sample (indexed)
        # =====================================================================
        idx = self.pointer + item
        sample = self.data_list_paired[idx]
        
        neutral_motion_path = sample['neutral_motion']
        neutral_length = sample['neutral_length']
        neutral_text_list = sample['neutral_text_list']
        neutral_video_path = sample['neutral_video_path']
        has_neutral_video = sample['has_neutral_video']
        
        styled_motion_path = sample['styled_motion']
        styled_length = sample['styled_length']
        styled_text_list = sample['styled_text_list']
        styled_video_path = sample['styled_video_path']
        
        style_name = sample['style_name']
        
        text_neutral = random.choice(neutral_text_list)
        text_styled = random.choice(styled_text_list)
        
        # =====================================================================
        # 2. Get RANDOM HumanML3D sample
        # =====================================================================
        idx_humanml = random.randint(0, len(self.name_list_humanml) - 1)
        name_humanml = self.name_list_humanml[idx_humanml]
        meta_humanml = self.data_dict_humanml[name_humanml]
        
        source_name = meta_humanml['source_name']
        start = meta_humanml['start']
        end = meta_humanml['end']
        m_length_humanml = meta_humanml['length']
        text_list_humanml = meta_humanml['text']

        motion_path = pjoin(self.humanml_motion_dir, source_name + '.npy')
        full_motion = np.load(motion_path)

        real_len = len(full_motion)
        safe_start = min(start, real_len - 1)
        safe_end = min(end, real_len)
        
        motion_humanml = full_motion[safe_start:safe_end]
        motion_humanml = motion_humanml[:, :self.dim_pose]
        
        text_data_humanml = random.choice(text_list_humanml)
        text_humanml = text_data_humanml['caption']
        
        # =====================================================================
        # 3. Process PAIRED motions (same crop for both)
        # =====================================================================
        neutral_motion = np.load(neutral_motion_path)[:, :self.dim_pose]
        styled_motion = np.load(styled_motion_path)[:, :self.dim_pose]

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        
        min_length = min(neutral_length, styled_length)
        
        if coin2 == 'double':
            m_length = (min_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (min_length // self.unit_length) * self.unit_length
        
        m_length = max(m_length, self.unit_length)
        m_length = min(m_length, self.max_motion_length)
        
        start_idx = random.randint(0, max(0, min_length - m_length))
        end_idx = start_idx + m_length
        
        neutral_motion = neutral_motion[start_idx:end_idx]
        styled_motion = styled_motion[start_idx:end_idx]
        
        # =====================================================================
        # 4. Process HumanML3D motion (independent crop)
        # =====================================================================
        if coin2 == 'double':
            target_len = (m_length_humanml // self.unit_length - 1) * self.unit_length
        else:
            target_len = (m_length_humanml // self.unit_length) * self.unit_length
        
        # FIXED: Use target_len consistently
        target_len = max(target_len, self.unit_length)
        target_len = min(target_len, self.max_motion_length)

        current_available_len = len(motion_humanml)
        if target_len > current_available_len:
            target_len = (current_available_len // self.unit_length) * self.unit_length

        max_start = max(0, current_available_len - target_len)
        start_idx_hml = random.randint(0, max_start)  # FIXED: renamed to avoid confusion
        
        motion_humanml = motion_humanml[start_idx_hml : start_idx_hml + target_len]
        
        # =====================================================================
        # 5. Load BOTH videos (aligned with motion crop)
        # =====================================================================
        
        # Load STYLED video (always available)
        try:
            vr = VideoReader(styled_video_path, ctx=cpu(0), width=224, height=224)
            video_length = len(vr)
            
            if video_length > 0 and styled_length > 0:  # FIXED: added division by zero check
                video_start = int(start_idx * video_length / styled_length)
                video_end = int((start_idx + m_length) * video_length / styled_length)
                video_end = min(video_end, video_length - 1)
                
                frame_indices = np.linspace(video_start, video_end, num=self.max_vid_length, dtype=int)
                frame_indices = np.clip(frame_indices, 0, video_length - 1)
                frames = vr.get_batch(frame_indices.tolist()).asnumpy()
                video_styled = list(frames)
            else:
                video_styled = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length
                
        except Exception as e:
            print(f"Warning: Could not load video {styled_video_path}: {e}")
            video_styled = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length

        # FIXED: Initialize video_neutral before the try block
        video_neutral = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length
        
        # Load NEUTRAL video (may not exist)
        if has_neutral_video and neutral_video_path:
            try:
                vr = VideoReader(neutral_video_path, ctx=cpu(0), width=224, height=224)
                video_length = len(vr)
                
                if video_length > 0 and neutral_length > 0:  # FIXED: added division by zero check
                    video_start = int(start_idx * video_length / neutral_length)
                    video_end = int((start_idx + m_length) * video_length / neutral_length)
                    video_end = min(video_end, video_length - 1)
                    
                    frame_indices = np.linspace(video_start, video_end, num=self.max_vid_length, dtype=int)
                    frame_indices = np.clip(frame_indices, 0, video_length - 1)
                    frames = vr.get_batch(frame_indices.tolist()).asnumpy()
                    video_neutral = list(frames)
            except Exception as e:
                print(f"Warning: Could not load video {neutral_video_path}: {e}")
                # Keep the initialized zero frames
        
        # =====================================================================
        # 6. Normalize motions
        # =====================================================================
        neutral_motion = neutral_motion[:, :self.dim_pose]
        neutral_motion = (neutral_motion - self.style_mean) / self.style_std
        
        styled_motion = styled_motion[:, :self.dim_pose]
        styled_motion = (styled_motion - self.style_mean) / self.style_std
        
        motion_humanml = motion_humanml[:, :self.dim_pose]
        motion_humanml = (motion_humanml - self.humanml_mean) / self.humanml_std
        
        # =====================================================================
        # 7. Pad to max_motion_length
        # =====================================================================
        if m_length < self.max_motion_length:
            padding = np.zeros((self.max_motion_length - m_length, neutral_motion.shape[1]))
            neutral_motion = np.concatenate([neutral_motion, padding], axis=0)
            styled_motion = np.concatenate([styled_motion, padding.copy()], axis=0)
        
        if target_len < self.max_motion_length:  # FIXED: use target_len instead of m_length_humanml
            padding = np.zeros((self.max_motion_length - target_len, motion_humanml.shape[1]))
            motion_humanml = np.concatenate([motion_humanml, padding], axis=0)
        
        # =====================================================================
        # 8. Check for NaNs
        # =====================================================================
        if np.any(np.isnan(neutral_motion)) or np.any(np.isnan(styled_motion)) or np.any(np.isnan(motion_humanml)):
            bad_name = sample.get('styled_name', 'Unknown')
            print(f"Warning: NaN found in sample '{bad_name}' (Index {idx}). Retrying with random sample...")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)
        
        # =====================================================================
        # 9. Return combined data as dictionary
        # =====================================================================
        return {
            'text_neutral': text_neutral,
            'motion_neutral': neutral_motion,
            'length_neutral': m_length,  # FIXED: return actual cropped length
            'video_neutral': video_neutral,
            
            'text_styled': text_styled,
            'motion_styled': styled_motion,
            'length_styled': m_length,  # FIXED: return actual cropped length (same as neutral)
            'video_styled': video_styled,
            
            'length_common': m_length,
            
            'style_name': style_name,
            
            'text_humanml': text_humanml,
            'motion_humanml': motion_humanml,
            'length_humanml': target_len,  # FIXED: return actual cropped length
        }


class Text2MotionDatasetCombined_v4(data.Dataset):
    def __init__(
        self,
        # 100STYLES Params
        style_mean, style_std, style_split_file, style_motion_dir, style_text_dir, style_video_dir, style_dict_file,
        # HumanML3D Params
        humanml_mean, humanml_std, humanml_split_file, humnaml_motion_dir, humanml_latent_dir, humanml_text_dir, humanml_dict_file,  # CHANGED: humanml_latent_dir
        # Common Params
        dim_pose, unit_length, max_motion_length, min_motion_length=40, max_text_len=20, max_vid_length=32,
        styles=['Aeroplane','ArmsFolded','Chicken','Robot','Superman'], tiny=False, debug=False, progress_bar=True, evaluation=False,
        epoch_mode='100styles',
        **kwargs):

        self.epoch_mode = epoch_mode
        
        # 100STYLE Params
        self.style_mean = style_mean[:dim_pose]
        self.style_std = style_std[:dim_pose]
        self.styles = styles
        
        # HumanML3D Params (Kept for compatibility, but ignored for latents)
        self.humanml_mean = humanml_mean[:dim_pose]
        self.humanml_std = humanml_std[:dim_pose]
        
        self.dim_pose = dim_pose
        self.humanml_motion_dir = humnaml_motion_dir  
        self.humanml_latent_dir = humanml_latent_dir

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.max_vid_length = max_vid_length
        self.evaluation = evaluation

        # --- Data Containers ---
        self.data_dict_style = {}
        self.id_list_style = []
        
        self.data_dict_humanml = {}
        self.id_list_humanml = []

        # --- Load Split Files ---
        if os.path.exists(style_split_file):
            with cs.open(style_split_file, "r") as f:
                for line in f.readlines():
                    self.id_list_style.append(line.strip())
        else:
            raise FileNotFoundError(f"100STYLES split file not found: {style_split_file}")

        if os.path.exists(humanml_split_file):
            with cs.open(humanml_split_file, "r") as f:
                for line in f.readlines():
                    self.id_list_humanml.append(line.strip())
        else:
            raise FileNotFoundError(f"HumanML3D split file not found: {humanml_split_file}")

        if tiny or debug:
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        # =========================================================
        # LOAD DATASET 1: 100STYLES (Raw Motion)
        # =========================================================
        metadata_dict = build_dict_from_txt(style_dict_file)
        style_to_label = {style: i for i, style in enumerate(self.styles)} if self.styles else {}

        new_name_list_style = []
        length_list_style = []
        count = 0
        
        print(f"Loading 100STYLES data from {style_split_file}...")
        iterator_style = tqdm(self.id_list_style) if progress_bar else self.id_list_style
        
        for name in iterator_style:
            if count > maxdata: break
            try:
                _, style_name, motion_type, _, length = metadata_dict[name]
                
                if self.styles and style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):
                    continue

                video_path = pjoin(style_video_dir, name + '_FV.mp4')
                motion_path = pjoin(style_motion_dir, name + '.npy')
                text_path = pjoin(style_text_dir, name + ".txt")

                if not (os.path.exists(motion_path) and os.path.exists(video_path) and os.path.exists(text_path)):
                    continue

                if length < self.min_motion_length or length >= 400:
                    continue

                text_data_style = []
                seen_captions = set()
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        if not caption: continue
                        
                        canonical_caption = caption.lower()
                        if canonical_caption in seen_captions: continue
                        seen_captions.add(canonical_caption)
                        
                        text_data_style.append({'caption': caption, 'tokens': []})

                
                    self.data_dict_style[name] = {
                        'motion': motion_path,
                        'length': length,
                        'style': style_to_label.get(style_name, 0),
                        'style_name': style_name,
                        'text': text_data_style,
                        'video': video_path
                    }
                    
                    new_name_list_style.append(name)
                    length_list_style.append(length)
                    count += 1
            except Exception as e:
                pass

        self.name_list_style, self.length_list_style = zip(*sorted(zip(new_name_list_style, length_list_style), key=lambda x: x[1]))
        self.length_arr_style = np.array(self.length_list_style)

        # =========================================================
        # LOAD DATASET 2: HumanML3D (Pre-Encoded Latents)
        # =========================================================
        metadata_dict_humanml = build_dict_from_txt2(humanml_dict_file)
        
        new_name_list_humanml = []
        length_list_humanml = []
        count = 0

        print(f"Loading HumanML3D data from {humanml_split_file}...")
        iterator_humanml = tqdm(self.id_list_humanml) if progress_bar else self.id_list_humanml
        for name in iterator_humanml:
            if count > maxdata: break
            try:
                if len(text_data_style) > 0:
                 total_length = metadata_dict_humanml[name]
                
                # Just read the raw string
                with cs.open(pjoin(humanml_text_dir, name + '.txt'), 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                
                # Store only what is actually needed
                self.data_dict_humanml[name] = {
                    'source_name': name,
                    'length': total_length,
                    'text': caption  # Just the string, no lists or dicts!
                }
                
                new_name_list_humanml.append(name)
                length_list_humanml.append(total_length)
                count += 1
            except Exception as e:
                pass

        # Safe zip
        if len(new_name_list_humanml) > 0:
            self.name_list_humanml, self.length_list_humanml = zip(*sorted(zip(new_name_list_humanml, length_list_humanml), key=lambda x: x[1]))
            self.length_arr_humanml = np.array(self.length_list_humanml)
        else:
            self.name_list_humanml = []
            self.length_list_humanml = []
            self.length_arr_humanml = np.array([])
            print("WARNING: HumanML3D list is empty!")
            
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer_style = np.searchsorted(self.length_arr_style, length)
        self.max_length = length

    def inv_transform(self, data, mean, std):
        return data * std + mean

    def transform(self, data, mean, std):
        return (data - mean) / std

    def __len__(self):
        if self.epoch_mode == '100styles':
            return len(self.data_dict_style) - self.pointer_style
        elif self.epoch_mode == 'humanml3d':
            return len(self.name_list_humanml)
        else:
            raise ValueError("epoch_mode must be '100styles' or 'humanml3d'")

    def __getitem__(self, item):
        try:
            return self._safe_getitem(item)
        except Exception as e:
            print(f"Critical Error in __getitem__ for index {item}: {e}")
            return self._safe_getitem(0)

    def _safe_getitem(self, item):
        # ==============================
        # 1. Get 100STYLES Sample (Raw Motion)
        # ==============================
        if self.epoch_mode == '100styles':
            idx_style = self.pointer_style + item
        else:
            total_style_items = len(self.name_list_style) - self.pointer_style
            idx_style = self.pointer_style + (item % total_style_items)
            
        name_style = self.name_list_style[idx_style]
        data_style = self.data_dict_style[name_style]
        
        motion_style_path = data_style['motion']
        m_length_style = data_style['length']
        text_list_style = data_style['text']
        style_name_style = data_style['style_name']
        video_path_style = data_style['video']

        text_item = random.choice(text_list_style)
        caption_style = text_item['caption']

        # ==============================
        # 2. Get HumanML3D Sample (Pre-Sliced Latent)
        # ==============================
        if self.epoch_mode == 'humanml3d':
            idx_humanml = item
        else:
            idx_humanml = random.randint(0, len(self.name_list_humanml) - 1)
            
        # name_humanml is now exactly the slice name (e.g., "00001_0")
        name_humanml = self.name_list_humanml[idx_humanml]

        # Use the dictionary you built in __init__()!
        meta_humanml = self.data_dict_humanml[name_humanml]
        m_length_humanml = meta_humanml['length']
        caption_humanml = meta_humanml['text']

        # A. Load Latent  and motion both 
        motion_path = pjoin(self.humanml_motion_dir , name_humanml + '.npy')
        motion_humanml = np.load(motion_path)[:, :self.dim_pose]  

        latent_path = pjoin(self.humanml_latent_dir, name_humanml + '.npy')
        latent_humanml = np.load(latent_path)

        # Clean shape to guarantee [512, T]
        if latent_humanml.ndim == 3 and latent_humanml.shape[0] == 1:
            latent_humanml = latent_humanml[0]
        if latent_humanml.shape[0] != 512 and latent_humanml.shape[-1] == 512:
            latent_humanml = latent_humanml.transpose(1, 0)


       
        # ==============================
        # 3. Process Motions (Cropping & Normalization)
        # ==============================
        motion_style = np.load(motion_style_path)[:, :self.dim_pose]

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length_style = (m_length_style // self.unit_length - 1) * self.unit_length
            # m_length_humanml = (m_length_humanml // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length_style = (m_length_style // self.unit_length) * self.unit_length
            # m_length_humanml = (m_length_humanml // self.unit_length) * self.unit_length

        m_length_style = min(self.max_motion_length, m_length_style)
        # m_length_humanml = min(self.max_motion_length, m_length_humanml)

        # Crop 100STYLES
        max_start_style = max(0, len(motion_style) - m_length_style)
        start_idx_style = random.randint(0, max_start_style)
        motion_style = motion_style[start_idx_style : start_idx_style + m_length_style]
        
        # Crop HumanML3D (Latent)
        # latent_m_length = m_length_humanml // self.unit_length
        latent_m_length = m_length_humanml
        max_start_latent = max(0, latent_humanml.shape[1] - latent_m_length)
        start_idx_latent = random.randint(0, max_start_latent)
        latent_humanml = latent_humanml[:, start_idx_latent : start_idx_latent + latent_m_length]

        # Crop HumanML3D (Raw Motion) to sync with Latent
        start_idx_raw_humanml = start_idx_latent * self.unit_length
        raw_m_length_humanml = m_length_humanml * self.unit_length
        motion_humanml = motion_humanml[start_idx_raw_humanml : start_idx_raw_humanml + raw_m_length_humanml]
        # motion_humanml = motion_humanml[start_idx_raw_humanml : start_idx_raw_humanml + m_length_humanml]
        

        # Z-Normalization ONLY applied to raw 100STYLES motion
        motion_style = (motion_style - self.style_mean) / self.style_std
        motion_humanml = (motion_humanml - self.humanml_mean) / self.humanml_std

        # ==============================
        # 4. Load Video (100STYLES Only)
        # ==============================
        try:
            vr = VideoReader(video_path_style, ctx=cpu(0), width=224, height=224)
            video_length = len(vr)
            if video_length > 0:
                window_start = min(start_idx_style, max(video_length - 1, 0))
                window_end = min(start_idx_style + m_length_style - 1, video_length - 1)
                
                if window_end <= window_start:
                    window_end = window_start
                
                frame_indices = np.linspace(window_start, window_end, num=self.max_vid_length, dtype=int)
                frame_indices = frame_indices.tolist()
                
                while len(frame_indices) < self.max_vid_length:
                    frame_indices.append(frame_indices[-1])
                
                frame_indices = np.array(frame_indices[:self.max_vid_length], dtype=int)
                frames = vr.get_batch(frame_indices).asnumpy()
                video_snippet_style = list(frames)
        except Exception as e:
            raise ValueError(f"Failed to load video {video_path_style}: {e}")

        # ==============================
        # 5. Pad to max_motion_length (ONLY FOR RAW MOTION)
        # ==============================
        if m_length_style < self.max_motion_length:
            padding_len = self.max_motion_length - len(motion_style)
            padding = np.zeros((padding_len, motion_style.shape[1]))
            motion_style = np.concatenate([motion_style, padding], axis=0)
        
        # Note: latent_humanml padding is intentionally removed.
        # It remains [512, Variable_T]. The mld_collate_paired function handles padding safely.
                
        # ==============================
        # 6. Check for NaNs
        # ==============================
        if np.any(np.isnan(motion_style)) or np.any(np.isnan(latent_humanml)):
            print(f"⚠️ Warning: NaN found in sample '{style_name_style}' or '{name_humanml}'. Retrying with new random index...")
            new_item = random.randint(0, len(self) - 1)
            return self.__getitem__(new_item)

        # ==============================
        # 7. Return Combined Data 
        # ==============================
        return {
            'text_styled': caption_style,
            'motion_styled': motion_style,
            'length_styled': m_length_style,
            'style_name': style_name_style,
            'video_styled': video_snippet_style,
            
            'text_humanml': caption_humanml,
            'motion_humanml': motion_humanml,
            'latent_humanml': latent_humanml, 
            'length_humanml': m_length_humanml
        }


class Text2MotionDatasetCombined_v5(data.Dataset):
    def __init__(
        self,
        # 100STYLES Params
        style_mean, style_std, style_split_file, style_motion_dir, style_text_dir, style_video_dir, style_dict_file,
        # HumanML3D Params
        humanml_mean, humanml_std, humanml_split_file, humanml_latent_dir, humanml_text_dir, humanml_dict_file,  # CHANGED
        # Common Params
        dim_pose, unit_length, max_motion_length, min_motion_length=24, max_text_len=20, max_vid_length=32,
        styles=['Aeroplane','ArmsFolded','Chicken','Robot','Superman'], tiny=True, debug=False, progress_bar=True, evaluation=False,
        yield_mode='both', 
        **kwargs):

        self.yield_mode = yield_mode
        
        self.style_mean = style_mean[:dim_pose]
        self.style_std = style_std[:dim_pose]
        self.styles = styles
        
        self.humanml_mean = humanml_mean[:dim_pose]
        self.humanml_std = humanml_std[:dim_pose]
        
        self.dim_pose = dim_pose
        self.humanml_latent_dir = humanml_latent_dir # CHANGED

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.max_vid_length = max_vid_length
        self.evaluation = evaluation

        self.data_dict_style = {}
        self.id_list_style = []
        
        self.data_dict_humanml = {}
        self.id_list_humanml = []

        if os.path.exists(style_split_file):
            with cs.open(style_split_file, "r") as f:
                for line in f.readlines():
                    self.id_list_style.append(line.strip())
        else:
            raise FileNotFoundError(f"100STYLES split file not found: {style_split_file}")

        if os.path.exists(humanml_split_file):
            with cs.open(humanml_split_file, "r") as f:
                for line in f.readlines():
                    self.id_list_humanml.append(line.strip())
        else:
            raise FileNotFoundError(f"HumanML3D split file not found: {humanml_split_file}")

        if tiny or debug:
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        # =========================================================
        # LOAD DATASET 1: 100STYLES
        # =========================================================
        metadata_dict = build_dict_from_txt(style_dict_file)
        style_to_label = {style: i for i, style in enumerate(self.styles)} if self.styles else {}

        new_name_list_style = []
        length_list_style = []
        count = 0
        
        print(f"Loading 100STYLES data from {style_split_file}...")
        iterator_style = tqdm(self.id_list_style) if progress_bar else self.id_list_style
        
        for name in iterator_style:
            if count > maxdata: break
            try:
                _, style_name, motion_type, _, length = metadata_dict[name]
                
                if self.styles and style_name not in self.styles:
                    continue
                if motion_type.startswith("TR"):
                    continue

                video_path = pjoin(style_video_dir, name + '_FV.mp4')
                motion_path = pjoin(style_motion_dir, name + '.npy')
                text_path = pjoin(style_text_dir, name + ".txt")

                if not (os.path.exists(motion_path) and os.path.exists(video_path) and os.path.exists(text_path)):
                    continue

                if length < self.min_motion_length or length >= 400:
                    continue

                text_data_style = []
                seen_captions = set()
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        if not caption: continue
                        
                        canonical_caption = caption.lower()
                        if canonical_caption in seen_captions: continue
                        seen_captions.add(canonical_caption)
                        
                        text_data_style.append({'caption': caption, 'tokens': []})

                if len(text_data_style) > 0:
                    self.data_dict_style[name] = {
                        'motion': motion_path,
                        'length': length,
                        'style': style_to_label.get(style_name, 0),
                        'style_name': style_name,
                        'text': text_data_style,
                        'video': video_path
                    }
                    
                    new_name_list_style.append(name)
                    length_list_style.append(length)
                    count += 1
            except Exception as e:
                pass

        self.name_list_style, self.length_list_style = zip(*sorted(zip(new_name_list_style, length_list_style), key=lambda x: x[1]))
        self.length_arr_style = np.array(self.length_list_style)

        # =========================================================
        # LOAD DATASET 2: HumanML3D (Pre-Encoded Latents)
        # =========================================================
        metadata_dict_humanml = build_dict_from_txt2(humanml_dict_file)
        
        new_name_list_humanml = []
        length_list_humanml = []
        count = 0

        print(f"Loading HumanML3D data from {humanml_split_file}...")
        iterator_humanml = tqdm(self.id_list_humanml) if progress_bar else self.id_list_humanml

        for name in iterator_humanml:
            if count > maxdata: break
            try:
                total_length = metadata_dict_humanml[name]

                if total_length < self.min_motion_length or total_length >= 200:
                    continue
                
                text_data_humanml = []
                flag = False
                
                with cs.open(pjoin(humanml_text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data_humanml.append(text_dict)
                        else:
                            try:
                                start_idx = int(f_tag * 20)
                                end_idx = int(to_tag * 20)
                                seg_len = end_idx - start_idx
                                if seg_len < self.min_motion_length or seg_len >= 200:
                                    continue
                                
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in self.data_dict_humanml:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                
                                self.data_dict_humanml[new_name] = {
                                    'source_name': name,
                                    'start': start_idx,
                                    'end': end_idx,
                                    'length': seg_len,
                                    'text': [text_dict]
                                }
                                new_name_list_humanml.append(new_name)
                                length_list_humanml.append(seg_len)
                            except:
                                pass

                if flag:
                    self.data_dict_humanml[name] = {
                        'source_name': name,
                        'start': 0,
                        'end': total_length,    
                        'length': total_length,
                        'text': text_data_humanml
                    }
                    new_name_list_humanml.append(name)
                    length_list_humanml.append(total_length)
                    count += 1
            except Exception as e:
                pass

        self.name_list_humanml, self.length_list_humanml = zip(*sorted(zip(new_name_list_humanml, length_list_humanml), key=lambda x: x[1]))
        self.length_arr_humanml = np.array(self.length_list_humanml)

        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer_style = np.searchsorted(self.length_arr_style, length)
        self.max_length = length

    def inv_transform(self, data, mean, std):
        return data * std + mean

    def transform(self, data, mean, std):
        return (data - mean) / std

    def __len__(self):
        if self.yield_mode == '100styles':
            return len(self.name_list_style) - self.pointer_style
        elif self.yield_mode == 'humanml3d':
            return len(self.name_list_humanml)
        else:
            return max(len(self.name_list_style) - self.pointer_style, len(self.name_list_humanml))

    def __getitem__(self, item):
        try:
            return self._safe_getitem(item)
        except Exception as e:
            print(f"Critical Error in __getitem__ for index {item}: {e}")
            return self._safe_getitem(random.randint(0, len(self) - 1))

    def _safe_getitem(self, item):
        result = {}

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        # ==============================
        # 1. 100STYLES Processing Block
        # ==============================
        if self.yield_mode in ['100styles', 'both']:
            total_style = len(self.name_list_style) - self.pointer_style
            idx_style = self.pointer_style + (item % total_style) 
            
            name_style = self.name_list_style[idx_style]
            data_style = self.data_dict_style[name_style]
            
            motion_style_path = data_style['motion']
            m_length_style = data_style['length']
            text_list_style = data_style['text']
            style_name_style = data_style['style_name']
            video_path_style = data_style['video']

            caption_style = random.choice(text_list_style)['caption']
            motion_style = np.load(motion_style_path)[:, :self.dim_pose]

            if coin2 == 'double':
                m_length_style = (m_length_style // self.unit_length - 1) * self.unit_length
            elif coin2 == 'single':
                m_length_style = (m_length_style // self.unit_length) * self.unit_length

            m_length_style = min(self.max_motion_length, m_length_style)

            max_start_style = max(0, len(motion_style) - m_length_style)
            start_idx_style = random.randint(0, max_start_style)
            motion_style = motion_style[start_idx_style : start_idx_style + m_length_style]

            motion_style = (motion_style - self.style_mean) / self.style_std

            video_snippet_style = []
            try:
                vr = VideoReader(video_path_style, ctx=cpu(0), width=224, height=224)
                video_length = len(vr)
                if video_length > 0:
                    window_start = min(start_idx_style, max(video_length - 1, 0))
                    window_end = min(start_idx_style + m_length_style - 1, video_length - 1)
                    
                    if window_end <= window_start:
                        window_end = window_start
                    
                    frame_indices = np.linspace(window_start, window_end, num=self.max_vid_length, dtype=int).tolist()
                    while len(frame_indices) < self.max_vid_length:
                        frame_indices.append(frame_indices[-1])
                    
                    frame_indices = np.array(frame_indices[:self.max_vid_length], dtype=int)
                    frames = vr.get_batch(frame_indices).asnumpy()
                    video_snippet_style = list(frames)
            except Exception as e:
                raise ValueError(f"Failed to load video {video_path_style}: {e}")

            if m_length_style < self.max_motion_length:
                padding_len = self.max_motion_length - len(motion_style)
                padding = np.zeros((padding_len, motion_style.shape[1]))
                motion_style = np.concatenate([motion_style, padding], axis=0)

            if np.any(np.isnan(motion_style)):
                print(f"⚠️ Warning: NaN found in '{style_name_style}'. Retrying...")
                return self.__getitem__(random.randint(0, len(self) - 1))

            result.update({
                'text_styled': caption_style,
                'motion_styled': motion_style,
                'length_styled': m_length_style,
                'style_name': style_name_style,
                'video_styled': video_snippet_style,
            })

        # ==============================
        # 2. HumanML3D Processing Block
        # ==============================
        if self.yield_mode in ['humanml3d', 'both']:
            total_humanml = len(self.name_list_humanml)
            idx_humanml = item % total_humanml 
            
            name_humanml = self.name_list_humanml[idx_humanml]
            meta_humanml = self.data_dict_humanml[name_humanml]

            source_name = meta_humanml['source_name']
            start = meta_humanml['start']
            end = meta_humanml['end']
            m_length_humanml = meta_humanml['length']
            text_list_humanml = meta_humanml['text']

            # CHANGED: Load from latent directory
            latent_path = pjoin(self.humanml_latent_dir, source_name + '.npy')
            full_latent = np.load(latent_path)

            real_len = len(full_latent)
            safe_start = min(start, real_len - 1)
            safe_end = min(end, real_len)
            
            # CHANGED: Full latent dimension preservation
            latent_humanml = full_latent[safe_start:safe_end]

            caption_humanml = random.choice(text_list_humanml)['caption']

            if coin2 == 'double':
                m_length_humanml = (m_length_humanml // self.unit_length - 1) * self.unit_length
            elif coin2 == 'single':
                m_length_humanml = (m_length_humanml // self.unit_length) * self.unit_length

            m_length_humanml = min(self.max_motion_length, m_length_humanml)

            max_start_humanml = max(0, len(latent_humanml) - m_length_humanml)
            start_idx_humanml = random.randint(0, max_start_humanml)
            latent_humanml = latent_humanml[start_idx_humanml : start_idx_humanml + m_length_humanml]

            # CHANGED: No Z-Normalization for latents

            if len(latent_humanml) < self.max_motion_length:
                padding_len = self.max_motion_length - len(latent_humanml)
                padding = np.zeros((padding_len, latent_humanml.shape[1]))
                latent_humanml = np.concatenate([latent_humanml, padding], axis=0)

            if np.any(np.isnan(latent_humanml)):
                print(f"⚠️ Warning: NaN found in '{name_humanml}'. Retrying...")
                return self.__getitem__(random.randint(0, len(self) - 1))

            result.update({
                'text_humanml': caption_humanml,
                'latent_humanml': latent_humanml, # CHANGED KEY
                'length_humanml': m_length_humanml
            })

        return result


class StyleMotionDataset(data.Dataset):

    def __init__(self, stage: str, data_root: str, motion_dir: str, text_dir: str, dict_file: str, split_file: str, styles: Optional[List[str]], use_augmentation: Optional[bool], dim_pose: int = 67, max_motion_length: int = 196, min_motion_length: int = 40, unit_length = 4):

        self.stage = stage
        self.data_root = data_root
        self.dim_pose = dim_pose
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.styles = styles
        self.dim_pose = dim_pose
        self.use_augmentation = use_augmentation if use_augmentation is not None else (stage == 'train')
        self.pointer = 0


        
        # Load normalization stats (sliced to dim_pose)
        mean_path = pjoin(data_root, 'Mean.npy')
        std_path = pjoin(data_root, 'Std.npy')
        
        
        self.mean = np.load(mean_path)[:dim_pose]
        self.std = np.load(std_path)[:dim_pose]
        
        # Load metadata dict
        metadata_dict = build_dict_from_txt(dict_file)
        style_to_label = {style: i for i, style in enumerate(self.styles)} if self.styles else {}

        # Load split file
        id_list = []
        if os.path.exists(split_file):
            with cs.open(split_file, 'r', encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        id_list.append(name)
        else:
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Build data dict with filtering
        self.data_dict = {}
        name_list = []
        length_list = []
        labels = []
        
        for name in tqdm(id_list, desc=f"Loading {stage} data"):
            # Skip if no metadata
            if name not in metadata_dict:
                continue
            
            motion_path = pjoin(motion_dir, name + '.npy')
            label_num, style_name, motion_type, _, length = metadata_dict[name]
            
            # Style filtering
            if self.styles is not None and style_name not in self.styles:
                continue
            
            # # Transition filtering
            # if motion_type.startswith("TR"):
            #     continue
            
            if length < self.min_motion_length or length >= self.max_motion_length:
                    continue
        
            if not os.path.exists(motion_path):
                continue
            
            # Load text captions
            text_data = []
            seen_captions = set()
            text_path = pjoin(text_dir, f"{name}.txt")
            if os.path.exists(text_path):
                seen_captions = set()
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        if not caption: continue
                        
                        canonical_caption = caption.lower()
                        if canonical_caption in seen_captions: continue
                        seen_captions.add(canonical_caption)
                        
                        text_data.append({'caption': caption})

            if not text_data:
                text_data = [{"caption": ""}]
            
            # Store
            self.data_dict[name] = {
                'motion': motion_path,
                'length': length,
                'label': style_to_label.get(style_name, 0),
                'style_name': style_name,
                'text': text_data,
            }
            
            name_list.append(name)
            length_list.append(length)
            labels.append(style_to_label.get(style_name, 0))
        
        # Sort by length (for curriculum learning support)
        if name_list:
            sorted_data = sorted(zip(name_list, length_list, labels), key=lambda x: x[1])
            name_list, length_list, labels = zip(*sorted_data)
            self.name_list = list(name_list)
            self.length_arr = np.array(length_list)
        else:
            self.name_list = []
            self.length_arr = np.array([])
        
        # Store class info
        self.num_classes = len(style_to_label) if self.styles else len(set(labels))
        self.style_names = list(self.styles) if self.styles else []
        
        print(f"[{stage}] Loaded {len(self.name_list)} samples, "
              f"{self.num_classes} classes, feature dim: {self.dim_pose}")
    
    def __len__(self) -> int:
        return len(self.name_list) - self.pointer
    
    def __getitem__(self, item: int) -> dict:
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        
       
        m_length = data["length"]
        label = data["label"]
        style_name = data["style_name"]
        text_list = data["text"]
        motion_path = data["motion"]
        motion = np.load(motion_path)[:, :self.dim_pose]
        
        # Length adjustment (frame alignment)
        if self.unit_length < 10:
            coin = np.random.choice(["single", "single", "double"])
        else:
            coin = "single"
        
        if coin == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        
        m_length = min(self.max_motion_length, m_length)
        m_length = max(self.unit_length, m_length)  # Ensure minimum length
        
        # Random crop
        if len(motion) > m_length:
            start_idx = random.randint(0, len(motion) - m_length)
            motion = motion[start_idx:start_idx + m_length]
        else:
            motion = motion[:m_length]
        
        # Normalize
        motion = (motion - self.mean) / (self.std)
        
        # Augmentation (training only)
        if self.use_augmentation:
            motion = random_zero_out(motion)
        
        # Pad to max length
        if len(motion) < self.max_motion_length:
            padding = np.zeros((self.max_motion_length - len(motion), motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        # Random text selection
        text_data = random.choice(text_list)
        caption = text_data["caption"]
        
        
        return {
            "motion": torch.from_numpy(motion).float(),
            "label": torch.tensor(label).long(),
            "length": m_length,
            "text": caption,
            "style_name": style_name,
            "name": name,
        }

    def reset_max_len(self, length: int):
        """
        Filter to only use motions >= length (curriculum learning).
        Useful for progressive training on longer sequences.
        """
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print(f"Pointer set to {self.pointer} (filtering motions < {length} frames)")
    
    def inv_transform(self, data: np.ndarray) -> np.ndarray:
        """Denormalize motion data back to original scale."""
        return data * self.std + self.mean
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize raw motion data."""
        return (data - self.mean) / (self.std + 1e-8)
    
    def get_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get normalization statistics."""
        return self.mean, self.std
    
    def get_style_names(self) -> List[str]:
        """Get list of style names (ordered by dense label index)."""
        return self.style_names
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_classes
    
    def dense_to_original_label(self, dense_label: int) -> int:
        """Convert dense label back to original label."""
        return self.idx_to_label[dense_label]
    
    def original_to_dense_label(self, original_label: int) -> int:
        """Convert original label to dense label."""
        return self.label_mapping[original_label]


# class Text2MotionDatasetCombined_v2(data.Dataset):
#     """
#     Combined dataset for Option 7 (Hybrid) training.
    
#     Provides:
#     1. 100STYLES: Paired (neutral_motion, styled_motion, styled_video) for cycle consistency
#     2. HumanML3D: (text, motion) for text-to-content learning with zero style
    
#     Key Features:
#     - Exact matching between neutral and styled motions by motion_key
#     - Same crop window applied to both neutral and styled
#     - Video aligned with motion crop
#     - HumanML3D sampled randomly (not paired with 100STYLES)
#     """
    
#     def __init__(
#         self,
#         # 100STYLES Params
#         style_mean, style_std, style_split_file, style_motion_dir, style_text_dir, 
#         style_video_dir, style_dict_file,
#         # HumanML3D Params
#         humanml_mean, humanml_std, humanml_split_file, humanml_motion_dir, humanml_text_dir,
#         # Common Params
#         dim_pose, unit_length, max_motion_length, 
#         min_motion_length=40, max_text_len=20, max_vid_length=32,
#         styles=['Aeroplane', 'ArmsFolded', 'Chicken', 'Robot', 'Superman'],
#         neutral_style='Neutral',
#         tiny=False, debug=False, progress_bar=True, evaluation=False,
#         **kwargs
#     ):
        
#         # =====================================================================
#         # Store parameters
#         # =====================================================================
#         self.style_mean = style_mean[:dim_pose]
#         self.style_std = style_std[:dim_pose]
#         self.humanml_mean = humanml_mean[:dim_pose]
#         self.humanml_std = humanml_std[:dim_pose]
        
#         self.styles = styles
#         self.neutral_style = neutral_style
#         self.dim_pose = dim_pose
#         self.unit_length = unit_length
#         self.max_motion_length = max_motion_length
#         self.min_motion_length = min_motion_length
#         self.max_text_len = max_text_len
#         self.max_vid_length = max_vid_length
#         self.evaluation = evaluation
        
#         self.max_length = 20
#         self.pointer = 0
        
#         # =====================================================================
#         # Load split files
#         # =====================================================================
#         id_list_style = []
#         if os.path.exists(style_split_file):
#             with cs.open(style_split_file, "r") as f:
#                 for line in f.readlines():
#                     id_list_style.append(line.strip())
#         else:
#             raise FileNotFoundError(f"100STYLES split file not found: {style_split_file}")
        
#         id_list_humanml = []
#         if os.path.exists(humanml_split_file):
#             with cs.open(humanml_split_file, "r") as f:
#                 for line in f.readlines():
#                     id_list_humanml.append(line.strip())
#         else:
#             raise FileNotFoundError(f"HumanML3D split file not found: {humanml_split_file}")
        
#         # Debug limits
#         if tiny:
#             maxdata = 10
#         elif debug:
#             maxdata = 100
#         else:
#             maxdata = float('inf')
        
#         # =====================================================================
#         # STEP 1: Build NEUTRAL motion dictionary (indexed by motion_key)
#         # =====================================================================
#         metadata_dict = build_dict_from_txt_mirror(style_dict_file)
#         style_to_label = {style: i for i, style in enumerate(self.styles)}
        
#         neutral_by_key = {}  # {"BR_00": {...}, "M_BR_00": {...}}
        
#         print(f"Loading NEUTRAL style motions...")
#         iterator = tqdm(id_list_style) if progress_bar else id_list_style
        
#         for name in iterator:
#             try:
#                 _, style_name, motion_type, motion_idx, _, is_mirrored = metadata_dict[name]
                
#                 # Only load neutral motions
#                 if style_name != self.neutral_style:
#                     continue
#                 if motion_type.startswith("TR"):  # Skip transitions
#                     continue
                
#                 motion_path = pjoin(style_motion_dir, name + '.npy')
#                 text_path = pjoin(style_text_dir, name + '.txt')
                
#                 if not os.path.exists(motion_path) or not os.path.exists(text_path):
#                     continue
                
#                 motion = np.load(motion_path)
                
#                 if len(motion) < self.min_motion_length or len(motion) >= 400:
#                     continue
                
#                 # Load text
#                 text_data = []
#                 seen_captions = set()
#                 with cs.open(text_path) as f:
#                     for line in f.readlines():
#                         line_split = line.strip().split("#")
#                         caption = line_split[0]
#                         caption = normalize_caption(caption)
#                         if not caption:
#                             continue
#                         canonical = caption.lower()
#                         if canonical in seen_captions:
#                             continue
#                         seen_captions.add(canonical)
#                         text_data.append(caption)
                
#                 if not text_data:
#                     continue
                
#                 # Create motion_key: "BR_00" or "M_BR_00"
#                 motion_key = f"{motion_type}_{motion_idx}"
#                 if is_mirrored:
#                     motion_key = f"M_{motion_key}"
                
#                 neutral_by_key[motion_key] = {
#                     'name': name,
#                     'motion': motion,
#                     'length': len(motion),
#                     'text': text_data,
#                 }
                
#             except Exception as e:
#                 pass
        
#         print(f"Loaded {len(neutral_by_key)} neutral motion keys")
        
#         # =====================================================================
#         # STEP 2: Build PAIRED dataset (styled + matched neutral)
#         # =====================================================================
#         self.data_list_paired = []
#         skipped_no_match = 0
#         skipped_no_video = 0
#         count = 0
        
#         print(f"\nCreating paired (neutral, styled, video) samples...")
#         iterator = tqdm(id_list_style) if progress_bar else id_list_style
        
#         for name in iterator:
#             if count >= maxdata:
#                 break
            
#             try:
#                 _, style_name, motion_type, motion_idx, _, is_mirrored = metadata_dict[name]
                
#                 # Skip neutral (we use it as target, not source)
#                 if style_name == self.neutral_style:
#                     continue
#                 if style_name not in self.styles:
#                     continue
#                 if motion_type.startswith("TR"):
#                     continue
                
#                 # Create motion_key to find matching neutral
#                 motion_key = f"{motion_type}_{motion_idx}"
#                 if is_mirrored:
#                     motion_key = f"M_{motion_key}"
                
#                 # Must have matching neutral
#                 if motion_key not in neutral_by_key:
#                     skipped_no_match += 1
#                     continue
                
#                 # Check paths
#                 video_path = pjoin(style_video_dir, name + '_FV.mp4')
#                 styled_motion_path = pjoin(style_motion_dir, name + '.npy')
                
#                 if not os.path.exists(video_path):
#                     skipped_no_video += 1
#                     continue
#                 if not os.path.exists(styled_motion_path):
#                     continue
                
#                 # Load styled motion
#                 styled_motion = np.load(styled_motion_path)
#                 if len(styled_motion) < self.min_motion_length or len(styled_motion) >= 400:
#                     continue

#                 # Load styled text (captions for styled sequence)
#                 styled_text_path = pjoin(style_text_dir, name + '.txt')
#                 styled_text_list = []
#                 if os.path.exists(styled_text_path):
#                     seen_styled = set()
#                     with cs.open(styled_text_path) as f:
#                         for line in f.readlines():
#                             caption = line.strip().split("#")[0]
#                             caption = normalize_caption(caption)
#                             if not caption:
#                                 continue
#                             canonical = caption.lower()
#                             if canonical in seen_styled:
#                                 continue
#                             seen_styled.add(canonical)
#                             styled_text_list.append(caption)
#                 # If styled captions missing, fall back to neutral captions
#                 if not styled_text_list:
#                     styled_text_list = neutral_sample['text']
                
#                 # Get matching neutral
#                 neutral_sample = neutral_by_key[motion_key]
                
#                 # Create paired sample
#                 paired_sample = {
#                     # Neutral motion & text (content)
#                     'neutral_motion': neutral_sample['motion'].copy(),
#                     'neutral_length': neutral_sample['length'],
#                     'text_list': neutral_sample['text'],
                    
                    
#                     # Styled motion (target for style addition)
#                     'styled_motion': styled_motion.copy(),
#                     'styled_length': len(styled_motion),
#                     'styled_text_list': styled_text_list,
#                     'styled_name': name,
                    
#                     # Video (style source)
#                     'video_path': video_path,
#                     'style_label': style_to_label.get(style_name, 0),
#                     'style_name': style_name,
                    
#                     # Matching key
#                     'motion_key': motion_key,
#                 }
                
#                 self.data_list_paired.append(paired_sample)
#                 count += 1
                
#             except Exception as e:
#                 pass
        
#         print(f"Created {len(self.data_list_paired)} paired samples")
#         print(f"Skipped {skipped_no_match} (no matching neutral)")
#         print(f"Skipped {skipped_no_video} (no video)")
        
#         # Print distribution
#         style_counts = {}
#         for sample in self.data_list_paired:
#             s = sample['style_name']
#             style_counts[s] = style_counts.get(s, 0) + 1
#         print(f"\nPaired samples by style:")
#         for style, cnt in sorted(style_counts.items()):
#             print(f"  {style}: {cnt}")
        
#         # =====================================================================
#         # STEP 3: Load HumanML3D dataset
#         # =====================================================================
#         self.data_dict_humanml = {}
#         new_name_list_humanml = []
#         length_list_humanml = []
#         count = 0
        
#         print(f"\nLoading HumanML3D data...")
#         iterator = tqdm(id_list_humanml) if progress_bar else id_list_humanml
        
#         for name in iterator:
#             if count >= maxdata:
#                 break
            
#             try:
#                 motion = np.load(pjoin(humanml_motion_dir, name + '.npy'))[:, :dim_pose]
                
#                 if len(motion) < self.min_motion_length or len(motion) >= 200:
#                     continue
                
#                 text_data = []
#                 flag = False
                
#                 with cs.open(pjoin(humanml_text_dir, name + '.txt')) as f:
#                     for line in f.readlines():
#                         text_dict = {}
#                         line_split = line.strip().split('#')
#                         caption = line_split[0]
#                         tokens = line_split[1].split(' ') if len(line_split) > 1 else []
#                         f_tag = float(line_split[2]) if len(line_split) > 2 else 0.0
#                         to_tag = float(line_split[3]) if len(line_split) > 3 else 0.0
#                         f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                         to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
#                         text_dict['caption'] = caption
#                         text_dict['tokens'] = tokens
                        
#                         if f_tag == 0.0 and to_tag == 0.0:
#                             flag = True
#                             text_data.append(text_dict)
#                         else:
#                             # Handle sub-segments
#                             try:
#                                 n_motion = motion[int(f_tag * 20):int(to_tag * 20)]
#                                 if len(n_motion) < self.min_motion_length or len(n_motion) >= 200:
#                                     continue
                                
#                                 new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
#                                 while new_name in self.data_dict_humanml:
#                                     new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                
#                                 self.data_dict_humanml[new_name] = {
#                                     'motion': n_motion,
#                                     'length': len(n_motion),
#                                     'text': [text_dict]
#                                 }
#                                 new_name_list_humanml.append(new_name)
#                                 length_list_humanml.append(len(n_motion))
#                             except:
#                                 pass
                
#                 if flag:
#                     self.data_dict_humanml[name] = {
#                         'motion': motion,
#                         'length': len(motion),
#                         'text': text_data
#                     }
#                     new_name_list_humanml.append(name)
#                     length_list_humanml.append(len(motion))
#                     count += 1
                    
#             except:
#                 pass
        
#         # Sort HumanML3D by length
#         if len(new_name_list_humanml) > 0:
#             self.name_list_humanml, self.length_list_humanml = zip(
#                 *sorted(zip(new_name_list_humanml, length_list_humanml), key=lambda x: x[1])
#             )
#             self.length_arr_humanml = np.array(self.length_list_humanml)
#         else:
#             self.name_list_humanml = []
#             self.length_arr_humanml = np.array([])
        
#         print(f"Loaded {len(self.data_dict_humanml)} HumanML3D samples")
        
#         # =====================================================================
#         # Sort paired data by styled_length for evaluation
#         # =====================================================================
#         if self.evaluation:
#             self.data_list_paired = sorted(self.data_list_paired, key=lambda x: x['styled_length'])
        
#         self.length_arr_paired = np.array([s['styled_length'] for s in self.data_list_paired])
        
#         if self.evaluation:
#             self.reset_max_len(self.max_length)
        
#         print(f"\n{'='*60}")
#         print(f"Dataset Summary:")
#         print(f"  100STYLES paired: {len(self.data_list_paired)}")
#         print(f"  HumanML3D: {len(self.data_dict_humanml)}")
#         print(f"{'='*60}")
    
#     def reset_max_len(self, length):
#         assert length <= self.max_motion_length
#         self.pointer = np.searchsorted(self.length_arr_paired, length)
#         self.max_length = length
#         print(f"Pointer set to {self.pointer}")
    
#     def __len__(self):
#         return len(self.data_list_paired) - self.pointer
    
    # def __getitem__(self, item):
    #     # =====================================================================
    #     # 1. Get PAIRED 100STYLES sample (indexed)
    #     # =====================================================================
    #     idx = self.pointer + item
    #     sample = self.data_list_paired[idx]
        
    #     neutral_motion = sample['neutral_motion'].copy()
    #     neutral_length = sample['neutral_length']
    #     styled_motion = sample['styled_motion'].copy()
    #     styled_length = sample['styled_length']
    #     text_list = sample['text_list']
    #     styled_text_list = sample.get('styled_text_list', text_list)
    #     style_name = sample['style_name']
    #     video_path = sample['video_path']
        
    #     # Random caption from list
    #     text_normal = random.choice(text_list)
    #     text_styled = random.choice(styled_text_list)
        
    #     # =====================================================================
    #     # 2. Get RANDOM HumanML3D sample
    #     # =====================================================================
    #     idx_humanml = random.randint(0, len(self.name_list_humanml) - 1)
    #     name_humanml = self.name_list_humanml[idx_humanml]
    #     data_humanml = self.data_dict_humanml[name_humanml]
        
    #     motion_humanml = data_humanml['motion'].copy()
    #     length_humanml = data_humanml['length']
    #     text_list_humanml = data_humanml['text']
        
    #     # Random caption
    #     text_data_humanml = random.choice(text_list_humanml)
    #     text_humanml = text_data_humanml['caption']
        
    #     # =====================================================================
    #     # 3. Process PAIRED motions (same crop for both)
    #     # =====================================================================
    #     if self.unit_length < 10:
    #         coin2 = np.random.choice(['single', 'single', 'double'])
    #     else:
    #         coin2 = 'single'
        
    #     # Use minimum length to ensure valid cropping for both
    #     min_length = min(neutral_length, styled_length)
        
    #     if coin2 == 'double':
    #         m_length = (min_length // self.unit_length - 1) * self.unit_length
    #     else:
    #         m_length = (min_length // self.unit_length) * self.unit_length
        
    #     m_length = max(m_length, self.unit_length)
    #     m_length = min(m_length, self.max_motion_length)
        
    #     # Same start_idx for BOTH neutral and styled (keep them aligned!)
    #     start_idx = random.randint(0, max(0, min_length - m_length))
        
    #     neutral_motion = neutral_motion[start_idx:start_idx + m_length]
    #     styled_motion = styled_motion[start_idx:start_idx + m_length]
        
    #     # =====================================================================
    #     # 4. Process HumanML3D motion (independent crop)
    #     # =====================================================================
    #     if coin2 == 'double':
    #         m_length_humanml = (length_humanml // self.unit_length - 1) * self.unit_length
    #     else:
    #         m_length_humanml = (length_humanml // self.unit_length) * self.unit_length
        
    #     m_length_humanml = max(m_length_humanml, self.unit_length)
    #     m_length_humanml = min(m_length_humanml, self.max_motion_length)
        
    #     start_idx_humanml = random.randint(0, max(0, length_humanml - m_length_humanml))
    #     motion_humanml = motion_humanml[start_idx_humanml:start_idx_humanml + m_length_humanml]
        
    #     # =====================================================================
    #     # 5. Load VIDEO (aligned with motion crop)
    #     # =====================================================================
    #     try:
    #         vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
    #         video_length = len(vr)
            
    #         if video_length > 0:
    #             # Align video frames with the motion crop window
    #             video_start = int(start_idx * video_length / styled_length)
    #             video_end = int((start_idx + m_length) * video_length / styled_length)
    #             video_end = min(video_end, video_length - 1)
                
    #             frame_indices = np.linspace(video_start, video_end, num=self.max_vid_length, dtype=int)
    #             frame_indices = np.clip(frame_indices, 0, video_length - 1)
    #             frames = vr.get_batch(frame_indices.tolist()).asnumpy()
    #             video_snippet = list(frames)
    #         else:
    #             video_snippet = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length
                
    #     except Exception as e:
    #         print(f"Warning: Could not load video {video_path}: {e}")
    #         video_snippet = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.max_vid_length
        
    #     # =====================================================================
    #     # 6. Normalize motions
    #     # =====================================================================
    #     # 100STYLES motions (both neutral and styled use same normalization)
    #     neutral_motion = neutral_motion[:, :self.dim_pose]
    #     neutral_motion = (neutral_motion - self.style_mean) / self.style_std
        
    #     styled_motion = styled_motion[:, :self.dim_pose]
    #     styled_motion = (styled_motion - self.style_mean) / self.style_std
        
    #     # HumanML3D motion
    #     motion_humanml = motion_humanml[:, :self.dim_pose]
    #     motion_humanml = (motion_humanml - self.humanml_mean) / self.humanml_std
        
    #     # =====================================================================
    #     # 7. Pad to max_motion_length
    #     # =====================================================================
    #     # Pad 100STYLES motions
    #     if m_length < self.max_motion_length:
    #         padding = np.zeros((self.max_motion_length - m_length, neutral_motion.shape[1]))
    #         neutral_motion = np.concatenate([neutral_motion, padding], axis=0)
    #         styled_motion = np.concatenate([styled_motion, padding.copy()], axis=0)
        
    #     # Pad HumanML3D motion
    #     if m_length_humanml < self.max_motion_length:
    #         padding = np.zeros((self.max_motion_length - m_length_humanml, motion_humanml.shape[1]))
    #         motion_humanml = np.concatenate([motion_humanml, padding], axis=0)
        
    #     # =====================================================================
    #     # 8. Check for NaNs
    #     # =====================================================================
    #     if np.any(np.isnan(neutral_motion)) or np.any(np.isnan(styled_motion)) or np.any(np.isnan(motion_humanml)):
    #         raise ValueError("NaN found in motion data")
        
    #     # =====================================================================
    #     # 9. Return combined data as dictionary (cleaner for Option 7)
    #     # =====================================================================
    #     return {
    #         # 100STYLES Paired Data
    #         'text_normal': text_normal,
    #         'text_satyled': text_styled,
    #         'motion_neutral': neutral_motion,
    #         'motion_styled': styled_motion,
    #         'length_style': m_length,
    #         'style_name': style_name,
    #         'video_styled': video_snippet,
            
    #         # HumanML3D Data
    #         'text_humanml': text_humanml,
    #         'motion_humanml': motion_humanml,
    #         'length_humanml': m_length_humanml,
    #     }