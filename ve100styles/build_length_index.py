"""Append each sequence's frame count to the 100STYLE name dictionary.

    030001 Aeroplane_BR_00.bvh 0   ->   030001 Aeroplane_BR_00.bvh 0 263

Writes <data_root>/100STYLE_name_dict_length.txt, which the VISTA data loaders read.
Ids without a motion file are dropped (and reported).

    python ve100styles/build_length_index.py --data_root datasets/100STYLE-SMPL
"""
import argparse
import os

import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', default=os.path.join('datasets', '100STYLE-SMPL'))
    ap.add_argument('--input', default=None, help='default: <data_root>/100STYLE_name_dict.txt')
    ap.add_argument('--output', default=None, help='default: <data_root>/100STYLE_name_dict_length.txt')
    ap.add_argument('--motion_subdir', default='new_joint_vecs')
    args = ap.parse_args()

    src = args.input or os.path.join(args.data_root, '100STYLE_name_dict.txt')
    dst = args.output or os.path.join(args.data_root, '100STYLE_name_dict_length.txt')
    motion_dir = os.path.join(args.data_root, args.motion_subdir)

    out, missing = [], []
    with open(src) as f:
        lines = [l for l in f if l.strip()]
    for line in tqdm(lines, desc='lengths'):
        motion_id = line.split()[0]
        path = os.path.join(motion_dir, motion_id + '.npy')
        if not os.path.exists(path):
            missing.append(motion_id)
            continue
        length = np.load(path, mmap_mode='r').shape[0]  # header only
        out.append(f"{line.rstrip()} {length}\n")

    with open(dst, 'w') as f:
        f.writelines(out)
    print(f"wrote {len(out)} entries to {dst}; {len(missing)} ids without motion file"
          + (f" (e.g. {missing[:5]})" if missing else ""))


if __name__ == '__main__':
    main()
