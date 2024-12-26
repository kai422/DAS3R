import sys
import torch
sys.path.append('.')
import os
import numpy as np
import glob
from tqdm import tqdm

dataset_location = '../data/point_odyssey'
# print(dataset_location)
for dset in ["train", "test", "sample"]:
    sequences = []
    subdir = os.path.join(dataset_location, dset)
    for seq in glob.glob(os.path.join(subdir, "*/")):
        sequences.append(seq)
    # sequences = sorted(sequences)
    squences = sorted(sequences)

    print('found %d unique videos in %s (dset=%s)' % (len(sequences), dataset_location, dset))

    ## load trajectories
    print('loading trajectories...')

    for seq in sequences:
        # print('seq', seq)
        # if os.path.exists(os.path.join(seq, 'trajs_2d')):
        #     print('skipping', seq)
        #     continue
        info_path = os.path.join(seq, 'info.npz')
        info = np.load(info_path, allow_pickle=True)
        trajs_3d_shape = info['trajs_3d'].astype(np.float32)

        if len(trajs_3d_shape):
            print('processing', seq)        
            rgb_path = os.path.join(seq, 'rgbs')
            info_path = os.path.join(seq, 'info.npz')
            annotations_path = os.path.join(seq, 'anno.npz')

            trajs_3d_path = os.path.join(seq, 'trajs_3d')
            trajs_2d_path = os.path.join(seq, 'trajs_2d')
            os.makedirs(trajs_3d_path, exist_ok=True)
            os.makedirs(trajs_2d_path, exist_ok=True)


            info = np.load(info_path, allow_pickle=True)
            trajs_3d_shape = info['trajs_3d']
            anno = np.load(annotations_path, allow_pickle=True)
            keys = {'trajs_2d': 'traj_2d', 'trajs_3d': 'traj_3d', 'valids': 'valid', 'visibs': 'visib', 'intrinsics': 'intrinsic', 'extrinsics': 'extrinsic'}
            if len(trajs_3d_shape) == 0:
                print(anno['trajs_3d'])
                print('skipping', seq)
                continue
            tensors = {key: torch.tensor(anno[key]).cuda() for key in keys}
            
            for t in tqdm(range(trajs_3d_shape[0])):
                for key, item_key in keys.items():
                    path = os.path.join(seq, key)
                    os.makedirs(path, exist_ok=True)
                    filename = os.path.join(path, f'{item_key}_{t:05d}.npy')
                    np.save(filename, tensors[key][t].cpu().numpy())




