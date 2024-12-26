import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from torch._C import dtype, set_flush_denormal
import dust3r.utils.po_utils.basic
import dust3r.utils.po_utils.improc
from dust3r.utils.po_utils.misc import farthest_point_sample_py
from dust3r.utils.po_utils.geom import apply_4x4_py, apply_pix_T_cam_py
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
from functools import partial
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, is_good_type, transpose_to_landscape
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution
from dust3r.datasets.utils.geom import apply_4x4_py, realative_T_py
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


from pyntcloud import PyntCloud
import pandas as pd

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')




class PointOdysseyDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/pointodyssey',
                 dset='train',
                 use_augs=False,
                 S=2,
                 N=16,
                 strides=[1,2,3,4,5,6,7,8,9],
                 clip_step=2,
                 quick=False,
                 verbose=False,
                 dist_type=None,
                 clip_step_last_skip = 0,
                 motion_thresh = 1e-6,
                 *args, 
                 **kwargs
                 ):

        print('loading pointodyssey dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'pointodyssey'
        self.split = dset
        self.S = S # stride
        self.N = N # min num points
        self.verbose = verbose
        self.motion_thresh = motion_thresh
        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_2d_paths = []
        self.traj_3d_paths = []
        self.extrinsic_paths = []
        self.intrinsic_paths = []
        self.masks_paths = []
        self.valids_paths = []
        self.visibs_paths = []
        self.annotation_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-1]
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        
        if quick:
           self.sequences = self.sequences[1:2] 
        
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')


        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            rgb_path = os.path.join(seq, 'rgbs')
            info_path = os.path.join(seq, 'info.npz')
            annotations_path = os.path.join(seq, 'anno.npz')
            
            if os.path.isfile(info_path) and os.path.isfile(annotations_path):

                traj_3d_files = glob.glob(os.path.join(seq, 'trajs_3d', '*.npy'))
                if len(traj_3d_files):
                    traj_3d_files_0 = np.load(traj_3d_files[0], allow_pickle=True)
                    trajs_3d_shape = traj_3d_files_0.shape[0]
                else:
                    trajs_3d_shape = 0

                if len(traj_3d_files) and trajs_3d_shape > self.N:
                
                    for stride in strides:
                        for ii in range(0,len(os.listdir(rgb_path))-self.S*max(stride,clip_step_last_skip)+1, clip_step):
                            full_idx = ii + np.arange(self.S)*stride
                            self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx) for idx in full_idx])
                            self.depth_paths.append([os.path.join(seq, 'depths', 'depth_%05d.png' % idx) for idx in full_idx])
                            self.normal_paths.append([os.path.join(seq, 'normals', 'normal_%05d.jpg' % idx) for idx in full_idx])
                            # self.traj_2d_paths.append([os.path.join(seq, 'trajs_2d', 'traj_2d_%05d.npy' % idx) for idx in full_idx])
                            self.traj_3d_paths.append([os.path.join(seq, 'trajs_3d', 'traj_3d_%05d.npy' % idx) for idx in full_idx])
                            self.extrinsic_paths.append([os.path.join(seq, 'extrinsics', 'extrinsic_%05d.npy' % idx) for idx in full_idx])
                            self.intrinsic_paths.append([os.path.join(seq, 'intrinsics', 'intrinsic_%05d.npy' % idx) for idx in full_idx])
                            self.masks_paths.append([os.path.join(seq, 'masks', 'mask_%05d.png' % idx) for idx in full_idx])
                            self.valids_paths.append([os.path.join(seq, 'valids', 'valid_%05d.npy' % idx) for idx in full_idx])
                            self.visibs_paths.append([os.path.join(seq, 'visibs', 'visib_%05d.npy' % idx) for idx in full_idx])

                            self.full_idxs.append(full_idx)
                            self.sample_stride.append(stride)
                        if self.verbose:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                elif self.verbose:
                    print('rejecting seq for missing 3d')
            elif self.verbose:
                print('rejecting seq for missing info or anno')
        
        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print('stride counts:', self.stride_counts)
        
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

    def _resample_clips(self, strides, dist_type):

        # Get distribution of strides, and sample based on that
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [min(self.stride_counts[stride], int(dist[i]*max_num_clips)) for i, stride in enumerate(strides)]
        print('resampled_num_clips_each_stride:', num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(self.stride_idxs[stride], num_clips_each_stride[i], replace=False).tolist()
        
        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.normal_paths = [self.normal_paths[i] for i in resampled_idxs]
        # self.traj_2d_paths = [self.traj_2d_paths[i] for i in resampled_idxs]
        self.traj_3d_paths = [self.traj_3d_paths[i] for i in resampled_idxs]
        self.extrinsic_paths = [self.extrinsic_paths[i] for i in resampled_idxs]
        self.intrinsic_paths = [self.intrinsic_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]
        self.masks_paths = [self.masks_paths[i] for i in resampled_idxs]
        self.valids_paths = [self.valids_paths[i] for i in resampled_idxs]
        self.visibs_paths = [self.visibs_paths[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)
    
    def _get_views(self, index, resolution, rng):

        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        # normal_paths = self.normal_paths[index]
        traj_3d_paths = self.traj_3d_paths[index]
        extrinsic_paths = self.extrinsic_paths[index]
        intrinsic_paths = self.intrinsic_paths[index]
        masks_paths = self.masks_paths[index]
        valids_paths = self.valids_paths[index]
        visibs_paths = self.visibs_paths[index]

        # full_idx = self.full_idxs[index]




        traj_3d = [np.load(traj_3d_path, allow_pickle=True) for traj_3d_path in traj_3d_paths]
        pix_T_cams = [np.load(intrinsic_path, allow_pickle=True) for intrinsic_path in intrinsic_paths]
        cams_T_world = [np.load(extrinsic_path, allow_pickle=True) for extrinsic_path in extrinsic_paths]

        # motion_vector = traj_3d[0] - traj_3d[1]
        # motion_vector_norm = np.linalg.norm(motion_vector, axis=-1)
        # motion_mask_3d = motion_vector_norm > self.motion_thresh

        motion_mask_3d = (traj_3d[0]==traj_3d[1]).sum(axis=1)!=3
        # # Project motion_mask_3d to camera space
        # traj_3d_cam_space = apply_4x4_py(cams_T_world[0], traj_3d[0])
        # motion_mask_3d_cam_space = apply_pix_T_cam_py(pix_T_cams[0], traj_3d_cam_space)
        # rgb_image = imread_cv2(rgb_paths[0])
        # rgb_image2 = imread_cv2(rgb_paths[1])
        # height, width, _ = rgb_image.shape

        # # Filter motion_mask_3d_cam_space to be within image boundaries
        # motion_mask_3d_cam_space = np.round(motion_mask_3d_cam_space).astype(int)
        # x, y = motion_mask_3d_cam_space[:, 0], motion_mask_3d_cam_space[:, 1]
        # valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height) & valid_mask & visib_mask
        # motion_mask_3d_cam_space = motion_mask_3d_cam_space[valid_mask]
        
        # motion_mask = np.zeros_like(rgb_image, dtype=np.float32)
        # motion_mask[motion_mask_3d_cam_space[:, 1], motion_mask_3d_cam_space[:, 0]] = [255, 255, 255]
        # # Save the RGB image and motion mask
        # rgb_image_path = os.path.join('tmp', '%05d_rgb.jpg' % index)
        # motion_mask_path = os.path.join('tmp', '%05d_motion_mask.png' % index)
        # sem_path = os.path.join('tmp', '%05d_sem.png' % index)
        # rgb_image2_path = os.path.join('tmp', '%05d_rgb2.jpg' % index)
        # cv2.imwrite(rgb_image_path, rgb_image)
        # cv2.imwrite(motion_mask_path, motion_mask)
        # cv2.imwrite(sem_path, sem_mask)
        # cv2.imwrite(rgb_image2_path, rgb_image2)
        # print(rgb_image_path)
        # print(motion_mask_path)
        # print(sem_path)
        # print(rgb_image2_path)

        # # Create a DataFrame for the point cloud
        # points = traj_3d[0]
        # colors = np.zeros_like(points)
        # colors[motion_mask_3d] = [255, 0, 0]  # Red for motion points
        # colors[~motion_mask_3d] = [0, 0, 0]  # Green for static points

        # point_cloud_df = pd.DataFrame(
        #     np.hstack((points, colors)),
        #     columns=["x", "y", "z", "red", "green", "blue"]
        # )

        # # Create a PyntCloud object
        # point_cloud = PyntCloud(point_cloud_df)
        # point_cloud.plot()

        # try ten samples to see if the motion mask is correct.
        
        views = []
        for i in range(2):
            
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            # masks_path = masks_paths[i]
            # valids_path = valids_paths[i]
            # visibs_path = visibs_paths[i]

            # load camera params
            extrinsics = cams_T_world[i]
            R = extrinsics[:3,:3]
            t = extrinsics[:3,3]
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R.T
            camera_pose[:3,3] = -R.T @ t
            intrinsics = pix_T_cams[i]

            # load image and depth
            rgb_image = imread_cv2(impath)
            # masks_image = imread_cv2(masks_path)


            depth16 = cv2.imread(depthpath, cv2.IMREAD_ANYDEPTH)
            depthmap = depth16.astype(np.float32) / 65535.0 * 1000.0 # 1000 is the max depth in the dataset

            # masks_image, _, _ = self._crop_resize_if_necessary(
                # masks_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)


            views.append(dict(
                img=rgb_image,
                # mask=masks_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))
        return views, motion_mask_3d, traj_3d
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views, motion_mask_3d, traj_3d = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        # img = []
        # mask = []
        # mmask_save = []
        
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # img.append(np.array(view['img']))
            # mask.append(np.array(view['mask']))
            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])
            # view['mask'] = self.transform(view['mask'])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            view['z_far'] = self.z_far
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            pts3d = view['pts3d'].copy()
            pts3d[~view['valid_mask']]=0
            pts3d = pts3d.reshape(-1, pts3d.shape[-1])
            
            try:
                mmask = griddata(traj_3d[v], motion_mask_3d, pts3d, method='nearest', fill_value=0).astype(np.float32)
                mmask = np.clip(mmask, 0, 1)
            except Exception as e:
                print(f"Failed to compute mmask for view {v} at index {idx}: {e}")
                mmask = np.zeros((pts3d.shape[0],), dtype=np.float32)


            view['dynamic_mask'] = mmask.reshape(valid_mask.shape)
            

            # mmask_save.append((mmask.reshape(valid_mask.shape) * 255).astype(np.uint8))

            # visualize masks
            

            # # visualization
            # colors = np.zeros((pts3d.shape[0], 3))
            # colors[:, 0] = 255 * mmask  # Green channel weighted by mmask

            
            # point_cloud_df = pd.DataFrame(
            #     np.hstack((pts3d, colors)),
            #     columns=["x", "y", "z", "red", "green", "blue"]
            # )

            # point_cloud = PyntCloud(point_cloud_df)
            # point_cloud.to_file(f"./tmp/po/point_cloud_{idx}.ply")
            # psudo

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
                # if val.dtype in (torch.bool, np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):   
                    # print(f"{key}={val.shape} for view {view['label']}")
            K = view['camera_intrinsics']

        # Concatenate images, masks, and motion masks into one image and save to tmp/
        # concatenated_images = []
        # for i in range(len(img)):
        #     concatenated_image = np.concatenate((img[i], mask[i], mmask_save[i][...,None]*[255,255,255]), axis=0)
        #     concatenated_images.append(concatenated_image)
        
        # concatenated_images = np.concatenate(concatenated_images, axis=1)
        # concatenated_image_path = os.path.join('tmp', f'{idx}_concatenated.jpg')
        # cv2.imwrite(concatenated_image_path, concatenated_images)

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views
        

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import gradio as gr
    import random


    dataset_location = 'data/point_odyssey'  # Change this to the correct path
    dset = 'train'
    use_augs = False
    S = 2
    N = 1
    strides = [1,2,3,4,5,6,7,8,9]
    clip_step = 2
    quick = False  # Set to True for quick testing

    def visualize_scene(idx):
        views = dataset[idx]
        assert len(views) == 2
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                        focal=views[view_idx]['camera_intrinsics'][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        os.makedirs('./tmp/po', exist_ok=True)
        path = f"./tmp/po/po_scene_{idx}.glb"
        return viz.save_glb(path)

    dataset = PointOdysseyDUSt3R(
        dataset_location=dataset_location,
        dset=dset,
        use_augs=use_augs,
        S=S,
        N=N,
        strides=strides,
        clip_step=clip_step,
        quick=quick,
        verbose=False,
        resolution=224, 
        aug_crop=16,
        dist_type='linear_9_1',
        aug_focal=1.5,
        z_far=80)
# around 514k samples

    idxs = np.arange(0, len(dataset)-1, (len(dataset)-1)//10)
    # idx = random.randint(0, len(dataset)-1)
    # idx = 0
    for idx in idxs:
        print(f"Visualizing scene {idx}...")
        visualize_scene(idx)
