import os
import cv2
import PIL.Image as Image
import numpy as np
from vo_eval import file_interface
from pathlib import Path
from plyfile import PlyData, PlyElement
import torch

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def main(dataset_path, output_path):

    output_colmap_path=os.path.join(output_path, 'sparse/0')
    output_images_path=os.path.join(output_path, 'images')

    output_dynamic_masks_path=os.path.join(output_path, 'dynamic_masks')
    output_enlarged_dynamic_masks_path=os.path.join(output_path, 'enlarged_dynamic_masks')
    output_depth_maps_path=os.path.join(output_path, 'depth_maps')
    output_confidence_maps_path=os.path.join(output_path, 'confidence_maps')
    output_dyna_max_path=os.path.join(output_path, 'dyna_max')
    output_dyna_avg_path=os.path.join(output_path, 'dyna_avg')

    os.makedirs(output_colmap_path, exist_ok=True)
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_dynamic_masks_path, exist_ok=True)
    os.makedirs(output_enlarged_dynamic_masks_path, exist_ok=True)
    os.makedirs(output_depth_maps_path, exist_ok=True)
    os.makedirs(output_confidence_maps_path, exist_ok=True)
    os.makedirs(output_dyna_max_path, exist_ok=True)
    os.makedirs(output_dyna_avg_path, exist_ok=True)

    traj = file_interface.read_tum_trajectory_file(os.path.join(dataset_path, "pred_traj.txt"))
    xyz = traj.positions_xyz
    quat = traj.orientations_quat_wxyz
    timestamps_mat = traj.timestamps
    traj_tum = np.column_stack((xyz, quat))

    # load pred intrinsics
    K_flattened = np.loadtxt(os.path.join(dataset_path, "pred_intrinsics.txt"), dtype=np.float32)  
    K = K_flattened.reshape(-1, 3, 3)

    # Copy pred_intrinsics to output path
    output_intrinsics_file = os.path.join(output_path, "pred_intrinsics.txt")
    os.system(f"cp {os.path.join(dataset_path, 'pred_intrinsics.txt')} {output_intrinsics_file}")

    # Copy pred_traj to output path
    output_traj_file = os.path.join(output_path, "pred_traj.txt")
    os.system(f"cp {os.path.join(dataset_path, 'pred_traj.txt')} {output_traj_file}")
    

    rgb_files = sorted(Path(dataset_path).glob('frame_*.png'), key=lambda x: int(x.stem.split('_')[-1]))
    rgbs = []
    for rgb_file in rgb_files:
        output_rgb_file = os.path.join(output_images_path, rgb_file.name)
        os.system(f"cp {rgb_file} {output_rgb_file}")


    ori_size = np.array(Image.open(rgb_files[0]).convert('RGB')).shape[:2][::-1]
    intrinsics = K
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))

    train_img_list = rgb_files

    poses = []
    for i in range(traj_tum.shape[0]):
        pose = tumpose_to_c2w(traj_tum[i])
        poses.append(pose)

    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)

    # load predict dynamic masks
    mask_files = sorted(Path(dataset_path).glob('dynamic_mask_*.png'), key=lambda x: int(x.stem.split('_')[-1]))
    for i, mask_file in enumerate(mask_files):
        output_mask_file = os.path.join(output_dynamic_masks_path, f"dynamic_mask_{i:04d}.png")
        os.system(f"cp {mask_file} {output_mask_file}")
        mask_file = str(mask_file).replace('dynamic_mask', 'enlarged_dynamic_mask')
        output_mask_file = str(output_mask_file).replace('dynamic_mask', 'enlarged_dynamic_mask')
        os.system(f"cp {mask_file} {output_mask_file}")
        
    # load predict depth
    depth_map_files = sorted(Path(dataset_path).glob('frame_*.npy'), key=lambda x: int(x.stem.split('_')[-1]))
    for i, depth_map_file in enumerate(depth_map_files):
        output_depth_map_file = os.path.join(output_depth_maps_path, f"frame_{i:04d}.npy")
        os.system(f"cp {depth_map_file} {output_depth_map_file}")

    # load confidence
    confidence_files = sorted(Path(dataset_path).glob('conf_*.npy'), key=lambda x: int(x.stem.split('_')[-1]))
    for i, confidence_file in enumerate(confidence_files):
        output_confidence_file = os.path.join(output_confidence_maps_path, f"conf_{i:04d}.npy")
        os.system(f"cp {confidence_file} {output_confidence_file}")

    # load dyna_max
    dyna_max_files = sorted(Path(dataset_path).glob('dyna_max_*.npy'), key=lambda x: int(x.stem.split('_')[-1]))
    for i, dyna_max_file in enumerate(dyna_max_files):
        output_dyna_max_file = os.path.join(output_dyna_max_path, f"dyna_max_{i:04d}.npy")
        os.system(f"cp {dyna_max_file} {output_dyna_max_file}")

    # load dyna_avg
    dyna_avg_files = sorted(Path(dataset_path).glob('dyna_avg_*.npy'), key=lambda x: int(x.stem.split('_')[-1]))
    for i, dyna_avg_file in enumerate(dyna_avg_files):
        output_dyna_avg_file = os.path.join(output_dyna_avg_path, f"dyna_avg_{i:04d}.npy")
        os.system(f"cp {dyna_avg_file} {output_dyna_avg_file}")
    



def depth_to_pts3d(K, pose, W, H, depth):
    # Get depths and  projection params if not provided
    assert (K[:, 0, 0] == K[:, 1, 1]).all()
    focals = K[:, 0, 0] 
    pp = K[:, :2, 2]
    im_poses = pose
    grid = [xy_grid(W, H) for _ in range(len(depth))] 

    # get pointmaps in camera frame
    rel_ptmaps = _fast_depthmap_to_pts3d(torch.tensor(depth), torch.tensor(grid), torch.tensor(focals), pp=torch.tensor(pp))
    # project to world frame
    return geotrf(torch.tensor(im_poses), rel_ptmaps)

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o+s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d+1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim-2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1]+1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal[:, None, None]
    
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    pixel_grid = pixel_grid.reshape(len(depth), -1, 2)
    depth = depth.reshape(len(depth), -1, 1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def tumpose_to_c2w(tum_pose):
    """
    Convert a TUM pose (translation and quaternion) back to a camera-to-world matrix (4x4) in CUDA mode.
    
    input: tum_pose - 7-element array: [x, y, z, qw, qx, qy, qz]
    output: c2w - 4x4 camera-to-world matrix
    """
    # Extract translation and quaternion from the TUM pose
    xyz = tum_pose[:3]

    # the order should be qx qy qz qw
    qw, qx, qy, qz = tum_pose[3:]
    quat = torch.tensor([qx, qy, qz, qw])

    # Convert quaternion to rotation matrix using PyTorch3D
    R = quaternion_to_matrix(quat.unsqueeze(0)).squeeze(0).numpy()  # 3x3 rotation matrix

    # Create the 4x4 camera-to-world matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R  # Rotation part
    c2w[:3, 3] = xyz  # Translation part
    
    return c2w

def save_colmap_images(poses, images_file, train_img_list):
    with open(images_file, 'w') as f:
        for i, pose in enumerate(poses, 1):  # Starting index at 1
            # breakpoint()
            pose = np.linalg.inv(pose)
            R = pose[:3, :3]
            t = pose[:3, 3]
            q = R_to_quaternion(R)  # Convert rotation matrix to quaternion
            f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} {train_img_list[i-1]}\n")
            f.write(f"\n")

def save_colmap_cameras(ori_size, intrinsics, camera_file):
    with open(camera_file, 'w') as f:
        for i, K in enumerate(intrinsics, 1):  # Starting index at 1
            width, height = ori_size
            scale_factor_x = width/2  / K[0, 2]
            scale_factor_y = height/2  / K[1, 2]
            # assert scale_factor_x==scale_factor_y, "scale factor is not same for x and y"
            # print(f'scale factor is not same for x{scale_factor_x} and y {scale_factor_y}')
            f.write(f"{i} PINHOLE {width} {height} {K[0, 0]*scale_factor_x} {K[1, 1]*scale_factor_x} {width/2} {height/2}\n")               # scale focal
            # f.write(f"{i} PINHOLE {width} {height} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def R_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.

    Parameters:
    - R: A 3x3 numpy array representing a rotation matrix.

    Returns:
    - A numpy array representing the quaternion [w, x, y, z].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z])

if __name__ == "__main__":
    dataset_path = 'results/davis'
    output_path = dataset_path.replace('davis', 'davis_rearranged')

    dataset_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    for seq in sorted(dataset_folders):
        if seq != '__pycache__':
            print(f'Processing {seq}')
            data_path = os.path.join(dataset_path, seq)
            output_path = data_path.replace('davis', 'davis_rearranged')
            main(data_path, output_path)