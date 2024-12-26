#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, to_open3d_point_cloud
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.vo_eval import file_interface
import torch
from utils.pose_utils import quad2rotation

class CameraInfo(NamedTuple):
    uid: int
    intr: object
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    dynamic_mask: np.array
    enlarged_dynamic_mask: np.array
    conf_map: np.array
    depth_map: np.array
    dyna_avg_map: np.array
    dyna_max_map: np.array
    original_pose: np.array
    gt_dynamic_mask: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    train_poses: list
    test_poses: list

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

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
    R = quad2rotation(quat.unsqueeze(0)).squeeze(0).numpy()  # 3x3 rotation matrix

    # Create the 4x4 camera-to-world matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R  # Rotation part
    c2w[:3, 3] = xyz  # Translation part
    
    return c2w


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, eval, opt):

    cam_infos = []
    poses = []
    original_poses=[]
    extrinsics_path = os.path.join(os.path.dirname(images_folder), "pred_traj.txt")

    traj = file_interface.read_tum_trajectory_file(extrinsics_path)
    xyz = traj.positions_xyz
    quat = traj.orientations_quat_wxyz
    timestamps_mat = traj.timestamps
    traj_tum = np.column_stack((xyz, quat))
    for i in range(traj_tum.shape[0]):
        pose = tumpose_to_c2w(traj_tum[i])
        original_poses.append(pose)

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()


        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        uid = intr.id

        height = intr.height
        width = intr.width            
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))
        poses.append(pose)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        seq_name = image_path.split("/")[-3]
        idx_str = image_path.split("/")[-1].split(".")[0].split("_")[-1]
        seq_path = '/'.join(image_path.split("/")[:-2])
        intrinsics_path = os.path.join(seq_path, "pred_intrinsics.txt")
        conf_path = os.path.join(seq_path, "confidence_maps", f"conf_{idx_str}.npy")
        depth_path = os.path.join(seq_path, "depth_maps", f"frame_{idx_str}.npy")
        dyna_avg_path = os.path.join(seq_path, "dyna_avg", f"dyna_avg_{idx_str}.npy")
        dyna_max_path = os.path.join(seq_path, "dyna_max", f"dyna_max_{idx_str}.npy")
        
        dynamic_mask_path = os.path.join(seq_path, "dynamic_masks", f"dynamic_mask_{idx_str}.png")
        enlarged_dynamic_mask_path = os.path.join(seq_path, "enlarged_dynamic_masks", f"enlarged_dynamic_mask_{idx_str}.png")

        if opt.dataset == 'sintel':
            gt_dynamic_mask_path = os.path.join(opt.gt_dynamic_mask, seq_name, f"frame_{int(idx_str)+1:04d}.png")
        elif opt.dataset == 'davis':
            gt_dynamic_mask_path = os.path.join(opt.gt_dynamic_mask, seq_name, f"{int(idx_str):05d}.png")

        try:
            conf_map = np.load(conf_path)
        except:
            conf_map = None
            
        try:
            K_flattened = np.loadtxt(intrinsics_path, dtype=np.float32)  
            K = K_flattened.reshape(-1, 3, 3)
            K = K[int(idx_str)]
        except:
            K = None

        try:
            depth_map = np.load(depth_path)
        except:
            depth_map = None

        try: 
            dyna_avg_map = np.load(dyna_avg_path)
        except:
            dyna_avg_map = None
        try:
            dyna_max_map = np.load(dyna_max_path)
        except:
            dyna_max_map = None
        try:
            dynamic_mask = np.array(Image.open(dynamic_mask_path)) / 255.0 > 0.5 
        except:
            dynamic_mask = None
        try:
            enlarged_dynamic_mask = np.array(Image.open(enlarged_dynamic_mask_path)) / 255.0 > 0.5 
        except:
            enlarged_dynamic_mask = None
        
        try:
            if opt.dataset == 'davis':
                gt_dynamic_mask = np.array(Image.open(gt_dynamic_mask_path)) > 0.5 
            else:
                gt_dynamic_mask = np.array(Image.open(gt_dynamic_mask_path)) / 255.0 > 0.5 
        except:
            gt_dynamic_mask = None

        # original_pose = None
        original_pose = original_poses[int(idx_str)]



        cam_info = CameraInfo(uid=uid, intr = intr, R=R, T=T, original_pose = original_pose, FovY=FovY, FovX=FovX, image=image, conf_map=conf_map, depth_map=depth_map,
                              image_path=image_path, image_name=image_name, width=width, height=height, dynamic_mask = dynamic_mask, enlarged_dynamic_mask = enlarged_dynamic_mask, dyna_avg_map=dyna_avg_map, dyna_max_map=dyna_max_map, gt_dynamic_mask=gt_dynamic_mask)
    
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos, poses

# For interpolated video, open when only render interpolated video
def readColmapCamerasInterp(cam_extrinsics, cam_intrinsics, images_folder, model_path):
    
    pose_interpolated_path = model_path + 'pose/pose_interpolated.npy'
    pose_interpolated = np.load(pose_interpolated_path)
    intr = cam_intrinsics[1]

    cam_infos = []
    poses=[]
    for idx, pose_npy in enumerate(pose_interpolated):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, pose_interpolated.shape[0]))
        sys.stdout.flush()

        extr = pose_npy
        intr = intr
        height = intr.height
        width = intr.width

        uid = idx
        R = extr[:3, :3].transpose()
        T = extr[:3, 3]
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))
        # print(uid)
        # print(pose.shape)
        # pose = np.linalg.inv(pose)
        poses.append(pose)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        images_list = os.listdir(os.path.join(images_folder))
        image_name_0 = images_list[0]
        image_name = str(idx).zfill(4)
        image = Image.open(images_folder + '/' + image_name_0)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=images_folder, image_name=image_name, width=width, height=height,
                              dynamic_mask = None, enlarged_dynamic_mask = None,
                              intr=None, conf_map=None, depth_map=None, dyna_avg_map=None, dyna_max_map=None, gt_dynamic_mask=None, original_pose=None)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, poses


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

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

def readColmapSceneInfo(path, images, eval, args, opt, llffhold=2):
    # try:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    #     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # except:

    ##### For initializing test pose using PCD_Registration

    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")

    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    if opt.get_video:
        cam_infos_unsorted, poses = readColmapCamerasInterp(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), model_path=args.model_path)
    else:
        cam_infos_unsorted, poses = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), eval=eval, opt=opt)
    sorting_indices = sorted(range(len(cam_infos_unsorted)), key=lambda x: cam_infos_unsorted[x].image_name)
    cam_infos = [cam_infos_unsorted[i] for i in sorting_indices]
    sorted_poses = [poses[i] for i in sorting_indices]
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)


    
    if eval:
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx+1) % llffhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx+1) % llffhold == 0]
        # train_poses = [c for idx, c in enumerate(sorted_poses) if (idx+1) % llffhold != 0]
        # test_poses = [c for idx, c in enumerate(sorted_poses) if (idx+1) % llffhold == 0]
        num_cams = len(cam_infos)
        offset = 5
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx + offset) % 10 == 0]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx + offset) % 10 != 0]
        train_poses = [c for idx, c in enumerate(sorted_poses) if (idx + offset) % 10 != 0]
        test_poses = [c for idx, c in enumerate(sorted_poses) if (idx + offset) % 10 == 0]


    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        train_poses = sorted_poses
        test_poses = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    try:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    except:
        pcd = None
        ply_path = None
    
    # Create an Open3D point cloud object
    # o3d.visualization.draw_geometries([to_open3d_point_cloud(pcd)])
    # np.save("poses_family.npy", sorted_poses)
    # breakpoint()
    # np.save("3dpoints.npy", pcd.points)
    # np.save("3dcolors.npy", pcd.colors)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_poses=train_poses,
                           test_poses=test_poses)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}