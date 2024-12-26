import numpy as np
import torch
import torch.nn as nn
import cv2

from dust3r.utils.geometry import inv, geotrf, depthmap_to_absolute_camera_coordinates
from dust3r.post_process import estimate_focal_knowing_depth


def camera_parameter_estimation(view_n, view_m, pred_n, pred_m, im_conf):
    

    # for frame n and m.
    # pair_n2m: n->m
    conf            = float(pred_n['conf'].mean() * pred_m['conf'].mean())
    K, focal        = estimate_intrinsic(view_n['true_shape'], pred_n['pts3d'], pred_n['conf'])
    
    try:
        rel_pose_n2m = estimate_extrinsic(view_m['true_shape'], pred_m['pts3d_in_other_view'], im_conf.cpu().numpy(), K)
        R_mn, T_mn = rel_pose_n2m[:3, :3], rel_pose_n2m[:3, 3]
    except Exception as e:
        print(f"Error estimating extrinsic parameters: {e}")
        rel_pose_n2m = torch.eye(4).cuda()
        R_mn, T_mn = rel_pose_n2m[:3, :3], rel_pose_n2m[:3, 3]
    

    # ptcloud is expressed in camera n
    

    depth_n, depth_m = pred_n['pts3d'][..., 2], geotrf(inv(rel_pose_n2m), pred_m['pts3d_in_other_view'])[..., 2]
    
    return conf, K, focal, R_mn, T_mn, depth_n, depth_m


def estimate_intrinsic(img_shape, pts3d, conf):
    H, W = img_shape
    pts3d = pts3d.cpu()
    pp = torch.tensor((W/2, H/2))
    focal = float(estimate_focal_knowing_depth(pts3d[None], pp, focal_mode='weiszfeld')) 
    K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

    return K, focal

def estimate_extrinsic(img_shape, pts3d, conf, K):
    min_conf_thr = 3

    H, W = img_shape
    H, W = H.item(), W.item()
    # estimate the pose of pts1 in image 2
    pts3d = pts3d.cpu().numpy()
    pixels = np.mgrid[:W, :H].T.astype(np.float32)
    msk = (conf > min_conf_thr)

    res = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
    success, R, T, inliers = res
    assert success

    R = cv2.Rodrigues(R)[0]  # world to cam
    pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world

    return torch.from_numpy(pose.astype(np.float32)).cuda()


