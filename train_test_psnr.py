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
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_confidence
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.cameras import Camera
from utils.pose_utils import get_camera_from_tensor
import torchvision
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation
import random

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=args, shuffle=False)      
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

        
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if args.optim_pose==False:
            gaussians.get_P().requires_grad_(False)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 3000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background


        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        pose = gaussians.get_RT(viewpoint_cam.uid)


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        static = gaussians._conf_static[viewpoint_cam.uid]

        image = image * static 
        gt_image = gt_image * static
        Ll1 = l1_loss(image, gt_image, reduce=False)
        Lssim = ssim(image, gt_image, size_average=False)

        psnr_frame = psnr(image, gt_image).mean()

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)
        loss = (loss).mean()

        
        loss.backward(retain_graph=True)

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            if psnr_frame > args.psnr_threshold:
                gaussians.optimizer_cam.step()
            gaussians.optimizer_cam.zero_grad(set_to_none = True)

        if not viewpoint_stack:

            viewpoint_stack = scene.getTestCameras().copy()
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            while len(viewpoint_stack) > 0:

                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                # print(f"getting test cam pose frame {viewpoint_cam.colmap_id}")

                pose = gaussians.get_RT_test(viewpoint_cam.uid)

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                gt_static_mask = 1 - viewpoint_cam.gt_dynamic_mask.to("cuda")
                image = image * gt_static_mask
                gt_image = gt_image * gt_static_mask

                Ll1 = l1_loss(image, gt_image, reduce=False)
                Lssim = ssim(image, gt_image, size_average=False)

                psnr_frame = psnr(image, gt_image).mean()

                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)
                loss = (loss).mean()

                
                loss.backward(retain_graph=True)

                with torch.no_grad():
                    # gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if psnr_frame > args.psnr_threshold:
                        gaussians.optimizer_cam.step()
                    gaussians.optimizer_cam.zero_grad(set_to_none = True)

        iter_end.record()

        with torch.no_grad():

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            # if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            # if iteration < opt.iterations:


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
    
def save_pose(path, quat_pose, train_cams, llffhold=2):
    output_poses=[]
    index_colmap = [cam.colmap_id for cam in train_cams]
    for quat_t in quat_pose:
        w2c = get_camera_from_tensor(quat_t)
        output_poses.append(w2c)

    return index_colmap, output_poses

def convert_colmap_to_quat(colmap_poses):
    quat_pose = []
    for pose in colmap_poses:
        rotation = Rotation.from_matrix(pose[:3, :3])
        quat = rotation.as_quat()
        translation = pose[:3, 3]
        quat_pose.append(np.concatenate([quat, translation]))
    return quat_pose

def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation
    
    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    c2w = c2w
    c2w = np.linalg.inv(c2w)
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose

def tumpose_to_c2w(tum_pose):
    """
    Convert a tuple of translation and rotation to a camera-to-world matrix
    
    input: tum_pose: tuple of translation and rotation (x y z qw qx qy qz)
    output: c2w: 4x4 matrix
    """
    xyz = tum_pose[:3]
    qw, qx, qy, qz = tum_pose[3:]
    rot = Rotation.from_quat([qx, qy, qz, qw])
    c2w = np.eye(4)
    c2w[:3, :3] = rot.as_matrix()
    c2w[:3, -1] = xyz
    c2w = np.linalg.inv(c2w)
    return c2w

def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):

    # Report test and samples of training set
    if iteration in testing_iterations:

        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lens = 0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    torchvision.utils.save_image(
                        image, os.path.join(scene.model_path, "{0:05d}".format(viewpoint.colmap_id) + ".png")
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if hasattr(viewpoint, 'gt_dynamic_mask'):
                        gt_static_mask = 1 - viewpoint.gt_dynamic_mask.to("cuda")

                        np.save(os.path.join(scene.model_path, f"{viewpoint.colmap_id}_image.npy"), image.cpu().numpy())
                        np.save(os.path.join(scene.model_path, f"{viewpoint.colmap_id}_gt_image.npy"), gt_image.cpu().numpy())
                        np.save(os.path.join(scene.model_path, f"{viewpoint.colmap_id}_gt_static_mask.npy"), gt_static_mask.cpu().numpy())
                        
                        image = image * gt_static_mask
                        gt_image = gt_image * gt_static_mask
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        # import matplotlib.pyplot as plt

                        # plt.figure(figsize=(10, 5))

                        # plt.subplot(1, 2, 1)
                        # plt.title('Predicted Image')
                        # plt.imshow(image.cpu().permute(1,2,0))
                        # plt.axis('off')

                        # plt.subplot(1, 2, 2)
                        # plt.title('Ground Truth Image')
                        # plt.imshow(gt_image.cpu().permute(1,2,0))
                        # plt.axis('off')

                        # plt.show()
                        lens += 1




                psnr_test /= lens
                l1_test /= lens          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                with open(os.path.join(scene.model_path, f"{config['name']}_log.txt"), 'a') as log_file:
                    log_file.write(f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}\n")

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--optim_pose", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--eval_pose", action="store_true")
    parser.add_argument('--pose_eval_interval', type=int, default=100)
    parser.add_argument('--psnr_threshold', type=float, default=26)
    parser.add_argument('--gt_dynamic_mask', type=str, default='data/sintel/training/dynamic_label_perfect')
    parser.add_argument('--dataset', type=str, default='sintel')

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    lp.eval = True
    args.eval = True
    
    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
