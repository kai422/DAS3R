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
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_confidence
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.pose_utils import get_camera_from_tensor
from utils.vo_eval import load_traj, eval_metrics, plot_trajectory
from utils.gui_utils import orbit_camera, OrbitCamera
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from time import perf_counter, time

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

class GUI:
    def __init__(self, gui, w, h) -> None:
        self.gui = gui
        # For UI
        self.visualization_mode = 'RGB'
        self.W, self.H = w*2, h*2
        self.cam = OrbitCamera(self.W, self.H, r=5, fovy=50)

        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.buffer_image_gt = np.ones((self.W//2, self.H//2, 3), dtype=np.float32)
        self.buffer_image_dynamic_blend = np.ones((self.W//2, self.H//2, 3), dtype=np.float32)
        self.buffer_depth_model = np.ones((self.W//2, self.H//2, 3), dtype=np.float32)
        self.buffer_dynamic_blend_gt = np.ones((self.W//2, self.H//2, 3), dtype=np.float32)
        self.buffer_depth_gt = np.ones((self.W//2, self.H//2, 3), dtype=np.float32)
        self.buffer_conf_rendered = np.ones((self.W//2, self.H//2, 3), dtype=np.float32)
        self.buffer_image_traj = np.ones((550, 300, 3), dtype=np.float32)

        self.training = False


    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )
            dpg.add_raw_texture(
                self.W//2,
                self.H//2,
                self.buffer_image_gt,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture_gt",
            )
            dpg.add_raw_texture(
                self.W//2,
                self.H//2,
                self.buffer_image_gt,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture_dynamic_blend",
            )
            
            dpg.add_raw_texture(
                self.W//2,
                self.H//2,
                self.buffer_depth_model,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture_depth_model",
            )
            dpg.add_raw_texture(
                550,
                300,
                self.buffer_depth_model,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture_traj",
            )
            
            dpg.add_raw_texture(
                self.W//2,
                self.H//2,
                self.buffer_conf_rendered,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture_conf",
            )
            dpg.add_raw_texture(
                self.W//2,
                self.H//2,
                self.buffer_dynamic_blend_gt,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture_dynamic_blend_gt",
            )
            
            
        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # Show model rendered depth image
        with dpg.window(
            tag="_rendered_conf_window",
            label="GS Staticness",
            width=self.W//2,
            height=self.H//2,
            pos=[0, self.H],
            no_move=True,
            no_scrollbar=True,
        ):
            dpg.add_image("_texture_conf")

        with dpg.window(
            tag="_dynamic_blend_window_gt",
            label="GT Dynamic Mask",
            width=self.W//2,
            height=self.H//2,
            pos=[self.W//2, self.H],
            no_move=True,
            no_scrollbar=True,
        ):
            dpg.add_image("_texture_dynamic_blend_gt")

        # Show ground truth RGB image
        with dpg.window(
            tag="_ground_truth_window",
            label="Ground Truth RGB",
            width=self.W//2,
            height=self.H//2,
            pos=[0, self.H + self.H//2],
            no_move=True,
            no_scrollbar=True,
        ):
            dpg.add_image("_texture_gt")

        # Show ground truth RGB image
        with dpg.window(
            tag="_dynamic_blend_window",
            label="Model Pred Dynamic Mask",
            width=self.W//2,
            height=self.H//2,
            pos=[self.W//2, self.H + self.H//2],
            no_move=True,
            no_scrollbar=True,
        ):
            dpg.add_image("_texture_dynamic_blend")


        # pose
        with dpg.window(
            tag="_pose_eval_window",
            label="Pose Evaluation",
            width=600,
            height=self.H,
            pos=[self.W, self.H],
            no_move=True,
            no_scrollbar=True,
        ):
            dpg.add_text("", tag="_pose_log_input")
            dpg.add_image("_texture_traj")


        # control window
        with dpg.window(
            tag="_info_window",
            label="Info",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_scrollbar=True,
        ):
            dpg.add_text("Loss: ", tag="_loss_log")
            dpg.add_text("Training PSNR: ", tag="_pnsr_log")



        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian",
            width=self.W + 600,
            height=self.H + self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        dpg.show_viewport()
        

    @torch.no_grad()
    def test_step(self, vstacks, iteration, gaussians, pipe, bg, seq, pose_path, pose_eval_interval=50, eval_pose=True, msg=None):

        for k, v in msg.items():
            dpg.set_value(k, str(v))

        if eval_pose and (iteration % pose_eval_interval ==0 or iteration == 1):
            poses = np.load(pose_path)

            tt = np.arange(len(poses)).astype(float)
            tum_poses = [c2w_to_tumpose(p) for p in poses]
            tum_poses = np.stack(tum_poses, 0)
            pred_traj = [tum_poses, tt]
        
            gt_traj_file = f'/home/remote/data/sintel/training/camdata_left/{seq}'
            gt_traj = load_traj(gt_traj_file=gt_traj_file)


            _, ate, rpe_trans, rpe_rot = eval_metrics(
                pred_traj, gt_traj
            )

            pose_eval = f'iter: {iteration} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}'
            print(pose_eval)

            if self.gui:
                traj = plot_trajectory(
                    pred_traj, gt_traj, title=seq
                )
                dpg.set_value("_pose_log_input", pose_eval)

                self.buffer_image_traj = traj
                dpg.set_value(
                    "_texture_traj", self.buffer_image_traj
                ) 



        if self.gui:
            fps = 4
            viewpoint_cam = vstacks[int(time()*fps)%len(vstacks)]
            pose = gaussians.get_RT(viewpoint_cam.uid)
            self.cur_cam = viewpoint_cam


            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            buffer_image = image  # [3, H, W]


            buffer_image = torch.nn.functional.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)


            psnr = gaussians._conf_static[self.cur_cam.uid].unsqueeze(0).repeat(3, 1, 1)

            buffer_conf_rendered = torch.nn.functional.interpolate(
                psnr.unsqueeze(0),
                size=(self.H//2, self.W//2),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_conf_rendered = (
                buffer_conf_rendered.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )
            gt_image = self.cur_cam.original_image
            self.buffer_image_gt = (
                gt_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )


            
            alpha = 0.5
            dynamic_mask = self.cur_cam.dyna_avg_map
            buffer_image_dynamic_blend = (alpha * self.cur_cam.original_image.cpu().permute(1, 2, 0) + (1 - alpha) * dynamic_mask[:, :, None].cpu())
            self.buffer_image_dynamic_blend = (
                buffer_image_dynamic_blend
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )
            
            alpha = 0.5
            if hasattr(self.cur_cam, 'gt_dynamic_mask'):
                dynamic_mask = self.cur_cam.gt_dynamic_mask
                buffer_dynamic_blend_gt = (alpha * self.cur_cam.original_image.cpu().permute(1, 2, 0) + (1 - alpha) * dynamic_mask.cpu().permute(1, 2, 0))
            else:
                buffer_dynamic_blend_gt = self.cur_cam.original_image.cpu().permute(1, 2, 0)

            self.buffer_dynamic_blend_gt = (
                buffer_dynamic_blend_gt
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!
            dpg.set_value(
                "_texture_gt", self.buffer_image_gt
            )  # buffer must be contiguous, else seg fault!
            dpg.set_value(
                "_texture_dynamic_blend", self.buffer_image_dynamic_blend
            )  # buffer must be contiguous, else seg fault!
            
            dpg.set_value(
                "_texture_dynamic_blend_gt", self.buffer_dynamic_blend_gt
            )  # buffer must be contiguous, else seg fault!
            dpg.set_value(
                "_texture_conf", self.buffer_conf_rendered
            )  # buffer must be contiguous, else seg fault!       
            # dpg.set_value(
            #     "_texture_depth_gt", self.buffer_depth_gt
            # )  # buffer must be contiguous, else seg fault!          



    # no gui mode
    def train(self, iters=5000):
        if iters > 0:
            for i in tqdm.trange(iters):
                self.train_step()       

def save_pose(path, quat_pose, train_cams, llffhold=2):
    output_poses=[]
    index_colmap = [cam.colmap_id for cam in train_cams]
    for quat_t in quat_pose:
        w2c = get_camera_from_tensor(quat_t)
        output_poses.append(w2c)
    colmap_poses = []
    for i in range(len(index_colmap)):
        ind = index_colmap.index(i+1)
        bb=output_poses[ind]
        bb = bb#.inverse()
        colmap_poses.append(bb)
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)

def convert_colmap_to_quat(colmap_poses):
    quat_pose = []
    for pose in colmap_poses:
        rotation = Rotation.from_matrix(pose[:3, :3])
        quat = rotation.as_quat()
        translation = pose[:3, 3]
        quat_pose.append(np.concatenate([quat, translation]))
    return np.array(quat_pose)

def disable_gs_training(gaussians):
    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, gui: GUI):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=args, shuffle=False)      
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    train_cams_init = scene.getTrainCameras().copy()
    os.makedirs(scene.model_path + 'pose', exist_ok=True)
    save_pose(scene.model_path + 'pose' + "/pose_org.npy", gaussians.get_P(), train_cams_init)
    save_pose(scene.model_path + 'pose' + "/pose_test.npy", gaussians.get_P(), train_cams_init)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    start = perf_counter()
    if args.gui:
        dpg.create_context()
        gui.register_dpg()

    msg = {}
    
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        # disable_gs_training(gaussians)

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

        msg["_loss_log"] = f'[ITER {iteration}] Training Loss: {loss.item()}'
        
        loss.backward(retain_graph=True)

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            if psnr_frame > args.psnr_threshold:
                gaussians.optimizer_cam.step()
            gaussians.optimizer_cam.zero_grad(set_to_none = True)


        iter_end.record()

        with torch.no_grad():

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in testing_iterations:
                log = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                print(log)
                msg["_pnsr_log"] = log

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(scene.model_path + 'pose' + f"/pose_{iteration}.npy", gaussians.get_P(), train_cams_init)

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


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                

            if args.gui and (iteration % 4 == 0 or iteration == 1):
                # gui test
                if iteration % args.pose_eval_interval == 0:
                    save_pose(scene.model_path + 'pose' + "/pose_test.npy", gaussians.get_P(), train_cams_init)

                vstacks = scene.getTrainCameras()
                gui.test_step(vstacks, iteration, gaussians, pipe, bg, dataset.source_path.split('/')[-1], pose_path = scene.model_path + 'pose' + "/pose_test.npy", pose_eval_interval = args.pose_eval_interval, eval_pose=args.eval_pose, msg=msg)
                if gui.gui:
                    dpg.render_dearpygui_frame()

        end = perf_counter()
    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):

    torch.cuda.empty_cache()
    validation_configs = ({'name': 'train', 'cameras' : scene.getTrainCameras()},)

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
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                
                if hasattr(viewpoint, 'gt_dynamic_mask'):
                    gt_static_mask = 1 - viewpoint.gt_dynamic_mask.to("cuda")
                    image = image * gt_static_mask
                    gt_image = gt_image * gt_static_mask
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lens += 1

            psnr_test /= lens
            l1_test /= lens          
            log = "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test)
            with open(os.path.join(scene.model_path, f"{config['name']}_log.txt"), 'a') as log_file:
                log_file.write(f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}\n")
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

    return log
    
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--optim_pose", type=bool, default = True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--eval_pose", action="store_true")
    parser.add_argument('--pose_eval_interval', type=int, default=100)
    parser.add_argument('--psnr_threshold', type=float, default=26)
    parser.add_argument('--gt_dynamic_mask', type=str, default='/home/remote/data/sintel/training/dynamic_label_perfect')
    parser.add_argument('--dataset', type=str, default='sintel')

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.gui:
        w, h = Image.open(os.path.join(args.source_path, 'images', 'frame_0000.png')).size
        gui = GUI(gui = args.gui, w=w, h=h)
    else:
        gui = None

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args, gui)

    # All done
    print("\nTraining complete.")
