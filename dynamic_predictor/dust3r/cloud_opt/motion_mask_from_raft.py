import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import contextlib

from dust3r.cloud_opt.base_opt import BasePCOptimizer, edge_str
from dust3r.cloud_opt.pair_viewer import PairViewer
from dust3r.cloud_opt.camera_estimator import camera_parameter_estimation
from dust3r.utils.geometry import xy_grid, geotrf, depthmap_to_pts3d
from dust3r.utils.device import to_cpu, to_numpy
from dust3r.utils.goem_opt import DepthBasedWarping, OccMask, WarpImage, depth_regularization_si_weighted, tum_to_pose_matrix
from third_party.raft import load_RAFT
from dust3r.utils.image import rgb


def get_flow(imgs1, imgs2): #TODO: test with gt flow
    # print('precomputing flow...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_valid_flow_mask = OccMask(th=3.0)
    pair_imgs = [np.stack(imgs1), np.stack(imgs1)]

    flow_net = load_RAFT("third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
    flow_net = flow_net.to(device)
    flow_net.eval()

    with torch.no_grad():
        chunk_size = 12
        flow_ij = []
        flow_ji = []
        num_pairs = len(pair_imgs[0])
        for i in range(0, num_pairs, chunk_size):
            end_idx = min(i + chunk_size, num_pairs)
            imgs_ij = [torch.tensor(pair_imgs[0][i:end_idx]).float().to(device),
                    torch.tensor(pair_imgs[1][i:end_idx]).float().to(device)]
            flow_ij.append(flow_net(imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                    imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                    iters=20, test_mode=True)[1])
            flow_ji.append(flow_net(imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                    imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                    iters=20, test_mode=True)[1])

        flow_ij = torch.cat(flow_ij, dim=0)
        flow_ji = torch.cat(flow_ji, dim=0)
        valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
        valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
    # print('flow precomputed')
    # delete the flow net
    if flow_net is not None: del flow_net
    return flow_ij, flow_ji, valid_mask_i, valid_mask_j

def get_motion_mask_from_pairs(batch_result, motion_mask_thre=0.35):
    view1, view2, pred1, pred2 = batch_result['view1'], batch_result['view2'], batch_result['pred1'], batch_result['pred2']
    imgs1 = [rgb(view1['img'][i]) for i in range(view1['img'].shape[0])] 
    imgs2 = [rgb(view2['img'][i]) for i in range(view2['img'].shape[0])]
    
    flow_ij, flow_ji, valid_mask_i, valid_mask_j = get_flow(imgs1, imgs2)
    
    depth_wrapper = DepthBasedWarping() 
    print('precomputing self motion mask...')
    dynamic_masks = []
    for pair_i in range(view1['img'].shape[0]):

        v1 = {}
        v2 = {}
        p1 = {}
        p2 = {}

        for key in ['true_shape']:
            v1[key] = view1[key][pair_i]
            v2[key] = view2[key][pair_i]

        for key in pred1.keys():
            p1[key] = pred1[key][pair_i]

        for key in pred2.keys():
            p2[key] = pred2[key][pair_i]

        conf, K, focal, R2, T2, depth_1, depth_2 = camera_parameter_estimation(v1, v2, p1, p2, p2['conf'])
        K = torch.tensor(K).cuda()[None]
        T2 = T2[None, :, None]
        depth_1 = depth_1[None,None]
        R2 = R2[None]
        ego_flow_1_2, _ = depth_wrapper(torch.eye(3).cuda()[None], torch.zeros_like(T2), R2, T2, 1 / (depth_1 + 1e-6), K, torch.linalg.inv(K))

        err_map_i = torch.norm(ego_flow_1_2[:, :2, ...] - flow_ij[pair_i], dim=1)
        # normalize the error map for each pair
        err_map_i = (err_map_i - err_map_i.amin(dim=(1, 2), keepdim=True)) / (err_map_i.amax(dim=(1, 2), keepdim=True) - err_map_i.amin(dim=(1, 2), keepdim=True))


        dynamic_masks.append(err_map_i > motion_mask_thre)

    return dynamic_masks
            