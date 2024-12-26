import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from davis2017.davis import MaskDataset
from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017 import utils
from davis2017.results import Results
from scipy.optimize import linear_sum_assignment
from skimage.transform import resize
import cv2
import PIL

def _resize_pil_image(img, long_edge_size, nearest=False):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS if not nearest else PIL.Image.NEAREST
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def crop_img(img, size, square_ok=False, nearest=True, crop=True):
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)), nearest=nearest)
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size, nearest=nearest)
    W, H = img.size
    cx, cy = W//2, H//2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        if crop:
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        else: # resize
            img = img.resize((2*halfw, 2*halfh), PIL.Image.NEAREST)
    return img


class MaskEvaluation(object):
    def __init__(self, root, sequences):
        self.dataset = MaskDataset(root=root, sequences=sequences)
        self.sequences = sequences


    @staticmethod
    def _evaluate(all_gt_masks, all_res_masks, all_void_masks, metric):
        for i in range(len(all_gt_masks)):
            all_gt_masks[i]= (np.array(crop_img(all_gt_masks[i], 512, square_ok=True)) > 0.5) * 255 

        for i in range(len(all_res_masks)):
            all_res_masks[i]= np.array(all_res_masks[i])   
        
        for i in range(len(all_res_masks)):
            if i % 10 == 0:
                concatenated_mask = np.concatenate((all_gt_masks[i], all_res_masks[i]), axis=1).astype(np.uint8)
                import matplotlib.pyplot as plt
                plt.imshow(concatenated_mask, cmap='gray')
                plt.title(f'Mask {i}')
                plt.show()

        all_gt_masks = np.stack(all_gt_masks, axis=0)
        all_res_masks = np.stack(all_res_masks, axis=0)


        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            all_res_masks = all_res_masks[:all_gt_masks.shape[0], ...]
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
            # Resize all_res_masks to match all_gt_masks using interpolation

        
        # all_res_masks = resized_res_masks
        
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
       
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    def evaluate(self, res_path, metric=('J', 'F')):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        results = MaskDataset(root=res_path, sequences=self.sequences, is_label=False)

        # Sweep all sequences
        for seq in tqdm(self.sequences):
            all_gt_masks = self.dataset.read_masks(seq)
            all_res_masks = results.read_masks(seq)
            j_metrics_res, f_metrics_res = self._evaluate(all_gt_masks, all_res_masks, None, metric)
            for ii in range(len(all_gt_masks)):
                seq_name = f'{seq}_{ii+1}'
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM

        return metrics_res
