import os
import numpy as np
from PIL import Image
import sys


class Results(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _read_mask(self, sequence, frame_id):
        # mask_path = os.path.join(self.root_dir, sequence, f'{frame_id}.png')

        mask_path = os.path.join(self.root_dir, sequence, f'{int(frame_id[-4:]) -1 :05d}.png')
        obj = Image.open(mask_path)
        obj_mode = obj.mode
        if obj_mode == 'LA':
            obj = np.array(obj)[:,:,0]
        else:
            obj = np.array(obj)

        return obj


    def read_masks(self, sequence, masks_id):
        mask_0 = self._read_mask(sequence, masks_id[0])
        masks = np.zeros((len(masks_id), *mask_0.shape))
        for ii, m in enumerate(masks_id):
            masks[ii, ...] = self._read_mask(sequence, m)
        num_objects = 1
        tmp = np.ones((num_objects, *masks.shape)) * 255
        masks = (tmp == masks[None, ...]) > 0
        return masks
