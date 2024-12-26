import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image


class MaskDataset(object):
    def __init__(self, root, sequences, is_label=True):
        self.is_label = is_label
        self.sequences = {}
        for seq in sequences:
            print(root, seq)
            if is_label:
                masks = np.sort(glob(os.path.join(root, seq, '*.png'))).tolist()
            else:
                masks = sorted(glob(os.path.join(root, seq, 'dynamic_mask_*.png')), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            self.sequences[seq] = masks
    def read_masks(self, seq):
        masks = []
        for msk in self.sequences[seq]:
            if self.is_label:
                img = np.array(Image.open(msk))
                img[img>0] = 255
                img = Image.fromarray(img)  
                masks.append(img)
            else:
                masks.append(Image.open(msk))
        return masks
