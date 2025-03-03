# DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction

<b>DAS3R</b> is a novel framework for scene decomposition and static background reconstruction from unposed videos. By integrating the trained motion masks and modeling the static scene as Gaussian splats with dynamics-aware optimization, DAS3R is more robust in complex motion scenarios, capable of handling videos where dynamic objects occupy a significant portion of the scene, and does not require camera pose inputs or point cloud data from SLAM-based methods.

This repository is the official implementation of the paper:

[**DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction**](https://kai422.github.io/DAS3R/)

[*Kai Xu*](https://kai422.github.io/),
[*Tze Ho Elden Tse*](https://eldentse.github.io/),
[*Jizong Peng*](),
[*Angela Yao*](https://www.comp.nus.edu.sg/~ayao/)
<h5 align="LEFT">

[![arXiv](https://img.shields.io/badge/ArXiv-2412.19584-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.19584) 
[![Home Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://kai422.github.io/DAS3R/) 

</h5>


![Demo](assets/davis.gif)
![Demo](assets/sintel.gif)

## Getting Started


### Installation
- Clone DAS3R.
```bash
git clone --recursive https://github.com/kai422/DAS3R.git
cd DAS3R
```

- Create the environment (or use pre-built docker), here we show an example using conda.
```bash
conda create -n das3r python=3.11 cmake=3.14.0
conda activate das3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install submodules/simple-knn
# modify the rasterizer
vim submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h
'p_view.z <= 0.2f' -> 'p_view.z <= 0.001f' # line 154
pip install submodules/diff-gaussian-rasterization
```

- Optional but highly suggested, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd dynamic_predictor/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../..
```

## Gradio Demo

To start the Gradio server, simply execute the `app.py` script using Python. Once the server is running, you can access the demo at `http://0.0.0.0:7860`.
```bash
pip install --upgrade gradio
python app.py
```

## Data Preparation

- Download and prepare the **DAVIS** dataset for evaluation.
```bash
cd data; python download_davis.py; cd ..
```
- Download and prepare the **Sintel** dataset for evaluation.
```bash
cd data; bash download_sintel.sh; cd ..
```
- If you need to train dynamic_predictor, download the **PointOdyssey** dataset.
```bash
cd data; bash download_pointodyssey.sh; cd ..
```
- Rearrange it for faster data loading.
```bash
cd datasets_preprocess; python pointodyssey_rearrange.py; cd ..
```

## Training Dynamic-Aware Gaussian Splatting

- First compute dynamic masks and coarse geometric initialization or download them directly from  [Google Drive](https://drive.google.com/drive/folders/1uSI3raipU3aacSq5enAZd8EozSTn_kS9?usp=drive_link).
```bash
# download the pretrained RAFT weights:
cd dynamic_predictor; sh download_ckpt.sh; cd ..

# For DAVIS dataset:
python dynamic_predictor/launch.py --mode=eval_pose \
        --pretrained=Kai422kx/das3r \
        --eval_dataset=davis \
        --output_dir=results/davis \
        --use_pred_mask --evaluate_davis
# For Sintel dataset:
python dynamic_predictor/launch.py --mode=eval_pose \
        --pretrained=Kai422kx/das3r \
        --eval_dataset=sintel \
        --output_dir=results/sintel \
        --use_pred_mask

# Alternatively, download the precomputed results and place them in the `results` directory. Then, unzip `davis.zip` and `sintel.zip` in the same directory.

# For your own dataset:
python dynamic_predictor/launch.py --mode=eval_pose_custom \
        --pretrained=Kai422kx/das3r \
        --dir_path=data/custom/images \
        --output_dir=data/custom/output \
        --use_pred_mask 
```
- Rearrange results:

```bash
python utils/rearrange_davis.py
python utils/rearrange_sintel.py

# For your own dataset:
python utils/rearrange.py --output_dir=data/custom/output
```
**Training Gaussian Splatting:**

- For specific sequence with GUI:
```bash
python train_gui.py -s ${input_folder} -m ${output_folder} --iter ${total_iterations} --eval_pose --gui

# for example
python train_gui.py -s results/sintel_rearranged/market_2 -m results/sintel_rearranged/market_2 --iter 4000 --eval_pose --gui 

# for your own dataset
python train_gui.py -s data/custom/output_rearranged -m data/custom/output_rearranged --iter 4000 --gui 
```
- For training on all frames and rendering:
```bash
bash scripts/rendering_davis.sh
bash scripts/rendering_sintel.sh

# for your own dataset
python train_gui.py -s data/custom/output_rearranged -m data/custom/output_rearranged --iter 4000
python render.py -s data/custom/output_rearranged -m data/custom/output_rearranged --iter 4000 --get_video 
```

- For evaluation on test frames:
```bash
bash scripts/testing_psnr_davis.sh
bash scripts/testing_psnr_sintel.sh
```




## Training Dynamic Predictor

You can train DAS3R directly with:
```bash
bash DAS3R_b32_g4.sh
```

## Citation

If you find our work useful, please cite:

```bibtex
@article{xu2024das3r,
 title     = {DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction}, 
 author    = {Xu, Kai and Tse, Tze Ho Elden and Peng, Jizong and Yao, Angela},
 journal   = {arXiv preprint arxiv:2412.19584},
 year      = {2024}
}
```


## Acknowledgements
Our code builds upon the work of several outstanding projects, including [InstantSplat](https://github.com/NVlabs/InstantSplat), [MonST3R](https://github.com/Junyi42/monst3r), [DUSt3R](https://github.com/naver/dust3r), and [CasualSAM](https://github.com/ztzhang/casualSAM). Additionally, our camera pose estimation evaluation script is adapted from [LEAP-VO](https://github.com/chiaki530/leapvo), and our GUI implementation is based on [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians). We extend our gratitude to the authors for their contributions.
