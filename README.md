# DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction

<b>DAS3R</b> is a novel framework for scene decomposition and static background reconstruction from uposed videos. By integrating the trained motion masks and modeling the static scene as Gaussian splats with dynamics-aware optimization, DAS3R is more robust in complex motion scenarios, capable of handling videos where dynamic objects occupy a significant portion of the scene, and does not require camera pose inputs or point cloud data from SLAM-based methods.

This repository is the official implementation of the paper:

[**DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction**](https://kai422.github.io/DAS3R/)

[*Kai Xu*](https://kai422.github.io/),
[*Tze Ho Elden Tse*](https://eldentse.github.io/),
[*Jizong Peng*](),
[*Angela Yao*](https://www.comp.nus.edu.sg/~ayao/)

arXiv, 2024. [**[Project Page]**](https://kai422.github.io/DAS3R/) [**[PDF]**](https://github.com/kai422/das3r_page/blob/main/DAS3R.pdf)

![DAVIS Example](assets/davis.gif)
![Sintel Example](assets/sintel.gif)

## Getting Started


### Installation
1. Clone DAS3R and download pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1uSI3raipU3aacSq5enAZd8EozSTn_kS9?usp=drive_link).
```bash
git clone --recursive git@github.com:kai422/das3r.git
cd das3r
git submodule update --init --recursive
```

2. Create the environment (or use pre-built docker), here we show an example using conda.
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

3. Optional but highly suggested, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd dynamic_predictor/croco/models/curope/
python setup.py build_ext --inplace
```


## Data Preparation

To download and prepare the **DAVIS** dataset for evaluation, execute:
```bash
cd data; python download_davis.py; cd ..
```
To download and prepare the **Sintel** dataset for evaluation, execute:
```bash
cd data; bash download_sintel.sh; cd ..
```
If you need to train dynamic_predictor, download the **PointOdyssey** dataset:
```bash
cd data; bash download_pointodyssey.sh; cd ..
```
Rearrange it for faster data loading:
```bash
cd datasets_preprocess; python pointodyssey_rearrange.py; cd ..
```

## Training Dynamic-Aware Gaussian Splatting

First compute dynamic masks and coarse geometric initialization or download it directly from  [Google Drive](https://drive.google.com/drive/folders/1uSI3raipU3aacSq5enAZd8EozSTn_kS9?usp=drive_link).
```bash
# For DAVIS dataset:
python dynamic_predictor/launch.py --mode=eval_pose \
        --pretrained=das3r_checkpoint-last.pth \
        --eval_dataset=davis \
        --output_dir=results/davis \
        --use_pred_mask 
# For Sintel dataset:
python dynamic_predictor/launch.py --mode=eval_pose \
        --pretrained=das3r_checkpoint-last.pth \
        --eval_dataset=sintel \
        --output_dir=results/sintel \
        --use_pred_mask 

# If you download it, put them under `results` and unzip davis.zip and sintel.zip.
```
Rearrange results:

```bash
python utils/rearrange_davis.py
python utils/rearrange_sintel.py
```
**Training Gaussian Splatting:**

For specific sequence with GUI:
```bash
python train_gui.py -s ${input_folder} -m ${output_folder} --iter ${total_iterations} --eval_pose --gui
# for example
python train_gui.py -s results/sintel/market_2 -m results/sintel/market_2 --iter 4000 --eval_pose --gui 
```
For training on all frames and rendering:
```bash
bash scripts/rendering_davis.sh
bash scripts/rendering_sintel.sh
```

For evaluation on test frames:
```bash
bash scripts/testing_psnr_davis.sh
bash scripts/testing_psnr_sintel.sh
```




## Training Dynamic Predictor

First download the pretrained MonST3R weights from [Google Drive](https://drive.google.com/file/d/1Z1jO_JmfZj0z3bgMvCwqfUhyZ1bIbc9E/view?usp=sharing) or via [Hugging Face](https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt).

Then, you can train the model using the following command:
```bash
bash dynamic_predictor/DAS3R_b32_g4.sh
```



## Acknowledgements
Our code builds upon the work of several outstanding projects, including [InstantSplat](https://github.com/NVlabs/InstantSplat), [MonST3R](https://github.com/Junyi42/monst3r), [DUSt3R](https://github.com/naver/dust3r), and [CasualSAM](https://github.com/ztzhang/casualSAM). Additionally, our camera pose estimation evaluation script is adapted from [LEAP-VO](https://github.com/chiaki530/leapvo), and our GUI implementation is based on [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians). We extend our gratitude to the authors for their contributions.
