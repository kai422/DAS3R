# build upon InstantSplat https://huggingface.co/spaces/kairunwen/InstantSplat/blob/main/app.py
import os, subprocess, shlex, sys, gc
import numpy as np
import shutil
import argparse
import gradio as gr
import uuid
import glob
import re

# import spaces

# subprocess.run(shlex.split("pip install wheel/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl"))
# subprocess.run(shlex.split("pip install wheel/simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl"))
# subprocess.run(shlex.split("pip install wheel/curope-0.0.0-cp310-cp310-linux_x86_64.whl"))

GRADIO_CACHE_FOLDER = './gradio_cache_folder'


def get_dust3r_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--model_path", type=str, default="submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--focal_avg", type=bool, default=True)
    parser.add_argument("--n_views", type=int, default=3)
    parser.add_argument("--base_path", type=str, default=GRADIO_CACHE_FOLDER) 
    return parser


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key.split('/')[-1])]
    return sorted(l, key=alphanum_key)

def cmd(command):
    print(command)
    os.system(command)

# @spaces.GPU(duration=300)
def process(inputfiles, input_path='demo'):
    if inputfiles:
        frames = natural_sort(inputfiles)
    else:
        frames = natural_sort(glob.glob('./assets/example/' + input_path + '/*'))
    if len(frames) > 40:
        stride = int(np.ceil(len(frames) / 40))
        frames = frames[::stride]
    
    # Create a temporary directory to store the selected frames
    temp_dir = os.path.join(GRADIO_CACHE_FOLDER, str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy the selected frames to the temporary directory
    for i, frame in enumerate(frames):
        shutil.copy(frame, f"{temp_dir}/{i:04d}.{frame.split('.')[-1]}")

    imgs_path = temp_dir
    output_path = f'./results/{input_path}/output'
    cmd(f"python dynamic_predictor/launch.py --mode=eval_pose_custom \
        --pretrained=Kai422kx/das3r \
        --dir_path={imgs_path} \
        --output_dir={output_path} \
        --use_pred_mask ")
    
    cmd(f"python utils/rearrange.py --output_dir={output_path}")
    output_path = f'{output_path}_rearranged'

    print(output_path)
    cmd(f"python train_gui.py -s {output_path} -m {output_path} --iter 4000")
    cmd(f"python render.py -s {output_path} -m {output_path} --iter 4000 --get_video")

    output_video_path = f"{output_path}/rendered.mp4"
    output_ply_path = f"{output_path}/point_cloud/iteration_4000/point_cloud.ply"
    return  output_video_path, output_ply_path, output_ply_path



_TITLE = '''DAS3R'''
_DESCRIPTION = '''
<div style="display: flex; justify-content: center; align-items: center;">
    <div style="width: 100%; text-align: center; font-size: 30px;">
        <strong>DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction</strong>
    </div>
</div> 
<p></p>


<div align="center">
    <a style="display:inline-block" href="https://arxiv.org/abs/2412.19584"><img src="https://img.shields.io/badge/ArXiv-2412.19584-b31b1b.svg?logo=arXiv" alt='arxiv'></a>
    <a style="display:inline-block" href="https://kai422.github.io/DAS3R/"><img src='https://img.shields.io/badge/Project-Website-blue.svg'></a>&nbsp;
    <a style="display:inline-block" href="https://github.com/kai422/DAS3R"><img src='https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white'></a>&nbsp;
</div>
<p></p>


* Official demo of [DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction](https://kai422.github.io/DAS3R/).
* You can explore the sample results by clicking the sequence names at the bottom of the page.
* Due to GPU memory and time constraints, we apply uniform sampling to the input frames when the total number of frames exceeds 40.
* This Gradio demo is built upon InstantSplat, which can be found at [https://huggingface.co/spaces/kairunwen/InstantSplat](https://huggingface.co/spaces/kairunwen/InstantSplat).

'''

block = gr.Blocks().queue()
with block:
    with gr.Row():
        with gr.Column(scale=1):
            # gr.Markdown('# ' + _TITLE)
            gr.Markdown(_DESCRIPTION)
    
    with gr.Row(variant='panel'):
        with gr.Tab("Input"):
            inputfiles = gr.File(file_count="multiple", label="images")
            input_path = gr.Textbox(visible=False, label="example_path")
            button_gen = gr.Button("RUN")

    with gr.Row(variant='panel'):
        with gr.Tab("Output"):
            with gr.Column(scale=2):
                with gr.Group():
                    output_model = gr.Model3D(
                        label="3D Dense Model under Gaussian Splats Formats, need more time to visualize",
                        interactive=False,
                        camera_position=[0.5, 0.5, 1],  # 稍微偏移一点，以便更好地查看模型
                    )
                    gr.Markdown(
                        """
                        <div class="model-description">
                           &nbsp;&nbsp;Use the left mouse button to rotate, the scroll wheel to zoom, and the right mouse button to move.
                        </div>
                        """
                    )    
                output_file = gr.File(label="ply")
            with gr.Column(scale=1):
                output_video = gr.Video(label="video")
                
    button_gen.click(process, inputs=[inputfiles], outputs=[output_video, output_file, output_model])
    
    # gr.Examples(
    #     examples=[
    #         "davis-dog",
    #         "sintel-market_2",
    #     ],
    #     inputs=[input_path],
    #     outputs=[output_video, output_file, output_model],
    #     fn=lambda x: process(inputfiles=None, input_path=x),
    #     cache_examples=True,
    #     label='Sparse-view Examples'
    # )
block.launch(server_name="0.0.0.0", share=False)