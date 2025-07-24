
import os
import logging
import argparse
import numpy as np
import torch
import imageio.v3 as imageio
import imageio as iio  # For saving videos
from einops import rearrange
import sys
sys.path.append(os.path.abspath("/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/"))
import os

sys.path.append('/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/annotator/')
sys.path.append(os.path.abspath("/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/"))
#sys.path.append('/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/annotator/')
import sys
import os
from annotator.canny import CannyDetector  # Import Canny edge detector
import gc


from realesrgan import RealESRGANer


torch.cuda.empty_cache()  # Clear the cache
gc.collect()  # Run garbage collection


from typing import Optional, Union
from safetensors.torch import load_file as safe_load_file
from diffusers import (
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
    ControlNetModel
)
from diffusers.loaders import AttnProcsLayers
from diffusers.utils import is_xformers_available
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import os
import numpy as np
import matplotlib.pyplot as plt  # For optional color mapping

# Import new detectors: Uniformer for segmentation, MiDaS for depth detection
from annotator.uniformer import UniformerDetector  # Updated to Uniformer for segmentation
from annotator.midas import MidasDetector  # Updated for depth detection with MiDaS
from annotator.util import resize_image, HWC3
#from realesrgan import RealESRGAN
from realesrgan import RealESRGANer

logger = get_logger(__name__, log_level="INFO")

# Initialize the new annotators
apply_uniformer = UniformerDetector()  # Uniformer for segmentation
apply_midas = MidasDetector()  # MiDaS for depth detection
apply_canny = CannyDetector()
import os
import inspect
import logging
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import imageio.v3 as imageio
import imageio as iio  # For saving videos
from tqdm.auto import tqdm
from einops import rearrange
from torchvision.utils import make_grid
from typing import Optional, Literal, Union
from omegaconf import OmegaConf
from safetensors.torch import load_file as safe_load_file
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

import wandb
import diffusers
import transformers
from diffusers import (
    ModelMixin,
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    UNet2DConditionModel,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline
)
from diffusers.models.attention_processor import (
    LoRAAttnProcessor, 
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor, 
    AttnProcessor
)

from diffusers.utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from dataset import LoRADataset
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector  # Updated to Uniformer for segmentation
from annotator.midas import MidasDetector
apply_midas = MidasDetector()
default_neg_prompt = "abstract, cartoon, low resolution, unrealistic, fantasy elements, distorted perspective, blurry, overexposed, night scenes, extreme weather, futuristic vehicles, animals, flying objects, underwater, low-quality textures, unnatural colors, alien landscapes, incorrect proportions, fantasy characters, sci-fi buildings, otherworldly environments,long exposure, unnatural lighting, blur, distorted vehciles,blurry cars, out-of-focus vehicles, distorted vehicles, low-detail cars, low-resolution cars, incorrect road signs, abstract signs, low-detail signs, unrealistic road signs, distorted text on signs, cartoon grass, fake grass, unrealistic vegetation, overly bright grass, low-texture grass, artificial grass, 2D grass, heavy shadows, overly dark shadows, unnatural lighting, harsh shadows, extreme shadow contrast, underexposed, (semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, distorted vehicle, unnatural lightings), (deformed, distorted, disfigured:1.3, distorted vehciles), cartoon, low resolution, low quality, unrealistic, distorted perspective, abstract, night scenes, extreme weather, blurry, overexposed, futuristic vehicles, distorted vehicle,blurry cars, flying objects, underwater scenes, fantasy elements, alien landscapes, unrealistic vegetation, unrealistic road signs, scifi buildings, overly bright colors, artistic effects, long exposure, unnatural lighting, blur, distorted vehciles, half vehicle body, Do not change colors, maintain original colors, avoid color distortion, keep the color scheme of the input image."

#'stabilityai/stable-diffusion-2',#'dreamlike-art/dreamlike-photoreal-2.0',#'digiplay/AbsoluteReality_v1.8.1',#'SG161222/Realistic_Vision_V2.0', #'dreamlike-art/dreamlike-photoreal-2.0',#,#'runwayml/stable-diffusion-v1-5',#"stabilityai/stable-diffusion-2-1",


#'stabilityai/stable-diffusion-2',#'dreamlike-art/dreamlike-photoreal-2.0',#'digiplay/AbsoluteReality_v1.8.1',#'SG161222/Realistic_Vision_V2.0', #'dreamlike-art/dreamlike-photoreal-2.0',#,#'runwayml/stable-diffusion-v1-5',#"stabilityai/stable-diffusion-2-1",
#"/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/stable-diffusion-v1-5/my_experiment/pytorch_lora_weights.safetensors"
#"/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/stable-diffusion-2/my_experiment/pytorch_lora_weights.safetensors"
#"/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/Realistic_Vision_V2.0/my_experiment/pytorch_lora_weights.safetensors"
#"/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/dreamlike-photoreal-2.0_Model/my_experiment/pytorch_lora_weights.safetensors"
#"/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/AbsoluteReality_v1.8.1/my_experiment/pytorch_lora_weights.safetensors"
##"/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/output/my_experiment/pytorch_lora_weights.safetensors",
def parse_args():
    parser = argparse.ArgumentParser(description="Sampling")
    parser.add_argument("--pretrained_unet_lora_path", type=str, default="/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/AbsoluteReality_v1.8.1/my_experiment/pytorch_lora_weights.safetensors", help="The path of pretrained LoRA's state dict.")
    #parser.add_argument("--pretrained_controlnet_depth_path", type=str, default='lllyasviel/sd-controlnet-depth',help="The path or version of pretrained Depth ControlNet.")
    parser.add_argument("--pretrained_controlnet_segmentation_path", type=str, default='lllyasviel/sd-controlnet-seg',#'thibaud/controlnet-sd21-ade20k-diffusers',#'lllyasviel/sd-controlnet-seg',# ,#,
                        help="The path or version of pretrained Segmentation ControlNet.")
    parser.add_argument("--pretrained_controlnet_canny_path", type=str, default='lllyasviel/sd-controlnet-canny',#'thibaud/controlnet-sd21-canny-diffusers',#'lllyasviel/sd-controlnet-canny', #, #,
                        help="The path or version of pretrained Canny ControlNet.")   #benjamin-paine/stable-diffusion-v1-5, dreamlike-art/dreamlike-photoreal-2.0
    parser.add_argument("--pretrained_model_path", type=str, default='songkey/absolutereality_v181',#'stabilityai/stable-diffusion-2',#'dreamlike-art/dreamlike-photoreal-2.0',#'digiplay/AbsoluteReality_v1.8.1',#'SG161222/Realistic_Vision_V2.0', #'dreamlike-art/dreamlike-photoreal-2.0',#,#'runwayml/stable-diffusion-v1-5',#"stabilityai/stable-diffusion-2-1",
                        help="The path or version of pretrained Stable Diffusion.")
    parser.add_argument("--hint_image", type=str, default="/data/iqra/datasets/vkitti_2.0.3_rgb/Scene20/sunset/frames/rgb/Camera_1/images/rgb_00836.jpg",
                        help="The path of source image. If None, the script will perform random generation.")
    parser.add_argument("--prompt", type=str, default="Improve the image's realism while preserving every element exactly as it is. A realistic, high-resolution kitti_scene of a motorway with moving vehicles, road signs, grass, trees, viewed from the front of a moving vehicle. Make the image more realistic without changing any elements, colours, or introducing new objects.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                        help="The number of samples.")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="The inference steps for diffusion model.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Whether to use fp16 in inference.")
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                        help="The guidance scale for classifier-free-guidance.")
    parser.add_argument("--negative_prompt", type=str, default=default_neg_prompt,
                        help="The negative prompt for classifier-free-guidance.")
    parser.add_argument("--output_path", type=str, default='/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/output_realistic_image.png',
                        help="The output jpg/png path to save sampling results.")
    args = parser.parse_args()
    
    return args


def init_lora_attn(model, lora_rank=14):
    lora_attn_procs = {}
    for name in model.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]
        procs = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
        lora_attn_procs[name] = procs
    return lora_attn_procs

import cv2  # OpenCV to draw text on the image
import matplotlib.pyplot as plt



def calculate_dynamic_canny_thresholds(image, sigma=0.25):
    # Convert the image to grayscale if it's RGB
    if image.ndim == 3:
        grayscale_image = np.mean(image, axis=-1)
    else:
        grayscale_image = image
    
    # Compute the median of the pixel intensities
    median = np.median(grayscale_image)
    
    # Apply a formula to calculate the thresholds
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    
    return lower, upper
# Main function for image refinement using Depth and Segmentation
def main(
    pretrained_unet_lora_path: str,
    #pretrained_controlnet_depth_path: Optional[str],
    pretrained_controlnet_canny_path: Optional[str],
    pretrained_controlnet_segmentation_path: Optional[str],
    pretrained_model_path: str,
    hint_image: np.ndarray = None,
    prompt: Union[str, list[str]] = None,
    num_images_per_prompt: int = 1,
    num_inference_steps: int = 30,
    depth_image_reso: int = 786,
    seed: int = 0,
    fp16: bool = True,
    segmentation_map: np.ndarray = None,  # New parameter
    enable_xformers_memory_efficient_attention: bool = True,
    guidance_scale: float = 9.0,
    negative_prompt: str = default_neg_prompt,
    output_path: str = '/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/output_realistic_image.png',
    output_depth_path: str = '/path/to/save_depth_map.png',   # Add path for depth map
    output_segmentation_path: str = '/path/to/save_segmentation_map.png'  # Add path for segmentation map
):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = torch.device('cuda00' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    


    # Load the hint image as a NumPy array if it's a string (path)
    if isinstance(hint_image, str):
        hint_image = imageio.imread(hint_image)
        
    # Step 1: Generate Canny Edge Map
    low_threshold, high_threshold = calculate_dynamic_canny_thresholds(hint_image)
    hint_image_canny = resize_image(HWC3(hint_image), depth_image_reso)
    canny_edges = apply_canny(hint_image_canny, low_threshold, high_threshold)

    # Apply Depth 
    #hint_image = 'RealESRGANer.upscale(hint_image, scale_factor=2)
    #hint_image_depth = resize_image(HWC3(hint_image), depth_image_reso)
    #depth_map = apply_midas(hint_image_depth)  # Apply MiDaS depth detection
    #depth_map = depth_map[0] if isinstance(depth_map, tuple) else depth_map
    
    # Save the depth map
    #imageio.imwrite(output_depth_path, (depth_map * 255).astype(np.uint8))  # Save depth map as an image
    
    #apply segmentation to generate control images

    # Use the ground truth segmentation map
    if segmentation_map is not None:
        segmentation_map = resize_image(HWC3(segmentation_map), depth_image_reso)
        #imageio.imwrite(output_segmentation_path, segmentation_map)
    else:
        # Fallback to Uniformer segmentation
        hint_image_segmentation = resize_image(HWC3(hint_image), depth_image_reso)
        segmentation_map = apply_uniformer(hint_image_segmentation)
        segmentation_map = HWC3(segmentation_map)
        #imageio.imwrite(output_segmentation_path, segmentation_map)
    
    

    #imageio.imwrite(output_segmentation_path, (segmentation_map * 255).astype(np.uint8))  # Save segmentation map
    weight_dtype = torch.float16 if fp16 else torch.float32
    


    # Prepare control images
    #depth_control_image = torch.from_numpy(HWC3(depth_map).copy()).float() / 255.0
    segmentation_control_image = torch.from_numpy(HWC3(segmentation_map).copy()).float() / 255.0
    
    # Step 1: Generate Canny Edge Map
    low_threshold, high_threshold = calculate_dynamic_canny_thresholds(hint_image)
    hint_image_canny = resize_image(HWC3(hint_image), depth_image_reso)
    canny_edges = apply_canny(hint_image_canny, low_threshold, high_threshold)
    
    canny_control_image = torch.from_numpy(HWC3(canny_edges).copy()).float() / 255.0

    control_images = [canny_control_image.permute(2, 0, 1).unsqueeze(0).to(device, dtype=weight_dtype),
    segmentation_control_image.permute(2, 0, 1).unsqueeze(0).to(device, dtype=weight_dtype)]
    try:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet",weights_only=True).to(device)
        unet.load_attn_procs(pretrained_unet_lora_path)
    except Exception as e:
        print(f"Error loading UNet model: {e}")

    

    # Load UNet and set up LoRA and ControlNet
    #unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet",weights_only=True).to(device)
    #unet.load_attn_procs(pretrained_unet_lora_path)
    lora_attn_procs = init_lora_attn(unet)
    unet.set_attn_processor(lora_attn_procs)
    #vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema",weights_only=True).to(device)
    # Convert the VAE to FP16 if fp16 is enabled
    if fp16:
        vae = vae.half()  # Cast the entire VAE model to FP16

    
    #controlnet_depth = ControlNetModel.from_pretrained(pretrained_controlnet_depth_path).to(device)
    controlnet_segmentation = ControlNetModel.from_pretrained(pretrained_controlnet_segmentation_path,weights_only=True).to(device)
    controlnet_canny = ControlNetModel.from_pretrained(pretrained_controlnet_canny_path,weights_only=True).to(device)
    # Enable xformers attention if available
    if enable_xformers_memory_efficient_attention and is_xformers_available():
        controlnet_canny.enable_xformers_memory_efficient_attention()
        controlnet_segmentation.enable_xformers_memory_efficient_attention()



    controlnets = [controlnet_canny.to(dtype=weight_dtype), controlnet_segmentation.to(dtype=weight_dtype)]

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_path,
        unet=unet.to(dtype=weight_dtype),
        controlnet=controlnets,
        vae=vae,  # Add the VAE here
        torch_dtype=weight_dtype
    ).to(device)

    if isinstance(prompt, str):
        prompt = [prompt] * len(control_images)
        
    controlnet_weights = [0.7, 1.3]

    # Generate refined image
    images = pipeline(
        prompt=prompt,
        image=control_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=[negative_prompt] * len(prompt),
        controlnet_conditioning_scale=controlnet_weights
    ).images

    # Convert the image from PIL Image to NumPy array
    image_np = np.array(images[0])

    # Check if the values are already in the range [0, 255]
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)  # Only scale if values are in [0, 1]

    # Save the final image
    imageio.imwrite(output_path, image_np)


def process_folder_of_images(
    folder_path: str,
    output_folder: str,
    segmentation_folder_path: str,
    pretrained_unet_lora_path: str,
    #pretrained_controlnet_depth_path: str,
    pretrained_controlnet_canny_path: str,
    pretrained_controlnet_segmentation_path: str,
    pretrained_model_path: str,
    prompt: str,
    num_images_per_prompt: int,
    num_inference_steps: int,
    depth_image_reso: int,
    seed: int,
    fp16: bool,
    guidance_scale: float,
    negative_prompt: str,
    video_fps: int = 5  # Frames per second for the video
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    segmentation_files = sorted([f for f in os.listdir(segmentation_folder_path) if f.endswith('.png')])
    # Ensure both folders have the same number of files
    if len(image_files) != len(segmentation_files):
        print("Error: The number of images and segmentation maps do not match.")
        return

    refined_images, depth_images, segmentation_images = [], [], []

    # Loop through the sorted image and segmentation files together
    for image_file, segmentation_file in zip(image_files, segmentation_files):
        image_path = os.path.join(folder_path, image_file)
        segmentation_path = os.path.join(segmentation_folder_path, segmentation_file)

        # Ensure that the segmentation map exists
        if not os.path.exists(segmentation_path):
            print(f"Segmentation map not found for {image_file}. Skipping.")
            continue

        output_image_path = os.path.join(output_folder, f"refined_{image_file}")
        #output_depth_path = os.path.join(output_folder, f"depth_{image_file}")
        output_segmentation_path = os.path.join(output_folder, f"segmentation_{image_file}")

        # Load ground truth segmentation map
        try:
            segmentation_map = imageio.imread(segmentation_path)
        except FileNotFoundError as e:
            print(f"Error loading segmentation map {segmentation_path}: {e}")
            continue

        # Call your main processing function (e.g., main) with the correct inputs
        main(
            pretrained_unet_lora_path=pretrained_unet_lora_path,
            #pretrained_controlnet_depth_path=pretrained_controlnet_depth_path,
            pretrained_controlnet_canny_path=pretrained_controlnet_canny_path,
            pretrained_controlnet_segmentation_path=pretrained_controlnet_segmentation_path,
            pretrained_model_path=pretrained_model_path,
            hint_image=image_path,
            segmentation_map=segmentation_map,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            depth_image_reso=depth_image_reso,
            seed=seed,
            fp16=fp16,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            output_path=output_image_path,
            #output_depth_path=output_depth_path,
            output_segmentation_path=output_segmentation_path
        )

        refined_images.append(output_image_path)
        #depth_images.append(output_depth_path)
        segmentation_images.append(output_segmentation_path)

    generate_video(refined_images, os.path.join(output_folder, 'refined_images_video.mp4'), video_fps)
    #generate_video(depth_images, os.path.join(output_folder, 'depth_images_video.mp4'), video_fps)
    generate_video(segmentation_images, os.path.join(output_folder, 'segmentation_images_video.mp4'), video_fps)


def generate_video(image_paths, video_output_path, fps):
    with iio.get_writer(video_output_path, fps=fps) as video_writer:
        for image_path in image_paths:
            img = imageio.imread(image_path)
            video_writer.append_data(img)
    print(f"Video created at: {video_output_path}")

#/data/iqra/datasets/mixed_vkitti_dataset_image/
#"/data/iqra/datasets/vkitti_2.0.3_rgb/Scene20/overcast/frames/rgb/Camera_1/"

if __name__ == '__main__':
    args = parse_args()
    folder_path = "/data/iqra/datasets/mixed_vkitti_dataset_image/"  # Set your input folder path here
    segmentation_folder_path = "/data/iqra/datasets/mixed_vkitti_dataset_semantics/refined_mixed_vkitti_dataset_semantics/"
    output_folder = "/users/iqra_n/diffusion/Kitti_Lora/ST_LORA/originalGT/AbsoluteReality_v1.8.1/"
    process_folder_of_images(
        folder_path=folder_path,
        output_folder=output_folder,
        segmentation_folder_path=segmentation_folder_path,
        pretrained_unet_lora_path=args.pretrained_unet_lora_path,
        #pretrained_controlnet_depth_path=args.pretrained_controlnet_depth_path,
        pretrained_controlnet_canny_path=args.pretrained_controlnet_canny_path,
        pretrained_controlnet_segmentation_path=args.pretrained_controlnet_segmentation_path,
        pretrained_model_path=args.pretrained_model_path,
        prompt=args.prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        depth_image_reso=512,
        seed=args.seed,
        fp16=args.fp16,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt
    )
