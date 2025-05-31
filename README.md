# Enhancing-Synthetic-Data-Realism-Using-Segmentation-Guided-ControlNet

# Enhancing-Synthetic-Image-Realism

This work focuses on improving the visual fidelity of synthetic images using controlled diffusion models, guided text prompts, and super-resolution techniques. Our approach aims to generate high-quality, realistic samples suitable for downstream tasks in computer vision and generative AI. Includes a modular pipeline with ControlNet integration and optimization strategies for faster processing.

## 🔧 Pipeline Overview
The following block diagram illustrates the core components of our pipeline:

![Pipeline Diagram](path/to/pipeline_diagram.png)

### Pipeline Stages:
1. **Input:** Synthetic image or segmentation map
2. **ControlNet:** Edge, Depth, and Segmentation map along with Text-prompt guided image refinement
3. **Super-Resolution:** Second step for upscaling and photorealistic enhancement
4. **Output:** High-resolution, realistic image

## 🚀 Real-ESRGAN for Super-Resolution
In the second stage of our pipeline, we apply Real-ESRGAN x4+ to further enhance the image quality and sharpness. This step significantly boosts visual fidelity, making the images more photorealistic and suitable for downstream tasks.

We use Real-ESRGAN x4+ by Qualcomm, a powerful model for general-purpose 4× super-resolution.
It is especially effective in recovering fine textures and reducing artifacts in generated images.
The integration is modular and can be toggled depending on application needs.

## 🧪 Results Demonstration
Below are sample outputs comparing the original synthetic images with their enhanced versions:

| Input | Enhanced Output |
|-------|-----------------|
| ![input1](path/to/input1.png) | ![output1](path/to/output1.png) |
| ![input2](path/to/input2.png) | ![output2](path/to/output2.png) |

Further, we have also uploaded high-resolution refined image results for each of the weather conditions from the VKITTI dataset for viewing purposes. [🔗 Click here to view more results](#).

## 🔗 ControlNet Models
Download or explore the ControlNet models used in this project:

- [ControlNet Github Page](https://github.com/yourgithubusername/controlnet)
- [ControlNet for Canny Edges](#)
- [ControlNet for Depth Maps](#)
- Custom Trained ControlNet (Coming Soon)

## About
No description, website, or topics provided.

