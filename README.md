# Enhancing Synthetic Data Realism for Autonomous Vehicles Using Segmentation-Guided ControlNet

This project focuses on enhancing the realism of synthetic datasets for autonomous vehicle applications. We integrate a segmentation-guided ControlNet with Stable Diffusion fine-tuned on the KITTI dataset to bridge the domain gap between synthetic and real-world data, improving the realism and usability of synthetic data for downstream tasks like object detection and depth estimation.

## 🔧 Pipeline Overview

The following block diagram illustrates the core components of our pipeline:

![Pipeline Diagram](path/to/pipeline_diagram.png)

### Pipeline Stages:
1. **Input:** Synthetic image or segmentation map from VKITTI dataset
2. **ControlNet:** Segmentation maps, Canny edges, and text-prompt guided image refinement
3. **Super-Resolution:** Upscaling and photorealistic enhancement
4. **Output:** High-resolution, realistic image

## 🚀 LoRA-Fine-Tuned Stable Diffusion for Data Refinement
In the second stage of our pipeline, we apply LoRA to fine-tune a Stable Diffusion model on the real-world KITTI dataset. This enables the model to adapt to the target domain with fewer images and less computational power while maintaining high-quality image generation.

The model generates realistic, high-resolution synthetic images that preserve structural consistency, making them suitable for downstream tasks such as object detection and depth estimation.

## 🧪 Results Demonstration

Below are sample outputs comparing the original synthetic images from VKITTI with their enhanced versions using segmentation-guided ControlNet:

| Input | Enhanced Output |
|-------|-----------------|
| ![input1](path/to/input1.png) | ![output1](path/to/output1.png) |
| ![input2](path/to/input2.png) | ![output2](path/to/output2.png) |

Additionally, we have uploaded high-resolution refined images for each weather condition from the VKITTI dataset for viewing purposes. [🔗 Click here to view more results](#).

## 🔗 ControlNet Models

Download or explore the ControlNet models used in this project:

- [ControlNet Github Page](https://github.com/yourgithubusername/controlnet)
- [ControlNet for Canny Edges](#)
- [ControlNet for Depth Maps](#)
- Custom Trained ControlNet (Coming Soon)

## 📝 Key Contributions:
- Fine-tuned Stable Diffusion on the KITTI dataset with LoRA to generate high-quality images.
- Used segmentation-guided ControlNet with ground-truth segmentation maps from VKITTI and Canny edges to generate semantically accurate images.
- Evaluated our method on object detection (YOLOv8) and depth estimation tasks, showing significant improvements in model accuracy on real-world KITTI data.

## 🧑‍🔬 Downstream Task Performance

### Object Detection Results:
We employed the YOLOv8 model for object detection and demonstrated improved performance on the refined VKITTI dataset compared to the original dataset. The refined dataset improved the mAP50 from 0.434 to 0.643, validating its effectiveness.

| Dataset         | mAP50  |
|-----------------|--------|
| Original VKITTI | 0.434  |
| Refined VKITTI  | 0.643  |

### Depth Estimation Results:
Using a U-Net-based model, the refined VKITTI dataset led to a decrease in RMSE from 0.3121 to 0.2226, demonstrating enhanced accuracy in depth estimation.

| Dataset         | RMSE   |
|-----------------|--------|
| Original VKITTI | 0.3121 |
| Refined VKITTI  | 0.2226 |

## About
No description, website, or topics provided.

