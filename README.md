# Enhancing Synthetic Data Realism for Autonomous Vehicles Using Segmentation-Guided ControlNet

**Iqra Nosheen, Cathy Ennis,Michael G. Madden.**


This project focuses on enhancing the realism of synthetic datasets for autonomous vehicle applications. We integrate a segmentation-guided ControlNet with Stable Diffusion fine-tuned on the KITTI dataset to bridge the domain gap between synthetic and real-world data, improving the realism and usability of synthetic data for downstream tasks like object detection and depth estimation.

## 🔧 Pipeline Overview

The following block diagram illustrates the core components of our pipeline:

![Pipeline Diagram](Assests/ECML%20workflow.png)

1. **Input:** Real world Kitti dataset, Text-guided prommpts, Pre-trained stable diffusion model
2. **Finetuning Phase:** 
   - Pre-trained **Dreamlike Photoreal Model 2.0**
   - Text-guided prompts for photorealistic generation
   - **Fine-tuned Stable Diffusion Model** for realistic image generation
3. **Data Refinement Phase:**
   - **Synthetic VKITTI dataset, and Fine-tuned Stable Diffusion Model**
   - Conditions: **Segmentation maps**, **Canny edges**, **text-guided prompts**
   - Refined images
4. **Validation Phase:**
   - **Downstream ML tasks**: Object Detection (YOLO v8), Depth Estimation
   - **Refined Data** for validation and task performance

## 🛠️ Training Methodology: Fine-Tuning with LoRA and ControlNet
For training, we followed the approach outlined in this [GitHub repository](https://github.com/lizhiqi49/I2I-Stable-Diffusion-Finetune-LoRA-ControlNet.git), which focuses on fine-tuning Stable Diffusion using LoRA (Low-Rank Adaptation) and ControlNet for few-shot image transfer.
### 1. For Fine-tuning
#### Install Dependencies: Make sure your Python environment is activated. Then install the required packages, set the dataset path and configure hyper-parameters in the my_experiment.yaml file in the configs folder

```bash
pip install -r requirements.txt
python train.py
'''
### 🧪 Results Demonstration

Below are sample outputs comparing the original synthetic images from VKITTI with their enhanced versions using segmentation-guided ControlNet:

![input1](Assests/generated%20images.jpg) 

Additionally, we have uploaded the fine-tuned Stable Diffusion model on the real-world KITTI dataset for public access. [Finetuned stable diffusion model](Assests/pytorch_lora_weights.safetensors) 
Dataset: [VKITTI Dataset Access](https://nuigalwayie-my.sharepoint.com/:u:/g/personal/i_nosheen1_universityofgalway_ie/EQSBpXC6Ho9Lhkm4kr9NJ7kBVEgAtFZ_os08pOz46yTp8A?e=xywJ1T)

## 🔗 ControlNet Models

Download or explore the ControlNet models used in this project:

- [ControlNet Github Page](https://github.com/lllyasviel/ControlNet)
- [ControlNet For Canny Edges](https://github.com/lllyasviel/ControlNet)
- [ControlNet for Segmentation Maps](https://huggingface.co/lllyasviel/sd-controlnet-seg)

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

