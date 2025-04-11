# model.py

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    DDPMScheduler
)

from config import DEVICE

def load_pipeline():
    # Load ControlNet (Canny)
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )

    # Load Stable Diffusion XL Base Model
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    # Configure pipeline for training
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    pipe = pipe.to(DEVICE)

    return pipe