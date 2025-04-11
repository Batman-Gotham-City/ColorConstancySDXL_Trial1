# inference.py

import torch
from PIL import Image
from torchvision import transforms
import argparse
import os

from config import DEVICE, MODEL_SAVE_PATH
from model import load_pipeline
from utils import apply_canny_edge, extract_convnext_features

# Output image transform
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def run_inference(input_image_path, output_path="output_corrected.png"):
    # Load the pipeline
    pipe = load_pipeline()

    # Load your trained ControlNet weights
    print("🔁 Loading trained ControlNet weights...")
    pipe.controlnet.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    pipe.controlnet.eval()

    # Preprocess input image
    print("⚙️  Preprocessing input image...")
    edge_image = apply_canny_edge(input_image_path)
    convnext_features = extract_convnext_features(input_image_path)

    # Convert to tensors
    edge_tensor = to_tensor(edge_image).unsqueeze(0).to(DEVICE)
    convnext_features = convnext_features.unsqueeze(0).to(DEVICE)

    # Inference
    print("🧠 Running inference...")
    with torch.no_grad():
        result = pipe(
            prompt=[""],  # No prompt
            image=edge_tensor,
            added_cond_kwargs={"image_embeds": convnext_features},
            guidance_scale=1.0
        )

    # Save output image
    output_img = result.images[0]
    output_img.save(output_path)
    print(f"✅ Inference complete. Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using trained SDXL + ControlNet model")
    parser.add_argument("--input", required=True, help="Path to distorted input image")
    parser.add_argument("--output", default="output_corrected.png", help="Path to save corrected image")

    args = parser.parse_args()
    run_inference(args.input, args.output)
