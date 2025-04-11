# train.py

import torch
from torch.optim import AdamW
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity

from config import DEVICE, TRAIN_CSV, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_SAVE_PATH
from data_loader import get_dataloader
from model import load_pipeline

# Optional LPIPS loss
USE_PERCEPTUAL_LOSS = False

# Load training pipeline and model
pipe = load_pipeline()
pipe.controlnet.train()  # ✅ Train only ControlNet

# Loss functions
ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(DEVICE)

# Optimizer
optimizer = AdamW(pipe.controlnet.parameters(), lr=LEARNING_RATE)

# Transform for output images from pipe
to_tensor = transforms.ToTensor()

# DataLoader
train_loader = get_dataloader(TRAIN_CSV, batch_size=BATCH_SIZE)


def train():
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        max_batches = 50  # for quicker dev/test

        for batch_idx, (edges, features, gts) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            edges, features, gts = edges.to(DEVICE), features.to(DEVICE), gts.to(DEVICE)

            prompts = [""] * edges.shape[0]  # Blank prompt

            # Forward pass
            output = pipe(
                prompt=prompts,
                image=edges,
                added_cond_kwargs={"image_embeds": features},
                guidance_scale=1.0
            )

            pred_imgs = torch.stack([to_tensor(img).to(DEVICE) for img in output.images])

            # Compute loss
            if USE_PERCEPTUAL_LOSS:
                loss = lpips_loss(pred_imgs, gts)
            else:
                loss = 1 - ssim_loss(pred_imgs, gts)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1} | Batch {batch_idx}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / max_batches
        print(f"✅ Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

    # Save model
    torch.save(pipe.controlnet.state_dict(), MODEL_SAVE_PATH)
    print(f"🎉 Training complete! ControlNet saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
