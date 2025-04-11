# data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from config import CONVNEXT_SIZE, DEVICE
from utils import apply_canny_edge, extract_convnext_features

# Image transformation (for Stable Diffusion input + GT)
image_transform = transforms.Compose([
    transforms.Resize(CONVNEXT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet stats
                         [0.229, 0.224, 0.225])
])


class ImagePairsDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_img_path = row["Input Image"]
        gt_img_path = row["Ground Truth Image"]

        # Apply Canny edge
        edge_img = apply_canny_edge(input_img_path)
        edge_tensor = image_transform(edge_img)

        # Extract ConvNeXt features
        features = extract_convnext_features(input_img_path)

        # Load GT image
        gt_img = Image.open(gt_img_path).convert("RGB")
        gt_tensor = image_transform(gt_img)

        return edge_tensor, features, gt_tensor


def get_dataloader(csv_path, batch_size=1, shuffle=True):
    dataset = ImagePairsDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Example test
    loader = get_dataloader("train_pairs.csv", batch_size=2)
    for edges, feats, gts in loader:
        print("Batch shapes:", edges.shape, feats.shape, gts.shape)
        break
