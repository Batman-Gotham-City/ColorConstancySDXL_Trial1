# utils.py

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

from config import DEVICE, CONVNEXT_SIZE

# Pre-load ConvNeXt once globally
_convnext = models.convnext_tiny(pretrained=True).to(DEVICE)
_convnext.eval()

# Transformation for ConvNeXt input
_convnext_transform = transforms.Compose([
    transforms.Resize(CONVNEXT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def apply_canny_edge(image_path, size=(512, 512), low_thresh=100, high_thresh=200):
    """
    Apply Canny edge detection and return a PIL image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    edges = cv2.Canny(img, low_thresh, high_thresh)
    edge_rgb = Image.fromarray(edges).convert("RGB")
    return edge_rgb


def extract_convnext_features(image_path):
    """
    Extract global average pooled features from ConvNeXt Tiny.
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = _convnext_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = _convnext.features(img_tensor)
        pooled = features.mean(dim=[2, 3])  # Global average pooling

    return pooled.squeeze(0)
