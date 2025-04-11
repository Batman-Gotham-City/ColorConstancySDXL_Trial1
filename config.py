# config.py

import torch

# Paths
RAW_GT_DIR = "data/Cube_ground_truth_images"
RAW_INPUT_DIR = "data/Cube_input_images"
PROCESSED_GT_DIR = "data/Processed_Cube_ground_truth_images"
PROCESSED_INPUT_DIR = "data/Processed_Cube_input_images"

PAIR_CSV = "pairs.csv"
TRAIN_CSV = "train_pairs.csv"
VAL_CSV = "val_pairs.csv"
TEST_CSV = "test_pairs.csv"

MODEL_SAVE_PATH = "controlnet_SDXL_trained.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image sizes
TARGET_SIZE = (512, 512)
CONVNEXT_SIZE = (224, 224)

# Training
BATCH_SIZE = 1
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
