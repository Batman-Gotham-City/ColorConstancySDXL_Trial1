# data_preparation.py

from pathlib import Path
import shutil
import random
import csv
import json
import cv2
import numpy as np
import pandas as pd

from config import RAW_GT_DIR, RAW_INPUT_DIR, PROCESSED_GT_DIR, PROCESSED_INPUT_DIR, PAIR_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV, TARGET_SIZE

# Convert config paths to Path objects
RAW_GT_DIR = Path(RAW_GT_DIR)
RAW_INPUT_DIR = Path(RAW_INPUT_DIR)
PROCESSED_GT_DIR = Path(PROCESSED_GT_DIR)
PROCESSED_INPUT_DIR = Path(PROCESSED_INPUT_DIR)
PAIR_CSV = Path(PAIR_CSV)
TRAIN_CSV = Path(TRAIN_CSV)
VAL_CSV = Path(VAL_CSV)
TEST_CSV = Path(TEST_CSV)


def select_and_copy_images(num_samples=100):
    gt_files = [f for f in RAW_GT_DIR.glob("*.JPG")]
    selected_gt = random.sample(gt_files, num_samples)

    selected_input = []
    for f in RAW_INPUT_DIR.glob("*.JPG"):
        base = f.stem.split("_")[0] + ".JPG"
        if (RAW_GT_DIR / base).name in [g.name for g in selected_gt]:
            selected_input.append(f)

    PROCESSED_GT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in selected_gt:
        shutil.copy(f, PROCESSED_GT_DIR / f.name)

    for f in selected_input:
        shutil.copy(f, PROCESSED_INPUT_DIR / f.name)

    print(f"✅ Copied {len(selected_gt)} ground truth and {len(selected_input)} input images.")


def resize_images(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in input_dir.glob("*.JPG"):
        img = cv2.imread(str(f))
        if img is not None:
            img = cv2.resize(img, TARGET_SIZE)
            img = img.astype(np.float32) / 255.0
            cv2.imwrite(str(output_dir / f.name), (img * 255).astype(np.uint8))
    print(f"✅ Resized and normalized images in {input_dir}")


def create_csv_json_mapping():
    gt_map = {f.stem: f for f in PROCESSED_GT_DIR.glob("*.JPG")}
    mapping_list = []
    mapping_dict = {}

    for input_img in PROCESSED_INPUT_DIR.glob("*.JPG"):
        base = input_img.stem.split("_")[0]
        if base in gt_map:
            gt_path = PROCESSED_GT_DIR / gt_map[base].name
            mapping_list.append([str(input_img), str(gt_path)])
            mapping_dict.setdefault(str(gt_path), []).append(str(input_img))

    with PAIR_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Input Image", "Ground Truth Image"])
        writer.writerows(mapping_list)

    with Path("pairs.json").open("w") as f:
        json.dump(mapping_dict, f, indent=4)

    print("✅ Mapping saved to pairs.csv and pairs.json")


def split_dataset():
    df = pd.read_csv(PAIR_CSV)
    gt_imgs = df["Ground Truth Image"].unique().tolist()
    random.shuffle(gt_imgs)

    n = len(gt_imgs)
    train, val, test = gt_imgs[:int(0.8*n)], gt_imgs[int(0.8*n):int(0.9*n)], gt_imgs[int(0.9*n):]

    def filter_df(imgs):
        return df[df["Ground Truth Image"].isin(imgs)]

    filter_df(train).to_csv(TRAIN_CSV, index=False)
    filter_df(val).to_csv(VAL_CSV, index=False)
    filter_df(test).to_csv(TEST_CSV, index=False)

    print("✅ Dataset split into train, val, and test CSVs.")


if __name__ == "__main__":
    select_and_copy_images(num_samples=100)
    resize_images(PROCESSED_GT_DIR, PROCESSED_GT_DIR)
    resize_images(PROCESSED_INPUT_DIR, PROCESSED_INPUT_DIR)
    create_csv_json_mapping()
    split_dataset()
