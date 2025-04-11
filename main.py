# main.py

import argparse

def main():
    parser = argparse.ArgumentParser(description="Color Constancy Correction Pipeline")
    parser.add_argument('--prepare', action='store_true', help='Run data preparation')
    parser.add_argument('--train', action='store_true', help='Train the ControlNet model')

    args = parser.parse_args()

    if args.prepare:
        from data_preparation import (
            select_and_copy_images,
            resize_images,
            create_csv_json_mapping,
            split_dataset,
            PROCESSED_GT_DIR,
            PROCESSED_INPUT_DIR
        )

        print("📁 Running data preparation pipeline...")
        select_and_copy_images(num_samples=100)
        resize_images(PROCESSED_GT_DIR, PROCESSED_GT_DIR)
        resize_images(PROCESSED_INPUT_DIR, PROCESSED_INPUT_DIR)
        create_csv_json_mapping()
        split_dataset()

    if args.train:
        from train import train
        print("🚀 Starting training...")
        train()


if __name__ == "__main__":
    main()
