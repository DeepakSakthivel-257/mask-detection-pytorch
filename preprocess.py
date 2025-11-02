"""
preprocess.py
Splits the dataset into train, validation, and test sets.
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIG ---
DATASET_DIR = os.environ.get('DATASET_DIR', 'dataset')  # path to the downloaded dataset
OUT_DIR = 'data/split'
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def make_dirs(base):
    for split in ['train', 'val', 'test']:
        for cls in ['with_mask', 'without_mask']:
            os.makedirs(os.path.join(base, split, cls), exist_ok=True)


def gather_files(dataset_dir):
    data = {}
    for cls in ['with_mask', 'without_mask']:
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_dir):
            files = [
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            data[cls] = files
        else:
            print(f"Warning: expected directory {cls_dir} not found")
            data[cls] = []
    return data


def split_and_copy(data, out_dir):
    make_dirs(out_dir)
    for cls, files in data.items():
        if not files:
            continue

        # Split train/val/test
        train_val, test = train_test_split(files, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        val_fraction = VAL_SIZE / (1 - TEST_SIZE)
        train, val = train_test_split(train_val, test_size=val_fraction, random_state=RANDOM_STATE)

        print(f"\nClass: {cls}")
        print(f"Total={len(files)} | Train={len(train)} | Val={len(val)} | Test={len(test)}")

        for f in tqdm(train, desc=f"Copying train {cls}"):
            shutil.copy(f, os.path.join(out_dir, 'train', cls, os.path.basename(f)))
        for f in tqdm(val, desc=f"Copying val {cls}"):
            shutil.copy(f, os.path.join(out_dir, 'val', cls, os.path.basename(f)))
        for f in tqdm(test, desc=f"Copying test {cls}"):
            shutil.copy(f, os.path.join(out_dir, 'test', cls, os.path.basename(f)))


if __name__ == "__main__":
    data = gather_files(DATASET_DIR)
    split_and_copy(data, OUT_DIR)
    print("\n✅ Dataset split complete. Data stored in:", OUT_DIR)
import os
import shutil
import random
from sklearn.model_selection import train_test_split

DATASET_DIR = 'dataset'
OUTPUT_DIR = 'data/split'
CATEGORIES = ['with_mask', 'without_mask']

def make_dirs():
    for split in ['train', 'val', 'test']:
        for category in CATEGORIES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, category), exist_ok=True)

def split_data():
    for category in CATEGORIES:
        src = os.path.join(DATASET_DIR, category)
        images = [f for f in os.listdir(src) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        train, temp = train_test_split(images, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        for split, data in zip(['train', 'val', 'test'], [train, val, test]):
            for img in data:
                shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_DIR, split, category, img))

if __name__ == "__main__":
    make_dirs()
    split_data()
    print("✅ Dataset successfully split into train, val, and test folders.")
