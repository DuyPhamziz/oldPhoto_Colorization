#!/usr/bin/env python3
import os
import shutil
import random

RAW_DIR = "dataset/raw"      # ảnh màu
GRAY_DIR = "dataset/gray"    # ảnh xám
OUT_DIR = "dataset/data_split"
TRAIN_RATIO = 0.8      # 80% train, 20% val
SEED = 42

random.seed(SEED)

os.makedirs(os.path.join(OUT_DIR, "train", "raw"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "train", "gray"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "val", "raw"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "val", "gray"), exist_ok=True)

files = sorted(os.listdir(RAW_DIR))   # 0.jpg, 1.jpg, ...
num_train = int(len(files) * TRAIN_RATIO)

train_files = random.sample(files, num_train)
val_files = [f for f in files if f not in train_files]

for f in train_files:
    shutil.copy(os.path.join(RAW_DIR, f), os.path.join(OUT_DIR, "train", "raw", f))
    shutil.copy(os.path.join(GRAY_DIR, f), os.path.join(OUT_DIR, "train", "gray", f))

for f in val_files:
    shutil.copy(os.path.join(RAW_DIR, f), os.path.join(OUT_DIR, "val", "raw", f))
    shutil.copy(os.path.join(GRAY_DIR, f), os.path.join(OUT_DIR, "val", "gray", f))

print("Done!")
print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")
