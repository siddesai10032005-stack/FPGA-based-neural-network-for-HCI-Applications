import pandas as pd
import numpy as np
import os

INPUT = "data/processed/final_gaze_labels.csv"

TRAIN_OUT = "data/processed/train.csv"
VAL_OUT = "data/processed/val.csv"
TEST_OUT = "data/processed/test.csv"

df = pd.read_csv(INPUT)

# ---------------------------------------------------------
# Get unique head-pose folders
# ---------------------------------------------------------

hp_folders = sorted(df["hp_folder"].unique())

print("Total HP folders:", len(hp_folders))

# Reproducible shuffle
rng = np.random.default_rng(42)
rng.shuffle(hp_folders)

# ---------------------------------------------------------
# 80 / 10 / 10 split
# ---------------------------------------------------------

n = len(hp_folders)

n_train = int(0.80 * n)
n_val = int(0.10 * n)

train_hp = hp_folders[:n_train]
val_hp = hp_folders[n_train:n_train + n_val]
test_hp = hp_folders[n_train + n_val:]

train = df[df["hp_folder"].isin(train_hp)].copy()
val = df[df["hp_folder"].isin(val_hp)].copy()
test = df[df["hp_folder"].isin(test_hp)].copy()

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

train.to_csv(TRAIN_OUT, index=False)
val.to_csv(VAL_OUT, index=False)
test.to_csv(TEST_OUT, index=False)

print("\n==============================")
print("DATASET SPLIT")
print("==============================")

print(
    f"Train: {len(train)} samples "
    f"({len(train_hp)} HP folders)"
)

print(
    f"Validation: {len(val)} samples "
    f"({len(val_hp)} HP folders)"
)

print(
    f"Test: {len(test)} samples "
    f"({len(test_hp)} HP folders)"
)

print("\nTraining HP folders:")
print(train_hp)

print("\nValidation HP folders:")
print(val_hp)

print("\nTest HP folders:")
print(test_hp)

# ---------------------------------------------------------
# Check for leakage
# ---------------------------------------------------------

train_set = set(train_hp)
val_set = set(val_hp)
test_set = set(test_hp)

assert not train_set & val_set
assert not train_set & test_set
assert not val_set & test_set

print("\nNo HP-folder overlap detected.")
print("==============================")
