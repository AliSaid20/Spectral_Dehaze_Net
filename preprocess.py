import os
import torch
from dataset import AudioDataset

# ----------------------------
# CONFIG
# ----------------------------
NOISE_DIR = "noise path"
NOISY_DIR = "noisy path"
CACHE_DIR = "cache path"

os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------
# CREATE ORIGINAL DATASET
# ----------------------------
dataset = AudioDataset(
    noise_dir=NOISE_DIR,
    noisy_dir=NOISY_DIR,
    sample_rate=16000,
    segment_seconds=5,
)

print("Total files:", len(dataset))

# ----------------------------
# LOOP & SAVE FEATURES
# ----------------------------
for idx in range(len(dataset)):
    noisy_mag, noise_mag, illum = dataset[idx]

    id_ = dataset.ids[idx]  # "0001", "0002", etc
    out_path = os.path.join(CACHE_DIR, f"{id_}.pt")

    torch.save({
        "noisy_mag": noisy_mag,
        "noise_mag": noise_mag,
        "illum": illum,
    }, out_path)

    print(f"[{idx+1}/{len(dataset)}] Saved:", out_path)

print("\n✔ DONE — Cached dataset written to:", CACHE_DIR)
