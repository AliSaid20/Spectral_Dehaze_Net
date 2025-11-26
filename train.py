import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

from dataset import CachedAudioDataset
from model import HybridUNet
from dataset import guided_filter_np # optional guided filter (numpy) - reuse function from dataset if available; we'll implement small wrapper

# config
NOISY_DIR = "noisy path"
NOISE_DIR = "noise path"
CACHE_DIR = "cache path"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
N_FFT = 512
HOP = 256
BATCH = 4
EPOCHS = 60
LR = 1e-4
CHECKPOINT_DIR = "checkpoints path"
VIS_DIR = "spectal image saving path"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# VISUALIZATION FUNCTION
def save_spectrogram_grid(noisy_mag, true_noise, pred_mask, denoised_mag, epoch, outdir=VIS_DIR):
    """
    Saves a 2x2 image grid:
    - Noisy
    - True Noise
    - Pred Mask
    - Denoised
    """
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("Noisy (mag)")
    plt.imshow(20 * np.log10(np.maximum(noisy_mag, 1e-8)), origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2, 2, 2)
    plt.title("True Noise (mag)")
    plt.imshow(20 * np.log10(np.maximum(true_noise, 1e-8)), origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2, 2, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Denoised (mag)")
    plt.imshow(20 * np.log10(np.maximum(denoised_mag, 1e-8)), origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    path = os.path.join(outdir, f"epoch_{epoch:03d}.png")
    plt.savefig(path)
    plt.close()


# losses
l1 = nn.L1Loss()

# hyperweights
W_RECON = 0.5
W_VEIL = 0.2
W_MASK_REG = 1e-3
W_TV = 1e-3
GATE_K = 8.0

# tv for mask
def tv_loss(x):
    dx = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dy = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dx + dy

# dataloader
# dataset (FAST PRELOAD)
full_dataset = CachedAudioDataset(CACHE_DIR)

# small train/val split (80/20)
val_pct = 0.1
val_len = max(1, int(len(full_dataset) * val_pct))
train_len = len(full_dataset) - val_len
train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

# data loaders (MULTI-THREAD FAST)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=False
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    pin_memory=False
)

# model + optimizer
model = HybridUNet(base=32).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

#training loop
for epoch in range(1, EPOCHS+1):

    #train
    model.train()
    running = 0.0

    for noisy_mag, noise_mag, illum_log in train_loader:
        noisy_mag = noisy_mag.to(DEVICE)   # [B,1,F,T] linear magnitude
        noise_mag = noise_mag.to(DEVICE)
        illum_log = illum_log.to(DEVICE)   # [B,1,F,T] log-domain

        # normalize illum_log to have similar scale (per-sample)
        illum_mean = illum_log.mean(dim=[2,3], keepdim=True)
        illum_std = illum_log.std(dim=[2,3], keepdim=True) + 1e-9
        illum_norm = (illum_log - illum_mean) / illum_std

        pred_noise, pred_mask, pred_veil = model(noisy_mag, illum_norm)

        # Gate: soft (differentiable)
        vocal_est = torch.clamp(noisy_mag - pred_noise, min=0.0)
        gate = torch.sigmoid(GATE_K * (pred_noise - vocal_est))

        # Effective noise and clean estimate
        effective_noise = pred_noise * pred_mask * gate
        pred_clean = torch.clamp(noisy_mag - effective_noise, min=0.0)
        recon_noisy = pred_clean + pred_noise

        # losses
        loss_noise = l1(pred_noise, noise_mag)            # supervised noise prediction
        loss_recon = l1(recon_noisy, noisy_mag)           # reconstruction consistency
        loss_veil = l1(pred_veil, illum_norm)             # veil supervision (log-domain proxy)
        loss_mask_reg = pred_mask.mean()                  # prevent trivial mask
        loss_tv = tv_loss(pred_mask)                      # smoothness on mask

        loss = loss_noise + W_RECON * loss_recon + W_VEIL * loss_veil + W_MASK_REG * loss_mask_reg + W_TV * loss_tv

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running += loss.item()

    train_avg = running / len(train_loader)

    # validation
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for noisy_mag, noise_mag, illum_log in val_loader:
            noisy_mag = noisy_mag.to(DEVICE)
            noise_mag = noise_mag.to(DEVICE)
            illum_log = illum_log.to(DEVICE)

            illum_mean = illum_log.mean(dim=[2,3], keepdim=True)
            illum_std = illum_log.std(dim=[2,3], keepdim=True) + 1e-9
            illum_norm = (illum_log - illum_mean) / illum_std

            pred_noise, pred_mask, pred_veil = model(noisy_mag, illum_norm)

            vocal_est = torch.clamp(noisy_mag - pred_noise, min=0.0)
            gate = torch.sigmoid(GATE_K * (pred_noise - vocal_est))
            effective_noise = pred_noise * pred_mask * gate
            pred_clean = torch.clamp(noisy_mag - effective_noise, min=0.0)
            recon_noisy = pred_clean + pred_noise

            loss_noise = l1(pred_noise, noise_mag)
            loss_recon = l1(recon_noisy, noisy_mag)
            loss_veil = l1(pred_veil, illum_norm)
            loss_mask_reg = pred_mask.mean()
            loss_tv = tv_loss(pred_mask)

            loss = loss_noise + W_RECON * loss_recon + W_VEIL * loss_veil + W_MASK_REG * loss_mask_reg + W_TV * loss_tv

            val_running += loss.item()

    val_avg = val_running / max(1, len(val_loader))
    scheduler.step(val_avg)

    print(f"Epoch {epoch:03d}/{EPOCHS:03d}  TrainLoss: {train_avg:.6f}  ValLoss: {val_avg:.6f}")



    with torch.no_grad():
        noisy_mag, noise_mag, illum_log = next(iter(val_loader))
        noisy_mag = noisy_mag.to(DEVICE)
        noise_mag = noise_mag.to(DEVICE)
        illum_log = illum_log.to(DEVICE)

        illum_mean = illum_log.mean(dim=[2,3], keepdim=True)
        illum_std = illum_log.std(dim=[2,3], keepdim=True) + 1e-9
        illum_norm = (illum_log - illum_mean) / illum_std

        pred_noise, pred_mask, pred_veil = model(noisy_mag, illum_norm)

        vocal_est = torch.clamp(noisy_mag - pred_noise, min=0.0)
        gate = torch.sigmoid(8 * (pred_noise - vocal_est))
        effective_noise = pred_noise * pred_mask * gate
        pred_clean = torch.clamp(noisy_mag - effective_noise, min=0.0)

        # take first sample
        nn_ = noisy_mag[0].cpu().squeeze().numpy()
        tn_ = noise_mag[0].cpu().squeeze().numpy()
        pm_ = pred_mask[0].cpu().squeeze().numpy()
        dc_ = pred_clean[0].cpu().squeeze().numpy()

        save_spectrogram_grid(nn_, tn_, pm_, dc_, epoch)

    # save checkpoint and a quick visual sample
    if True:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"hybrid_epoch{epoch:03d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict()
        }, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

# final save
final_path = os.path.join(CHECKPOINT_DIR, "hybrid_final.pth")
torch.save({
    'epoch': EPOCHS,
    'model_state': model.state_dict(),
    'opt_state': opt.state_dict()
}, final_path)
print("Training finished. Model saved:", final_path)


