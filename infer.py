import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
from model import HybridUNet

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
N_FFT = 512
HOP = 256

CHECKPOINT = "check point path"

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
def load_model():
    model = HybridUNet(base=32).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Loaded checkpoint from:", CHECKPOINT)
    return model

# ------------------------------------------------
# AUDIO → STFT MAG + LOG ILLUM
# ------------------------------------------------
def preprocess_audio(path):
    # Load audio
    wav, sr = sf.read(path)
    wav = torch.tensor(wav, dtype=torch.float32)

    # Stereo → mono
    if wav.ndim > 1:
        wav = wav.mean(dim=1)

    # Add channel dim for STFT: [1, n_samples]
    wav = wav.unsqueeze(0)

    # Resample if needed
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    # STFT
    stft = torch.stft(
        wav,               # shape [1, n_samples]
        n_fft=N_FFT,
        hop_length=HOP,
        return_complex=True,
        window=torch.hann_window(N_FFT, device=wav.device)  # better window
    )

    mag = stft.abs()
    phase = stft.angle()
    illum = torch.log1p(mag + 1e-8)

    # Normalize illum: [1, freq, time]
    mean = illum.mean(dim=[1,2], keepdim=True)
    std = illum.std(dim=[1,2], keepdim=True) + 1e-9
    illum_norm = (illum - mean) / std

    return mag.to(DEVICE), illum_norm.to(DEVICE), phase.to(DEVICE)



# ------------------------------------------------
# RECONSTRUCT USING MASK + NOISE PRED
# ------------------------------------------------
def run_model(model, mag, illum_norm):
    with torch.no_grad():
        pred_noise, pred_mask, pred_veil = model(
            mag.unsqueeze(0),      # [1,1,F,T]
            illum_norm.unsqueeze(0)
        )

    pred_noise = pred_noise.squeeze(0)
    pred_mask  = pred_mask.squeeze(0)

    # follow the SAME logic as train.py
    vocal_est = torch.clamp(mag - pred_noise, min=0.0)
    gate = torch.sigmoid(8 * (pred_noise - vocal_est))
    effective_noise = pred_noise * pred_mask * gate
    pred_clean = torch.clamp(mag - effective_noise, min=0.0)

    return pred_clean

# ------------------------------------------------
# ISTFT RECONSTRUCTION
# ------------------------------------------------
def reconstruct_audio(mag, phase):
    complex_spec = torch.polar(mag, phase)
    window = torch.hann_window(N_FFT, device=mag.device)
    wav = torch.istft(
        complex_spec,
        n_fft=N_FFT,
        hop_length=HOP,
        window=window
    )
    return wav.unsqueeze(0)  # [1, n_samples]


# ------------------------------------------------
# MAIN INFER FUNCTION
# ------------------------------------------------
def infer(noisy_path, out_path):
    print("Loading model...")
    model = load_model()

    print("Preprocessing audio...")
    mag, illum_norm, phase = preprocess_audio(noisy_path)

    print("Running model...")
    clean_mag = run_model(model, mag, illum_norm)

    print("Reconstructing waveform...")
    clean_wav = reconstruct_audio(clean_mag.cpu(), phase.cpu())

    # Convert to 1D numpy array for soundfile
    clean_wav_np = clean_wav.squeeze().numpy()  # [n_samples]

    # Save using soundfile
    sf.write(out_path, clean_wav_np, SR)
    print("Saved cleaned vocal to:", out_path)

# ------------------------------------------------
# RUN EXAMPLE
# ------------------------------------------------
if __name__ == "__main__":
    noisy_file = "noisy audio path for cleaning" 
    out_file   = "cleaned output path"   # <-- OUTPUT

    infer(noisy_file, out_file)
