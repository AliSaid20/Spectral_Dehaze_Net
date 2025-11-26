
import os
import re
import torch
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf
import numpy as np
from scipy.ndimage import gaussian_filter



ID_RE = re.compile(r'[nv](\d{4})\.wav$')

# lightweight guided filter helpers (used to compute illum target)

def box_filter(img, r):
    # img: 2D numpy
    h, w = img.shape
    s = np.pad(img, ((1, 0), (1, 0)), mode='constant', constant_values=0.0)
    integral = np.cumsum(np.cumsum(s, axis=0), axis=1)
    out = np.empty_like(img, dtype=np.float32)
    for i in range(h):
        y0 = max(0, i - r)
        y1 = min(h - 1, i + r)
        for j in range(w):
            x0 = max(0, j - r)
            x1 = min(w - 1, j + r)
            ssum = integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]
            out[i, j] = ssum
    return out


def guided_filter_np(g, p, r=8, eps=1e-3):
    # g, p: 2D numpy
    g = g.astype(np.float32)
    p = p.astype(np.float32)
    h, w = g.shape
    N = box_filter(np.ones((h, w), dtype=np.float32), r)
    mean_g = box_filter(g, r) / N
    mean_p = box_filter(p, r) / N
    mean_gp = box_filter(g * p, r) / N
    cov_gp = mean_gp - mean_g * mean_p
    mean_gg = box_filter(g * g, r) / N
    var_g = mean_gg - mean_g * mean_g
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    mean_a = box_filter(a, r) / N
    mean_b = box_filter(b, r) / N
    q = mean_a * g + mean_b
    return q.astype(np.float32)


def compute_msr_illum_log(log_mag, scales=(15, 80, 250), weights=None, gf_r=8, gf_eps=1e-3):
    # log_mag: numpy (F,T)
    if weights is None:
        weights = [1.0 / len(scales)] * len(scales)
    msr = np.zeros_like(log_mag, dtype=np.float32)
    for s, w in zip(scales, weights):
        blurred = gaussian_filter(log_mag, sigma=s, mode='nearest')
        msr += w * blurred
    # guided filter refine using log_mag as guidance
    try:
        refined = guided_filter_np(log_mag, msr, r=gf_r, eps=gf_eps)
    except Exception:
        refined = msr
    return refined


class AudioDataset(Dataset):
    def __init__(self, noise_dir, noisy_dir, sample_rate=16000, n_fft=512, hop_length=256, segment_seconds=5, preload=False):
        self.noise_dir = noise_dir
        self.noisy_dir = noisy_dir
        self.sample_rate = sample_rate
        self.segment_len = sample_rate * segment_seconds
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.ids = self._collect_ids()

        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)

    def _collect_ids(self):
        noise_files = [f for f in os.listdir(self.noise_dir) if f.lower().endswith('.wav')]
        noisy_files = [f for f in os.listdir(self.noisy_dir) if f.lower().endswith('.wav')]
        noise_ids = set()
        noisy_ids = set()
        for f in noise_files:
            m = ID_RE.search(f)
            if m:
                noise_ids.add(m.group(1))
        for f in noisy_files:
            m = ID_RE.search(f)
            if m:
                noisy_ids.add(m.group(1))
        ids = sorted(list(noise_ids.intersection(noisy_ids)))
        return ids

    def __len__(self):
        return len(self.ids)

    def load_wav(self, path):
        wav, sr = sf.read(path)

        # Convert to mono
        if wav.ndim > 1:
            wav = wav.mean(axis=1)   # correct numpy style

        # Convert to torch
        wav = torch.tensor(wav, dtype=torch.float32)


        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def _crop_or_pad(self, wav):
        L = wav.shape[0]
        if L > self.segment_len:
            start = torch.randint(0, L - self.segment_len + 1, (1,)).item()
            return wav[start:start + self.segment_len]
        elif L < self.segment_len:
            pad = torch.zeros(self.segment_len - L)
            return torch.cat([wav, pad], dim=0)
        else:
            return wav

    def _mag_log(self, wav):
        # returns linear magnitude and log1p magnitude as numpy
        spec = self.stft(wav.unsqueeze(0))  # complex: (1,2,F,T)
        # torch.stft with complex not used here; torchaudio Spectrogram returns complex tensor with real/imag
        if torch.is_complex(spec):
            mag = torch.abs(spec)
        else:
            # older torchaudio returns real/imag in last dim
            mag = torch.sqrt(spec.pow(2).sum(-1))
        mag = mag.squeeze(0).cpu().numpy()  # (F, T)
        log = np.log1p(mag)
        return mag.astype(np.float32), log.astype(np.float32)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        noisy_path = os.path.join(self.noisy_dir, f"v{id_}.wav")
        noise_path = os.path.join(self.noise_dir, f"n{id_}.wav")

        noisy_wav = self.load_wav(noisy_path)
        noise_wav = self.load_wav(noise_path)

        print("Loading:", noisy_path)
        print("Loading:", noise_path)

        noisy_wav = self._crop_or_pad(noisy_wav)
        noise_wav = self._crop_or_pad(noise_wav)

        noisy_mag, noisy_log = self._mag_log(noisy_wav)
        noise_mag, noise_log = self._mag_log(noise_wav)



        # compute illumination (log domain)
        illum_log = compute_msr_illum_log(noisy_log)

        # return tensors: linear mags for subtraction supervision, illum_log for veil supervision
        noisy_t = torch.from_numpy(noisy_mag).unsqueeze(0)  # [1,F,T]
        noise_t = torch.from_numpy(noise_mag).unsqueeze(0)
        illum_t = torch.from_numpy(illum_log).unsqueeze(0)


        return noisy_t.float(), noise_t.float(), illum_t.float()

# ---------------------------------------------------------
#  MODE: CACHE DATASET (training from .pt cache)
# ---------------------------------------------------------

class CachedAudioDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.files = sorted([f for f in os.listdir(cache_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(os.path.join(self.cache_dir, self.files[idx]))
        return d["noisy_mag"], d["noise_mag"], d["illum"]
