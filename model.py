import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class HybridUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        # input channels: noisy mag (linear) and illum_log (normalized) -> 2 channels
        self.inc = ConvBlock(2, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.bot = ConvBlock(base*8, base*16)
        self.up1 = Up(base*16, base*8)
        self.up2 = Up(base*8, base*4)
        self.up3 = Up(base*4, base*2)
        self.up4 = Up(base*2, base)

        # three heads
        self.noise_head = nn.Conv2d(base, 1, 1)
        self.mask_head = nn.Conv2d(base, 1, 1)
        self.veil_head = nn.Conv2d(base, 1, 1)

    def forward(self, noisy_mag, illum_log):
        # both tensors expected shape [B,1,F,T]
        x = torch.cat([noisy_mag, illum_log], dim=1)  # [B,2,F,T]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bot(x4)
        x = self.up1(xb, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        pred_noise = F.softplus(self.noise_head(x))   # >=0
        pred_mask = torch.sigmoid(self.mask_head(x))  # 0..1
        pred_veil = self.veil_head(x)  # veil in same domain as illum_log (log domain), allow negative until supervised

        return pred_noise, pred_mask, pred_veil
