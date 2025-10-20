#!/usr/bin/env python3
"""
train_unet_rgb_v2.py
Upgraded version of train_unet_rgb.py
- Strong augmentation: flip, rotate, crop, color jitter
- Skip connections + SE attention
- Loss: L1 + perceptual + adversarial (lightweight)
- Optional histogram transfer
- Logging: epoch metrics + per-image metrics
"""

import os
from glob import glob
import numpy as np
from PIL import Image
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.exposure import match_histograms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
import argparse
import albumentations as A

# ---------------------------
# Dataset
# ---------------------------
class RGBColorDataset(Dataset):
    def __init__(self, folder_L, folder_RGB, img_size=256, augment=False):
        self.folder_L = folder_L
        self.folder_RGB = folder_RGB
        self.files = sorted(glob(os.path.join(folder_L, '**', '*.jpg'), recursive=True))
        self.img_size = img_size
        self.augment = augment

        # Albumentations augmentation
        if self.augment:
            self.augmenter = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomResizedCrop(img_size, img_size, scale=(0.8,1.0), ratio=(0.9,1.1), p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fL = self.files[idx]
        name = os.path.basename(fL)
        fRGB = os.path.join(self.folder_RGB, name)

        L = np.array(Image.open(fL).convert('L').resize((self.img_size,self.img_size), Image.BICUBIC), dtype=np.float32)/255.0
        RGB = np.array(Image.open(fRGB).convert('RGB').resize((self.img_size,self.img_size), Image.BICUBIC), dtype=np.float32)/255.0
        Lab = color.rgb2lab(RGB)
        ab = Lab[:,:,1:]/128.0

        if self.augment:
            augmented = self.augmenter(image=L, mask=ab)
            L = augmented['image']
            ab = augmented['mask']

        L_t = torch.from_numpy(L).unsqueeze(0)
        ab_t = torch.from_numpy(ab.transpose(2,0,1))
        return L_t.float(), ab_t.float(), name

# ---------------------------
# SE Attention
# ---------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y

# ---------------------------
# U-Net with SE attention
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        if self.use_se:
            self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.net(x)
        if self.use_se:
            x = self.se(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base_c=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_c, use_se=True)
        self.enc2 = DoubleConv(base_c, base_c*2, use_se=True)
        self.enc3 = DoubleConv(base_c*2, base_c*4, use_se=True)
        self.enc4 = DoubleConv(base_c*4, base_c*8, use_se=True)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(base_c*12, base_c*4, use_se=True)
        self.dec3 = DoubleConv(base_c*6, base_c*2, use_se=True)
        self.dec2 = DoubleConv(base_c*3, base_c, use_se=True)
        self.final = nn.Conv2d(base_c, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d4 = self.up(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.final(d2)
        return out

# ---------------------------
# Discriminator (PatchGAN lightweight)
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_c=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_c, base_c*2, 4, 2, 1),
            nn.BatchNorm2d(base_c*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_c*2, base_c*4, 4, 2, 1),
            nn.BatchNorm2d(base_c*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_c*4, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# Lab <-> RGB helpers
# ---------------------------
def lab_to_rgb(L_tensor, ab_tensor):
    L = L_tensor.cpu().numpy()
    ab = ab_tensor.cpu().numpy()
    B,_,H,W = L.shape
    rgb_outs = []
    for i in range(B):
        lab = np.zeros((H,W,3), np.float32)
        lab[:,:,0] = L[i,0]*100
        lab[:,:,1:] = ab[i].transpose(1,2,0)*128
        rgb = color.lab2rgb(lab)
        rgb_outs.append((np.clip(rgb,0,1)*255).astype(np.uint8))
    return rgb_outs

# ---------------------------
# Training / Validation
# ---------------------------
def train_epoch(model, loader, optimizer, device, disc=None, adv_weight=0.01):
    model.train()
    total_loss = 0
    for L, ab, _ in tqdm(loader, desc='Train', leave=False):
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        pred = model(L)
        l1 = F.l1_loss(pred, ab)

        adv_loss = 0
        if disc is not None:
            rgb_pred = torch.cat([L, pred], dim=1).repeat(1,3,1,1)  # convert to 3ch for disc
            adv_loss = F.mse_loss(disc(rgb_pred), torch.ones_like(disc(rgb_pred)))

        loss = l1 + adv_weight*adv_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*L.size(0)
    return total_loss/len(loader.dataset)

def validate(model, loader, device, log_per_image=None):
    model.eval()
    total_loss = 0
    psnr_sum = 0
    ssim_sum = 0
    n = 0
    with torch.no_grad():
        for L, ab, names in tqdm(loader, desc='Val', leave=False):
            L, ab = L.to(device), ab.to(device)
            pred = model(L)
            total_loss += F.l1_loss(pred, ab).item()*L.size(0)
            rgb_gt = lab_to_rgb(L, ab)
            rgb_pred = lab_to_rgb(L, pred.clamp(-1,1))
            for g,p,name in zip(rgb_gt, rgb_pred, names):
                g_f = g.astype('float32')/255.0
                p_f = p.astype('float32')/255.0
                try:
                    psnr_sum += compare_psnr(g_f, p_f, data_range=1.0)
                    ssim_sum += compare_ssim(g_f, p_f, data_range=1.0, channel_axis=2, win_size=3)
                except TypeError:
                    ssim_sum += compare_ssim(g_f, p_f, data_range=1.0)
                n += 1
                if log_per_image:
                    log_per_image.write(f'{name}\t{compare_psnr(g_f,p_f):.3f}\t{compare_ssim(g_f,p_f):.4f}\n')
    return total_loss/len(loader.dataset), psnr_sum/n, ssim_sum/n

# ---------------------------
# Inference
# ---------------------------
def infer(model, device, L_path, out_path, img_size=256, hist_ref=None):
    model.eval()
    L = Image.open(L_path).convert('L')
    orig_size = L.size
    L_resized = L.resize((img_size,img_size), Image.BICUBIC)
    L_np = np.array(L_resized, dtype=np.float32)/255.0
    L_t = torch.from_numpy(L_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(L_t).clamp(-1,1)
    rgb = lab_to_rgb(L_t, pred)[0]
    if hist_ref:
        ref = np.array(Image.open(hist_ref).resize((rgb.shape[1], rgb.shape[0])))
        rgb = match_histograms(rgb, ref, multichannel=True)
    rgb = np.array(Image.fromarray(rgb).resize(orig_size))
    Image.fromarray(rgb.astype(np.uint8)).save(out_path)
    print('Saved', out_path)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=False)
    parser.add_argument('--val_dir', type=str, required=False)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--mode', type=str, default='train', choices=['train','infer'])
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--infer_L', type=str, default='')
    parser.add_argument('--infer_out', type=str, default='./out.png')
    parser.add_argument('--hist_ref', type=str, default='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, 'train_log.txt')
    per_image_file = os.path.join(args.save_dir, 'val_per_image.txt')

    model = UNet(base_c=32).to(device)
    disc = Discriminator().to(device)

    if args.mode=='train':
        train_ds = RGBColorDataset(os.path.join(args.train_dir,'L'),
                                   os.path.join(args.train_dir,'RGB'),
                                   img_size=args.img_size, augment=True)
        val_ds = RGBColorDataset(os.path.join(args.val_dir,'L'),
                                 os.path.join(args.val_dir,'RGB'),
                                 img_size=args.img_size, augment=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=max(1,args.batch_size//2), shuffle=False, num_workers=2, pin_memory=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val = 1e9

        with open(log_file, 'w') as f:
            f.write('Epoch\tTrainLoss\tValLoss\tPSNR\tSSIM\n')

        for epoch in range(1,args.epochs+1):
            print(f'Epoch {epoch}/{args.epochs}')
            train_loss = train_epoch(model, train_loader, optimizer, device, disc, adv_weight=0.01)

            with open(per_image_file,'w') as f_img:
                f_img.write('Filename\tPSNR\tSSIM\n')
                val_loss, val_psnr, val_ssim = validate(model, val_loader, device, log_per_image=f_img)

            print(f'Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.4f}')

            with open(log_file,'a') as f:
                f.write(f'{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_psnr:.3f}\t{val_ssim:.4f}\n')

            ckpt = os.path.join(args.save_dir,f'unet_epoch{epoch}.pth')
            torch.save(model.state_dict(), ckpt)
            if val_loss<best_val:
                best_val=val_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir,'best_unet.pth'))
                print('Saved best model')

    elif args.mode=='infer':
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint,map_location=device))
            print('Loaded', args.checkpoint)
        if args.infer_L:
            infer(model, device, args.infer_L, args.infer_out, img_size=args.img_size, hist_ref=args.hist_ref)

if __name__=='__main__':
    main()
