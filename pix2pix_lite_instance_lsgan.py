# file: pix2pix_lite_instance_lsgan.py
"""
Lite Pix2Pix (U-Net generator small) + PatchGAN discriminator (70-like) +
InstanceNorm + LSGAN (MSE) + optional dropout in decoder + LR linear decay.

Usage examples:
  Train:
    python pix2pix_lite_instance_lsgan.py --mode train --dataset_root dataset --size 256 --batch 4 --epochs 60
  Inference:
    python pix2pix_lite_instance_lsgan.py --mode infer --checkpoint checkpoints/g_best.pth --input path/to/gray_dir --output out/
"""
import os
import time
from pathlib import Path
import numpy as np
import cv2
from glob import glob
from skimage import color, metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------
# Config / Device
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# -------------------------
# Small U-Net generator components (use InstanceNorm)
# -------------------------
class DWConvIN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # depthwise conv
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.in1 = nn.InstanceNorm2d(in_ch, affine=True)
        # pointwise conv
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.in1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.in2(x)
        x = self.act(x)
        return x

class DoubleConvDW_IN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            DWConvIN(in_ch, out_ch),
            DWConvIN(out_ch, out_ch)
        )
    def forward(self, x):
        return self.block(x)
        
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base=32, dropout=False):
        """
        Lightweight U-Net generator.
        dropout: if True, apply dropout in decoder (like pix2pix) to increase diversity.
        """
        super().__init__()
        f = base
        self.enc1 = DoubleConvDW_IN(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConvDW_IN(f, f*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConvDW_IN(f*2, f*4)
        self.pool3 = nn.MaxPool2d(2)
        # keep model small: 3 down-samples + bottleneck
        self.bottleneck = DoubleConvDW_IN(f*4, f*8)

        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, 2)
        self.dec3 = DoubleConvDW_IN(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, 2)
        self.dec2 = DoubleConvDW_IN(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, 2)
        self.dec1 = DoubleConvDW_IN(f*2, f)

        # optional dropout at decoder (after concat)
        self.dropout = nn.Dropout2d(0.5) if dropout else nn.Identity()

        # last conv + tanh to map ab to [-1,1]
        self.conv_last = nn.Sequential(
            nn.Conv2d(f, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        if u3.shape[2:] != e3.shape[2:]:
            e3 = F.interpolate(e3, size=u3.shape[2:], mode="bilinear", align_corners=True)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        if u2.shape[2:] != e2.shape[2:]:
            e2 = F.interpolate(e2, size=u2.shape[2:], mode="bilinear", align_corners=True)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[2:] != e1.shape[2:]:
            e1 = F.interpolate(e1, size=u1.shape[2:], mode="bilinear", align_corners=True)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        d1 = self.dropout(d1)
        out = self.conv_last(d1)
        return out

# -------------------------
# PatchGAN discriminator (70-like), with InstanceNorm
# input: concat(L, ab) => 3 channels
# -------------------------
class PatchDiscriminator70(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        b = base
        layers = []
        # conv1: no norm
        layers += [nn.Conv2d(in_channels, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        # conv2
        layers += [nn.Conv2d(b, b*2, 4, 2, 1), nn.InstanceNorm2d(b*2, affine=True), nn.LeakyReLU(0.2, inplace=True)]
        # conv3
        layers += [nn.Conv2d(b*2, b*4, 4, 2, 1), nn.InstanceNorm2d(b*4, affine=True), nn.LeakyReLU(0.2, inplace=True)]
        # conv4 (adds receptive field, making it ~70)
        layers += [nn.Conv2d(b*4, b*8, 4, 1, 1), nn.InstanceNorm2d(b*8, affine=True), nn.LeakyReLU(0.2, inplace=True)]
        # final patch output
        layers += [nn.Conv2d(b*8, 1, 4, 1, 1)]  # output logits (use MSE with targets 1/0)
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# -------------------------
# Dataset: uses dataset/raw folder to compute LAB (L and ab). Expects jpg images.
# -------------------------
class ColorizationDataset(Dataset):
    def __init__(self, root_dir="dataset", img_size=256, augment=False, train=True, val_split=0.1):
        self.root = Path(root_dir)
        self.img_size = img_size
        all_files = sorted([p.stem for p in (self.root/'raw').glob('*.jpg')])
        split_idx = int(len(all_files)*(1-val_split))
        self.names = all_files[:split_idx] if train else all_files[split_idx:]
        base = [A.Resize(img_size, img_size)]
        if augment:
            base += [A.HorizontalFlip(p=0.5), A.Rotate(limit=15, p=0.4), A.RandomBrightnessContrast(p=0.2)]
        base += [A.Normalize(mean=(0.0,), std=(1.0,)), ToTensorV2()]
        # mask is ab with 2 channels
        self.transform = A.Compose(base, additional_targets={'mask':'image'})

    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        raw_p = self.root / "raw" / f"{name}.jpg"
        img = cv2.imread(str(raw_p))
        if img is None:
            return self.__getitem__((idx+1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = color.rgb2lab(img).astype(np.float32)
        L = lab[...,0:1] / 100.0  # 0..1
        ab = lab[...,1:3] / 128.0  # approx -1..1
        aug = self.transform(image=L, mask=ab)
        inp = aug['image'].float()    # [1,H,W]
        target = aug['mask'].float()  # [2,H,W]
        return inp, target, name

# -------------------------
# Utilities
# -------------------------
def tv_loss(x):
    dx = torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]))
    dy = torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]))
    return dx + dy

# -------------------------
# Training loop (GAN) with LSGAN (MSE) & linear lr decay (optional)
# -------------------------
def train(args):
    # data
    train_ds = ColorizationDataset(root_dir=args.dataset_root, img_size=args.size, augment=True, train=True, val_split=args.val_split)
    val_ds = ColorizationDataset(root_dir=args.dataset_root, img_size=args.size, augment=False, train=False, val_split=args.val_split)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    vloader = DataLoader(val_ds, batch_size=max(1,args.batch//2), shuffle=False, num_workers=2, pin_memory=True)

    # models
    G = UNetGenerator(in_channels=1, out_channels=2, base=args.base_filters, dropout=args.dropout).to(DEVICE)
    D = PatchDiscriminator70(in_channels=3, base=args.d_base).to(DEVICE)

    # nếu có checkpoint resume
    if args.resume_g is not None and os.path.exists(args.resume_g):
        ckptG = torch.load(args.resume_g, map_location=DEVICE)
        if isinstance(ckptG, dict) and any(k.startswith('module.') for k in ckptG.keys()):
            ckptG = {k.replace('module.', ''): v for k,v in ckptG.items()}
        G.load_state_dict(ckptG)
        print("Loaded G checkpoint:", args.resume_g)

    if args.resume_d is not None and os.path.exists(args.resume_d):
        ckptD = torch.load(args.resume_d, map_location=DEVICE)
        if isinstance(ckptD, dict) and any(k.startswith('module.') for k in ckptD.keys()):
            ckptD = {k.replace('module.', ''): v for k,v in ckptD.items()}
        D.load_state_dict(ckptD)
        print("Loaded D checkpoint:", args.resume_d)

    # optimizers
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))


    start_epoch = args.start_epoch
    if hasattr(args, 'resume') and args.resume is not None and os.path.exists(args.resume):
        print(f"Resuming from checkpoint {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=DEVICE)
        
        # load state_dict cho G/D/opt
        if 'G' in ckpt and 'D' in ckpt and 'optG' in ckpt and 'optD' in ckpt and 'epoch' in ckpt:
            G.load_state_dict(ckpt['G'])
            D.load_state_dict(ckpt['D'])
            optG.load_state_dict(ckpt['optG'])
            optD.load_state_dict(ckpt['optD'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Resumed training from epoch {start_epoch}")
        else:
            # fallback: chỉ load G nếu file cũ kiểu g_best.pth
            G.load_state_dict(ckpt)
            print("Loaded G weights only (no optimizers, resume starts from epoch 1)")

    # schedulers: linear decay after args.lr_decay_start epoch
    if args.lr_decay_start >= 0:
        def lambda_rule(epoch):
            if epoch < args.lr_decay_start:
                return 1.0
            else:
                pct = float(epoch - args.lr_decay_start) / max(1, (args.epochs - args.lr_decay_start))
                return max(0.0, 1.0 - pct)
        schedG = torch.optim.lr_scheduler.LambdaLR(optG, lr_lambda=lambda_rule)
        schedD = torch.optim.lr_scheduler.LambdaLR(optD, lr_lambda=lambda_rule)
    else:
        schedG = schedD = None

    # losses: L1 for reconstruction, MSE for LSGAN
    l1 = nn.L1Loss()
    adv_crit = nn.MSELoss()  # LSGAN
    real_label = 1.0
    fake_label = 0.0

    best_val = 1e9
    os.makedirs('checkpoints', exist_ok=True)

    # optional pretrain (L1 only)
    if args.pretrain_epochs > 0:
        print(f"Pretraining G for {args.pretrain_epochs} epochs (L1 only)...")
        for e in range(1, args.pretrain_epochs+1):
            G.train()
            running = 0.0
            loop = tqdm(loader, desc=f"Pretrain {e}/{args.pretrain_epochs}")
            for inp, target, _ in loop:
                inp = inp.to(DEVICE); target = target.to(DEVICE)
                optG.zero_grad()
                pred = G(inp)
                loss = args.w_l1 * l1(pred, target)
                loss.backward(); optG.step()
                running += loss.item()
                loop.set_postfix(L1=f"{running/(loop.n+1):.4f}")
            torch.save(G.state_dict(), f"checkpoints/g_pretrain_epoch{e}.pth")
        print("Pretrain done.")

    # training
    for epoch in range(args.start_epoch, args.epochs+1):
        G.train(); D.train()
        loop = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        t0 = time.time()
        for inp, target, _ in loop:
            inp = inp.to(DEVICE); target = target.to(DEVICE)
            B = inp.size(0)

            # ---------------------
            # Update D (real and fake pairs)
            # ---------------------
            optD.zero_grad()
            real_pair = torch.cat([inp, target], dim=1)  # [B,3,H,W]
            pred_real = D(real_pair)
            real_tensor = torch.full_like(pred_real, real_label, device=DEVICE)
            loss_D_real = adv_crit(pred_real, real_tensor)

            with torch.no_grad():
                fake_ab = G(inp)
            fake_pair = torch.cat([inp, fake_ab], dim=1)
            pred_fake = D(fake_pair)
            fake_tensor = torch.full_like(pred_fake, fake_label, device=DEVICE)
            loss_D_fake = adv_crit(pred_fake, fake_tensor)

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward(); optD.step()

            # ---------------------
            # Update G
            # ---------------------
            optG.zero_grad()
            pred_ab = G(inp)
            fake_pair_forG = torch.cat([inp, pred_ab], dim=1)
            pred_forG = D(fake_pair_forG)
            adv_target = torch.full_like(pred_forG, real_label, device=DEVICE)
            loss_G_adv = adv_crit(pred_forG, adv_target)  # LSGAN loss
            loss_G_l1 = l1(pred_ab, target)
            loss_G_tv = tv_loss(pred_ab) if args.w_tv > 0 else 0.0

            loss_G = args.w_l1 * loss_G_l1 + args.w_adv * loss_G_adv + args.w_tv * loss_G_tv
            loss_G.backward(); optG.step()

            loop.set_postfix(D=f"{loss_D.item():.4f}", G=f"{loss_G.item():.4f}")

        # epoch end
        if schedG is not None:
            schedG.step(); schedD.step()
        t1 = time.time()
        print(f"Epoch {epoch} finished in {t1-t0:.1f}s. Saving...")
        torch.save(G.state_dict(), f"checkpoints/g_epoch{epoch}.pth")
        torch.save(D.state_dict(), f"checkpoints/d_epoch{epoch}.pth")

        # Basic validation: L1 + PSNR on RGB reconversion
        G.eval()
        val_loss = 0.0
        psnrs = []
        with torch.no_grad():
            for inp, target, _ in vloader:
                inp = inp.to(DEVICE); target = target.to(DEVICE)
                pred = G(inp)
                val_loss += l1(pred, target).item()
                # compute PSNR per-sample
                pred_ab = (pred * 128.0).cpu().numpy()
                L = (inp * 100.0).cpu().numpy()
                target_ab = (target * 128.0).cpu().numpy()
                for b in range(pred_ab.shape[0]):
                    L_b = np.transpose(L[b], (1,2,0))
                    ab_b = np.transpose(pred_ab[b], (1,2,0))
                    gt_ab_b = np.transpose(target_ab[b], (1,2,0))
                    lab = np.concatenate([L_b, ab_b], axis=2)
                    gt_lab = np.concatenate([L_b, gt_ab_b], axis=2)
                    rgb = color.lab2rgb(np.clip(lab, [0,-128,-128], [100,127,127]))
                    gt_rgb = color.lab2rgb(np.clip(gt_lab, [0,-128,-128], [100,127,127]))
                    psnrs.append(metrics.peak_signal_noise_ratio(gt_rgb, rgb, data_range=1.0))
        avg_val = val_loss / max(1, len(vloader))
        avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
        print(f"Val L1: {avg_val:.4f}, PSNR: {avg_psnr:.2f}dB")
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                'G': G.state_dict(),
                'D': D.state_dict(),
                'optG': optG.state_dict(),
                'optD': optD.state_dict(),
                'epoch': epoch
            }, f"checkpoints/ckpt_epoch{epoch}.pth")

            print("Saved best G checkpoint.")

# -------------------------
# Inference: use G only
# -------------------------
def infer(args):
    G = UNetGenerator(in_channels=1, out_channels=2, base=args.base_filters, dropout=False).to(DEVICE)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError("Checkpoint not found: " + args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    # handle DataParallel keys
    if isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', ''): v for k,v in ckpt.items()}
    G.load_state_dict(ckpt)
    G.eval()

    # input dir or file
    if os.path.isdir(args.input):
        exts = ('.jpg', '.png', '.jpeg', '.bmp', '.tif')
        files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(exts)])
    else:
        files = [args.input]

    # ensure output dir
    if args.output.endswith(os.sep) or os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
        out_is_dir = True
    else:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        out_is_dir = False

    for f in files:
        img = cv2.imread(f)
        if img is None:
            print("Cannot read", f)
            continue

        # Nếu ảnh có 3 kênh → chuyển sang Lab để lấy kênh L
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            gray = img_lab[..., 0]  # L channel
        else:
            gray = img  # đã là ảnh xám

        H, W = gray.shape
        gray_f = cv2.resize(gray, (args.size, args.size)).astype(np.float32) / 255.0
        inp_np = np.expand_dims(gray_f, axis=2)
        inp_t = torch.from_numpy(np.transpose(inp_np, (2, 0, 1))).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            pred_ab = G(inp_t)                  # (1,2,H,W)
            pred_ab = torch.clamp(pred_ab, -1,1)  # chắc chắn trong [-1,1]
            pred_ab = pred_ab * 128.0          # scale [-128,127]

            L = gray_f * 100.0                 # L kênh chuẩn
            lab = np.zeros((args.size, args.size, 3), dtype=np.float32)
            lab[...,0] = L
            lab[...,1:] = np.transpose(pred_ab.cpu().numpy()[0], (1,2,0))

            # clip LAB chuẩn
            lab[...,0] = np.clip(lab[...,0],0,100)
            lab[...,1:] = np.clip(lab[...,1:], -128,127)

            rgb = color.lab2rgb(lab)
            rgb = np.clip(rgb,0,1)
            rgb = (rgb*255).astype(np.uint8)
            rgb = cv2.resize(rgb,(W,H))

        if out_is_dir:
            out_path = os.path.join(args.output, os.path.splitext(os.path.basename(f))[0] + "_pix2pix_lite.png")
        else:
            out_path = args.output

        cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print("Wrote", out_path)
        print(pred_ab.min(), pred_ab.max())

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train','infer'], default='train')
    p.add_argument('--dataset_root', default='dataset')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--pretrain_epochs', type=int, default=3, help="Train G with L1 only before GAN")
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--w_l1', type=float, default=1.0)
    p.add_argument('--w_adv', type=float, default=0.01)
    p.add_argument('--w_tv', type=float, default=1e-5)
    p.add_argument('--base_filters', type=int, default=32)
    p.add_argument('--d_base', type=int, default=32)
    p.add_argument('--dropout', action='store_true', help="Enable dropout in decoder (like pix2pix)")
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--lr_decay_start', type=int, default=-1, help="Epoch to start linear decay; -1 to disable")
    p.add_argument('--checkpoint', type=str, default='checkpoints/g_best.pth')
    p.add_argument('--input', type=str, default=None)
    p.add_argument('--output', type=str, default='out/')
    p.add_argument('--resume_g', type=str, default=None, help='Path to G weights to resume from')
    p.add_argument('--resume_d', type=str, default=None, help='Path to D weights to resume from')
    p.add_argument('--start_epoch', type=int, default=1, help='Epoch to start training from')

    args = p.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        assert args.input is not None
        infer(args)
