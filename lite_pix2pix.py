# file: lite_pix2pix.py
# Modified to support grayscale input (in_channels=1) while optionally
# reusing pretrained ResNet34 conv1 weights by averaging RGB channels.

import os
import time
from pathlib import Path
from glob import glob

import cv2
import numpy as np
from skimage import color, metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import VGG16_Weights

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("default", category=UserWarning, module="skimage.color")

# -------------------------
# Config / Device
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# -------------------------
# Basic blocks
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)

# -------------------------
# UNet with ResNet34 encoder + Fusion (keeps API: forward returns pred_ab in [-1,1])
# -------------------------
class FusionModule(nn.Module):
    def __init__(self, in_channels, global_dim=512, mid_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, mid_dim),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels + mid_dim, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_feat, global_vec):
        g = self.fc(global_vec)  # (B, mid_dim)
        B, _, H, W = local_feat.shape
        g_map = g.view(B, g.size(1), 1, 1).expand(-1, -1, H, W)
        fused = torch.cat([local_feat, g_map], dim=1)
        return self.fuse_conv(fused)


class UNetGen(nn.Module):
    """
    Generator: ResNet34 encoder + Fusion module + lightweight decoder
    Returns predicted ab channels scaled to [-1,1] (so it's consistent with dataset target which is ab/128)
    """
    def __init__(self, in_channels=1, out_channels=2, base_filters=32, pretrained=True):
        super().__init__()
        # load resnet34 and take encoder layers
        # NOTE: torchvision's pretrained argument emits a deprecation warning in newer versions
        resnet = models.resnet34(pretrained=pretrained)

        # --- Important fix: handle grayscale (in_channels=1) while still allowing
        #     reuse of pretrained conv1 weights by averaging RGB channels.
        if in_channels == 3:
            conv1 = resnet.conv1
        else:
            # create a new conv1 that accepts `in_channels` (e.g. 1 for grayscale)
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                # convert pretrained RGB conv1 weights [64,3,7,7] -> [64,in_channels,7,7]
                with torch.no_grad():
                    pre_w = resnet.conv1.weight.data  # [64,3,7,7]
                    # average across RGB channels to get a reasonable grayscale init
                    new_w = pre_w.mean(dim=1, keepdim=True)  # [64,1,7,7]
                    if in_channels == 1:
                        conv1.weight.data.copy_(new_w)
                    else:
                        # if someone uses other channel sizes, repeat/trim as needed
                        conv1.weight.data.copy_(new_w.repeat(1, in_channels, 1, 1)[:, :in_channels, :, :])

        # initial stem (use conv1 we constructed)
        self.stem = nn.Sequential(conv1, resnet.bn1, resnet.relu)  # outputs 64 ch, /2 spatial
        self.maxpool = resnet.maxpool
        self.enc1 = resnet.layer1  # 64
        self.enc2 = resnet.layer2  # 128
        self.enc3 = resnet.layer3  # 256
        self.enc4 = resnet.layer4  # 512

        # bridge to adjust channels
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # fusion module (global prior)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = FusionModule(in_channels=512, global_dim=512, mid_dim=128)

        # decoder blocks (upsample + conv)
        def dec_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.dec4 = dec_block(512, 256)
        self.dec3 = dec_block(256, 128)
        self.dec2 = dec_block(128, 64)
        self.dec1 = dec_block(64, 64)

        # final conv -> out_channels (2 for ab) + tanh to map to [-1,1]
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # <-- upsample từ H/2 -> H
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()
        )

        # small convs to match skip channels if needed (not strictly necessary but stable)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # do not re-init conv1 if it was loaded from pretrained weights
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B,1,H,W) with L normalized [0,1]
        x0 = self.stem(x)    # (B,64,H/2,W/2)
        x0p = self.maxpool(x0)
        e1 = self.enc1(x0p)  # down1
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)   # (B,512,H/32,W/32)

        g = self.global_pool(e4).view(e4.size(0), -1)  # (B,512)
        b = self.bridge(e4)  # (B,512,h,w)
        fused = self.fusion(b, g)

        d4 = self.dec4(fused)
        # skip add (resize if needed)
        if d4.shape[2:] != e3.shape[2:]:
            e3 = F.interpolate(e3, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4 = d4 + e3

        d3 = self.dec3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            e2 = F.interpolate(e2, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = d3 + e2

        d2 = self.dec2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            e1 = F.interpolate(e1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = d2 + e1

        d1 = self.dec1(d2)

        out_ab = self.final(d1)  # in [-1,1] per pixel
        return out_ab

# -------------------------
# Discriminator (PatchGAN lightweight)
# -------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        b = base
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, b, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(b, b*2, 4, 2, 1),
            nn.BatchNorm2d(b*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(b*2, b*4, 4, 2, 1),
            nn.BatchNorm2d(b*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(b*4, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.model(x)

# -------------------------
# Dataset (use L as input, ab as target) -- unchanged
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
            base += [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.4),
                A.RandomBrightnessContrast(p=0.2),
            ]
        base += [A.Normalize(mean=(0.0,), std=(1.0,)), ToTensorV2()]
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
        L = lab[...,0:1] / 100.0       # 0..1
        ab = lab[...,1:3] / 128.0     # approx -1..1

        augmented = self.transform(image=L, mask=ab)
        inp = augmented['image'].float()   # [1,H,W]
        target = augmented['mask'].float() # [2,H,W]
        return inp, target, name

# -------------------------
# Utilities
# -------------------------
def tv_loss(x):
    dx = torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]))
    dy = torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]))
    return dx + dy

def lab_from_L_ab_np(L_np, ab_np):
    """
    L_np: (H,W) in [0,100]
    ab_np: (H,W,2) in [-128,128] (or [-127,127])
    returns lab (H,W,3) float64
    """
    lab = np.concatenate([L_np[...,None], ab_np], axis=2).astype(np.float64)
    lab[...,0] = np.clip(lab[...,0], 0.0, 100.0)
    lab[...,1:] = np.clip(lab[...,1:], -128.0, 127.0)
    return lab

def safe_lab_to_rgb_uint8_from_tensors(pred_ab_tensor, input_L_tensor=None):
    """
    pred_ab_tensor: torch tensor (B,2,H,W) with values in [-1,1]
    input_L_tensor: torch tensor (B,1,H,W) with values in [0,1]
    returns list of uint8 RGB images (H,W,3)
    """
    pred_ab = pred_ab_tensor.detach().cpu().numpy()  # (B,2,H,W)
    B = pred_ab.shape[0]
    imgs = []
    for i in range(B):
        ab = np.transpose(pred_ab[i], (1,2,0)).astype(np.float64)  # (H,W,2)
        # scale ab from [-1,1] -> [-128,128)
        ab_scaled = ab * 128.0
        if input_L_tensor is None:
            raise ValueError("input_L_tensor required for converting ab-only predictions to RGB")
        L = input_L_tensor.detach().cpu().numpy()[i,0] * 100.0  # (H,W)
        lab = lab_from_L_ab_np(L, ab_scaled)
        # final safety clamp
        lab = np.clip(lab, [0.0, -128.0, -128.0], [100.0, 127.0, 127.0])
        rgb = color.lab2rgb(lab)  # float64 in [0,1]
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_uint8 = (rgb * 255.0).round().astype(np.uint8)
        imgs.append(rgb_uint8)
    return imgs

# -------------------------
# Training (GAN loop)
# -------------------------
def train(args):
    # datasets & loaders
    train_ds = ColorizationDataset(root_dir=args.dataset_root, img_size=args.size, augment=True, train=True, val_split=args.val_split)
    val_ds = ColorizationDataset(root_dir=args.dataset_root, img_size=args.size, augment=False, train=False, val_split=args.val_split)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    vloader = DataLoader(val_ds, batch_size=max(1,args.batch//2), shuffle=False, num_workers=2, pin_memory=True)

    # models
    G = UNetGen(in_channels=1, out_channels=2, base_filters=args.base_filters if hasattr(args,'base_filters') else 32, pretrained=True).to(DEVICE)
    D = PatchDiscriminator(in_channels=3, base=args.d_base).to(DEVICE)

    # optimizers
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # losses
    l1 = nn.L1Loss()
    adv_criterion = nn.BCEWithLogitsLoss()

    best_val = 1e9
    os.makedirs('checkpoints', exist_ok=True)

    # pretrain generator with L1 only (optional)
    if args.pretrain_epochs > 0:
        print(f"Pretraining G for {args.pretrain_epochs} epochs (L1 only)...")
        for epoch in range(1, args.pretrain_epochs+1):
            G.train()
            running = 0.0
            loop = tqdm(loader, desc=f"Pretrain Epoch {epoch}/{args.pretrain_epochs}")
            for inp, target, _ in loop:
                inp = inp.to(DEVICE); target = target.to(DEVICE)
                optG.zero_grad()
                pred = G(inp)  # pred in [-1,1]
                loss = l1(pred, target) * args.w_l1
                loss.backward()
                optG.step()
                running += loss.item()
                loop.set_postfix(loss=f"{running/ (loop.n+1):.4f}")
            torch.save(G.state_dict(), f"checkpoints/g_pretrained_epoch{epoch}.pth")
        print("Pretraining done.")

    # labels for adversarial
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(1, args.epochs+1):
        G.train(); D.train()
        t0 = time.time()
        loop = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (inp, target, _) in enumerate(loop):
            inp = inp.to(DEVICE)        # [B,1,H,W]
            target = target.to(DEVICE)  # [B,2,H,W]
            B = inp.size(0)

            # ---------------------
            # 1) Update Discriminator
            # ---------------------
            optD.zero_grad()
            # real pair: concat(L, ab_gt) -> [B,3,H,W]
            real_pair = torch.cat([inp, target], dim=1)
            pred_real = D(real_pair)
            real_labels = torch.full_like(pred_real, real_label, device=DEVICE)
            loss_D_real = adv_criterion(pred_real, real_labels)

            # fake pair: concat(L, ab_pred)
            with torch.no_grad():
                fake_ab = G(inp)  # [B,2,H,W] in [-1,1]
            fake_pair = torch.cat([inp, fake_ab], dim=1)
            pred_fake = D(fake_pair)
            fake_labels = torch.full_like(pred_fake, fake_label, device=DEVICE)
            loss_D_fake = adv_criterion(pred_fake, fake_labels)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optD.step()

            # ---------------------
            # 2) Update Generator
            # ---------------------
            optG.zero_grad()
            pred_ab = G(inp)  # [B,2,H,W]
            fake_pair_forG = torch.cat([inp, pred_ab], dim=1)
            pred_for_G = D(fake_pair_forG)
            # adversarial wants D(fake) -> real
            adv_target = torch.full_like(pred_for_G, real_label, device=DEVICE)
            loss_G_adv = adv_criterion(pred_for_G, adv_target)

            loss_G_l1 = l1(pred_ab, target)
            loss_G_tv = tv_loss(pred_ab) if args.w_tv > 0 else 0.0

            loss_G = args.w_l1 * loss_G_l1 + args.w_adv * loss_G_adv + args.w_tv * loss_G_tv
            loss_G.backward()
            optG.step()

            loop.set_postfix(D=f"{loss_D.item():.4f}", G=f"{loss_G.item():.4f}")

        t1 = time.time()
        print(f"Epoch {epoch} done. Time: {t1-t0:.1f}s. Saving checkpoints...")
        torch.save(G.state_dict(), f"checkpoints/g_epoch{epoch}.pth")
        torch.save(D.state_dict(), f"checkpoints/d_epoch{epoch}.pth")

        # simple validation (L1 and PSNR)
        G.eval()
        val_loss = 0.0
        psnrs = []
        with torch.no_grad():
            for inp, target, _ in vloader:
                inp = inp.to(DEVICE); target = target.to(DEVICE)
                pred = G(inp)  # [-1,1]
                val_loss += l1(pred, target).item()

                # compute PSNR on RGB conversion (batch loop)
                # use safe converter to reduce warnings
                rgb_list = safe_lab_to_rgb_uint8_from_tensors(pred, input_L_tensor=inp)
                gt_ab = (target.cpu().numpy() * 128.0)  # (B,2,H,W) scaled
                L_np = (inp.cpu().numpy() * 100.0)  # (B,1,H,W)
                for b in range(pred.size(0)):
                    gt_ab_b = np.transpose(gt_ab[b], (1,2,0))
                    L_b = np.transpose(L_np[b], (1,2,0))
                    gt_lab = np.concatenate([L_b, gt_ab_b], axis=2)
                    # convert gt rgb via safe path too
                    gt_rgb = color.lab2rgb(np.clip(gt_lab.astype(np.float64), [0,-128,-128], [100,127,127]))
                    gt_rgb = np.clip(gt_rgb, 0.0, 1.0)
                    # predicted rgb is uint8 in rgb_list[b]
                    pred_rgb = rgb_list[b].astype(np.float32) / 255.0
                    psnrs.append(metrics.peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=1.0))

        avg_val = val_loss / max(1, len(vloader))
        avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
        print(f"Val L1: {avg_val:.4f}, PSNR: {avg_psnr:.2f}dB")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(G.state_dict(), "checkpoints/g_best.pth")
            print("Saved best G checkpoint.")

# -------------------------
# Inference (use G only)
# -------------------------
def infer(args):
    G = UNetGen(in_channels=1, out_channels=2, base_filters=args.base_filters if hasattr(args,'base_filters') else 32, pretrained=True).to(DEVICE)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError("Checkpoint not found: " + args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    if isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', ''): v for k,v in ckpt.items()}
    G.load_state_dict(ckpt)
    G.eval()

    if os.path.isdir(args.input):
        exts = ('.jpg', '.png', '.jpeg', '.bmp', '.tif')
        files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(exts)])
    else:
        files = [args.input]
    os.makedirs(args.output, exist_ok=True) if args.output.endswith(os.sep) else os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    from skimage import color as skcolor
    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print("Cannot read", f)
            continue
        H, W = gray.shape
        gray_f = cv2.resize(gray, (args.size, args.size)).astype(np.float32) / 255.0
        inp_np = np.expand_dims(gray_f, axis=2)
        inp_t = torch.from_numpy(np.transpose(inp_np, (2,0,1))).unsqueeze(0).float().to(DEVICE)  # (1,1,H,W)
        with torch.no_grad():
            pred = G(inp_t)  # (1,2,H,W) in [-1,1]
        pred = torch.clamp(pred, -1.0, 1.0)
        # convert to RGB safely
        rgb_list = safe_lab_to_rgb_uint8_from_tensors(pred.cpu(), input_L_tensor=inp_t.cpu())
        rgb = rgb_list[0]
        rgb = cv2.resize(rgb, (W, H))
        out_path = os.path.join(args.output, os.path.splitext(os.path.basename(f))[0] + "_pix2pix_lite.png") if os.path.isdir(args.output) or args.output.endswith(os.sep) else args.output
        cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print("Wrote", out_path)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train','infer'], default='train')
    p.add_argument('--dataset_root', default='dataset')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--batch', type=int, default=6)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--pretrain_epochs', type=int, default=5, help="Train G with L1 only before GAN")
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--w_l1', type=float, default=1.0)
    p.add_argument('--w_adv', type=float, default=0.01)
    p.add_argument('--w_tv', type=float, default=1e-5)
    p.add_argument('--base_filters', type=int, default=32)
    p.add_argument('--d_base', type=int, default=32)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--checkpoint', type=str, default='checkpoints/g_best.pth')
    p.add_argument('--input', type=str, default=None)
    p.add_argument('--output', type=str, default='out/')
    args = p.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        assert args.input is not None
        infer(args)
