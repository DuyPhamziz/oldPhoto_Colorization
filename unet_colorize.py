import os
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import torchvision.models as models
import time
import torch.nn.functional as F
from skimage import metrics, color
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.models import VGG16_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

import random
from pathlib import Path

try:
    import kornia
    KORNIA_AVAILABLE = True
except Exception:
    KORNIA_AVAILABLE = False

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -----------------------------
# Model
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_filters=32):
        super().__init__()
        f = base_filters
        self.encoder1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(f, f*2)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(f*2, f*4)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DoubleConv(f*4, f*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f*8, f*16)

        self.up4 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f*16, f*8)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f*2, f)

        self.conv_last = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x); p1 = self.pool1(e1)
        e2 = self.encoder2(p1); p2 = self.pool2(e2)
        e3 = self.encoder3(p2); p3 = self.pool3(e3)
        e4 = self.encoder4(p3); p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        if u4.shape[2:] != e4.shape[2:]:
            e4 = F.interpolate(e4, size=u4.shape[2:], mode="bilinear", align_corners=True)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)
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

        out = self.conv_last(d1)
        return out

# -----------------------------
# Dataset
# -----------------------------
class ColorizationDataset(Dataset):
    def __init__(self, root_dir="dataset", img_size=256, use_lab=True, augment=False, train=True, val_split=0.1):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.use_lab = use_lab
        self.augment = augment

        all_files = sorted([f.stem for f in (self.root_dir/'raw').glob('*.jpg')])
        split_idx = int(len(all_files)*(1-val_split))
        self.names = all_files[:split_idx] if train else all_files[split_idx:]

        base = [A.Resize(img_size, img_size)]
        if augment:
            aug_ops = [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(var_limit=(5,30), p=0.2),
                A.Affine(translate_percent=(0.05,0.05), scale=(0.95,1.05), rotate=(-15,15), p=0.5),
            ]
            base += aug_ops
        # Normalize for 2-channel input (gray + edge)
        base += [A.Normalize(mean=(0.0,)*2, std=(1.0,)*2), ToTensorV2()]
        # treat mask as image so multi-channel mask is handled
        self.transform = A.Compose(base, additional_targets={'mask': 'image'})

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        from skimage import color
        name = self.names[idx]
        raw_p = self.root_dir / "raw" / f"{name}.jpg"
        gray_p = self.root_dir / "gray" / f"{name}.jpg"
        edge_p = self.root_dir / "edge" / f"{name}.jpg"

        img_rgb = cv2.imread(str(raw_p))
        img_gray = cv2.imread(str(gray_p), cv2.IMREAD_GRAYSCALE)
        img_edge = cv2.imread(str(edge_p), cv2.IMREAD_GRAYSCALE)

        if img_rgb is None or img_gray is None or img_edge is None:
            print(f"Skipping corrupted/missing files for: {name}")
            return self.__getitem__((idx + 1) % len(self))

        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        if self.use_lab:
            lab = color.rgb2lab(img_rgb).astype(np.float32)
            L = lab[...,0:1] / 100.0
            ab = lab[...,1:3] / 128.0
            target = ab.astype(np.float32)
            inp_gray = (img_gray.astype(np.float32) / 255.0)[..., None]
        else:
            inp_gray = (img_gray.astype(np.float32) / 255.0)[..., None]
            target = img_rgb.astype(np.float32) / 255.0

        inp = np.concatenate([inp_gray, (img_edge.astype(np.float32) / 255.0)[..., None]], axis=2)  # H,W,2

        augmented = self.transform(image=inp, mask=target)
        inp_t = augmented['image'].float()   # [C,H,W]
        target_t = augmented['mask'].float() # [C,H,W]

        return inp_t, target_t, name

# -----------------------------
# Utilities
# -----------------------------
def psnr(gt, pred, data_range=1.0):
    return metrics.peak_signal_noise_ratio(gt, pred, data_range=data_range)

def sobel_torch_gray_tensor(x):
    if KORNIA_AVAILABLE:
        gx = kornia.filters.sobel(x)
        if gx.shape[1] == 2:
            mag = torch.sqrt(gx[:,0:1,:,:]**2 + gx[:,1:2,:,:]**2)
        else:
            mag = torch.abs(gx)
        return mag
    else:
        kernelx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], dtype=torch.float32).view(1,1,3,3).to(x.device)
        kernely = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], dtype=torch.float32).view(1,1,3,3).to(x.device)
        gx = F.conv2d(x, kernelx, padding=1)
        gy = F.conv2d(x, kernely, padding=1)
        mag = torch.sqrt(gx**2 + gy**2)
        return mag

class perceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)
        self.criterion = nn.L1Loss()
    def forward(self, pred, target):
        # pred/target should be 3-channel RGB images scaled appropriately before passing here
        return self.criterion(self.vgg(pred), self.vgg(target))

# -----------------------------
# Training
# -----------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = ColorizationDataset(root_dir=args.dataset_root, img_size=args.size, use_lab=True, augment=True, train=True)
    val_dataset = ColorizationDataset(root_dir=args.dataset_root, img_size=args.size, use_lab=True, augment=False, train=False)
    loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    vloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_channels=2, out_channels=2, base_filters=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss()
    # perceptual = perceptualLoss(device=device)  # optional

    best_val = 1e9
    os.makedirs('checkpoints', exist_ok=True)

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            t0 = time.time()

            loop = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
            for i, (inp, target, name) in enumerate(loop):
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad()
                pred = model(inp)

                loss_l1 = l1(pred, target)
                pred_ab_mag = torch.sqrt(pred[:,0:1,:,:]**2 + pred[:,1:2,:,:]**2)
                edge_input = inp[:,1:2,:,:]
                loss_edge = l1(pred_ab_mag, edge_input)

                loss = args.w_l1 * loss_l1 + args.w_edge * loss_edge
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                avg_loss = train_loss / (i + 1)
                loop.set_postfix(loss=f"{avg_loss:.4f}")

            t1 = time.time()
            print(f"Epoch [{epoch}/{args.epochs}] - Time: {t1 - t0:.2f}s - Loss: {avg_loss:.4f}")

            # validation
            model.eval()
            val_loss = 0.0
            val_psnr = []
            with torch.no_grad():
                for inp, target, name in tqdm(vloader, desc="Validation", leave=False):
                    inp = inp.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    pred = model(inp)

                    loss_l1 = l1(pred, target)
                    val_loss += loss_l1.item()

                    # convert once per batch
                    pred_ab = (pred * 128.0).cpu().numpy()    # B,2,H,W
                    L = (inp[:,0:1,:,:] * 100.0).cpu().numpy()# B,1,H,W
                    target_ab = (target * 128.0).cpu().numpy()

                    B = pred_ab.shape[0]
                    for b in range(B):
                        L_b = np.transpose(L[b], (1,2,0))
                        ab_b = np.transpose(pred_ab[b], (1,2,0))
                        gt_ab_b = np.transpose(target_ab[b], (1,2,0))

                        L_b = np.clip(L_b, 0.0, 100.0)
                        ab_b = np.clip(ab_b, -128.0, 127.0)
                        gt_ab_b = np.clip(gt_ab_b, -128.0, 127.0)

                        lab = np.concatenate([L_b, ab_b], axis=2)
                        gt_lab = np.concatenate([L_b, gt_ab_b], axis=2)

                        rgb = color.lab2rgb(lab)
                        gt_rgb = color.lab2rgb(gt_lab)

                        rgb = np.clip(rgb, 0.0, 1.0)
                        gt_rgb = np.clip(gt_rgb, 0.0, 1.0)

                        val_psnr.append(metrics.peak_signal_noise_ratio(gt_rgb, rgb, data_range=1.0))

            avg_val_loss = val_loss / max(1, len(vloader))
            avg_psnr = float(np.mean(val_psnr)) if val_psnr else 0.0
            print(f"Validation - Loss: {avg_val_loss:.4f} - PSNR: {avg_psnr:.2f}dB")

            if avg_val_loss < best_val:
                best_val = avg_val_loss
                torch.save(model.state_dict(), f"checkpoints/unet_colorization_best.pth")
                print(f"Saved Best Model with Loss: {best_val:.4f}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        torch.save(model.state_dict(), "checkpoints/unet_interrupted.pth")
        raise

# -----------------------------
# Inference
# -----------------------------
def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=2, out_channels=2, base_filters=32).to(device)

    # load checkpoint (accept plain state_dict saved with torch.save(model.state_dict(), path))
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # if a dict was saved as {'model_state': state_dict} handle it
    if isinstance(ckpt, dict) and ('model_state' in ckpt or 'state_dict' in ckpt):
        state = ckpt.get('model_state', ckpt.get('state_dict'))
    else:
        state = ckpt
    # fix keys if saved from DataParallel
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Read input
    inp_path = args.input
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Input not found: {inp_path}")

    # If input is directory, process first file only (or you can extend to loop)
    if os.path.isdir(inp_path):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        files = sorted([os.path.join(inp_path, f) for f in os.listdir(inp_path) if f.lower().endswith(exts)])
        if len(files) == 0:
            raise RuntimeError(f"No image files found in directory: {inp_path}")
        files_to_process = files
    else:
        files_to_process = [inp_path]

    # Determine if args.output is directory or file pattern
    out_arg = args.output
    out_is_dir = out_arg.endswith(os.sep) or os.path.isdir(out_arg) or len(files_to_process) > 1
    if out_is_dir:
        os.makedirs(out_arg, exist_ok=True)
    else:
        # ensure parent dir exists
        parent = os.path.dirname(out_arg) or '.'
        os.makedirs(parent, exist_ok=True)

    from skimage import color as skcolor

    for fpath in files_to_process:
        fname = os.path.basename(fpath)
        gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Warning: cannot read {fpath}, skipping.")
            continue
        H, W = gray.shape

        edge = cv2.Canny(gray, 100, 200)
        gray_f = cv2.resize(gray, (args.size, args.size)).astype(np.float32) / 255.0
        edge_f = cv2.resize(edge, (args.size, args.size)).astype(np.float32) / 255.0
        inp_np = np.stack([gray_f, edge_f], axis=2)
        inp_t = torch.from_numpy(np.transpose(inp_np, (2,0,1))).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(inp_t)

        # clamp pred if model trained normalized to [-1,1]
        pred = torch.clamp(pred, -1.0, 1.0)
        pred_ab = (pred * 128.0).cpu().numpy()[0]  # 2,H,W
        ab_img = np.transpose(pred_ab, (1,2,0))    # H,W,2

        L_img = cv2.resize(gray, (args.size, args.size)).astype(np.float32)/255.0 * 100.0
        lab_img = np.concatenate([L_img[...,None], ab_img], axis=2)

        # clip LAB to valid ranges
        lab_img[...,0] = np.clip(lab_img[...,0], 0.0, 100.0)
        lab_img[...,1] = np.clip(lab_img[...,1], -128.0, 127.0)
        lab_img[...,2] = np.clip(lab_img[...,2], -128.0, 127.0)

        rgb = skcolor.lab2rgb(lab_img)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255.0).astype(np.uint8)
        rgb = cv2.resize(rgb, (W, H))

        # Build output paths
        if out_is_dir:
            out_path = os.path.join(out_arg, os.path.splitext(fname)[0] + '_colorized.png')
        else:
            out_path = out_arg

        # ensure parent dir
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

        ok = cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"Attempted write -> {os.path.abspath(out_path)}  ; cv2.imwrite returned {ok}")
        if not ok:
            print("Warning: cv2.imwrite returned False. Check permissions / disk space / path validity.")

        # write comparison image (gray | color)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        combined = np.concatenate([gray_rgb, rgb], axis=1)
        cmp_path = out_path.replace('.png', '_compare.png') if out_path.lower().endswith('.png') else out_path + '_compare.png'
        ok2 = cv2.imwrite(cmp_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Comparison write -> {os.path.abspath(cmp_path)}  ; cv2.imwrite returned {ok2}")
        if not ok2:
            print("Warning: failed to write comparison image.")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='train', choices=['train','infer'])
    p.add_argument('--dataset_root', type=str, default='dataset')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--w_l1', type=float, default=1.0)
    p.add_argument('--w_edge', type=float, default=1.0)
    p.add_argument('--checkpoint', type=str, default='checkpoints/unet_colorization_best.pth')
    p.add_argument('--input', type=str, default=None)
    p.add_argument('--output', type=str, default='out.png')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'train':
        train(args)
    else:
        assert args.input is not None
        infer(args)
