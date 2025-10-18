import os # Thư viện để thao tác với hệ thống file
import cv2 # Thư viện xử lý ảnh
import numpy as np # Thư viện toán học
import matplotlib.pyplot as plt # Thư viện vẽ đồ thị

from glob import glob # Tìm tất cả các file ảnh trong thư mục dataset
from tqdm import tqdm # Hiển thị thanh tiến trình

import torch # Thư viện học sâu
import torch.nn as nn # Các lớp mạng nơ-ron
import torch.optim as optim # Các thuật toán tối ưu hóa
from torch.utils.data import Dataset, DataLoader # Quản lý dữ liệu
from torch.utils.tensorboard import SummaryWriter # Ghi log cho TensorBoard

import albumentations as A # Thư viện tăng cường dữ liệu
from albumentations.pytorch import ToTensorV2 # Chuyển đổi ảnh sang tensor PyTorch

import segmentation_models_pytorch as smp # Mô hình phân đoạn ảnh

import random # Thư viện tạo số ngẫu nhiên
from sklearn.model_selection import train_test_split # Chia dữ liệu thành tập huấn luyện và kiểm tra
from torchvision.utils import make_grid # Tạo lưới ảnh để hiển thị


try:
    import kornia # Thư viện xử lý ảnh nâng cao
    KORNIA_AVAILABLE = True
except Exception:
    KORNIA_AVAILABLE = False

# Kiểm tra xem có GPU không
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# model: U-net Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # lop tich chap
            nn.BatchNorm2d(out_channels), # chuan hoa loai batch
            nn.ReLU(inplace=True), # ham kich hoat ReLU, hoc phi tuyen
            # them mot lop conv nua
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_filters=32):
        super().__init__()
        f = base_filters
        self.encoder1 = DoubleConv(in_channels, f) # Encoder block 1
        self.pool1 = nn.MaxPool2d(2) # giam kich thuoc chieu cao va chieu rong di 2 lan
        self.encoder2 = DoubleConv(f, f*2) # Encoder block 2
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(f*2, f*4)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DoubleConv(f*4, f*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f*8, f*16) # Bottleneck trich xuat dac trung manh nhat truoc khi up-sampling

        self.up4 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2) # deconvolution tang kich thuoc anh len 2 lan
        # sau khi up-sampling, gop noi dung tu encoder tuong ung (up4 concate encoder4)
        self.dec4 = DoubleConv(f*16, f*8)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f*2, f)

        self.conv_last = nn.Conv2d(f, out_channels, kernel_size=1) # lop conv cuoi cung de chuyen ve so kenh mong muon

        def forward(self, x):
            # ma hoa
            e1 = self.encoder1(x); p1 = self.pool1(e1) # encoder1 + pool1 trich xuat dac trung, giam kich thuoc anh
            # e1 giu lai dac trung truoc khi giam kich thuoc dung cho skip connection, p1 la anh sau khi giam kich thuoc dung lam dau vao cho encoder2
            e2 = self.encoder2(p1); p2 = self.pool2(e2)
            e3 = self.encoder3(p2); p3 = self.pool3(e3)
            e4 = self.encoder4(p3); p4 = self.pool4(e4)

            b = self.bottleneck(p4) # bottleneck

            u4 = self.up4(b) # giai ma
            if u4.shape[2:] != e4.shape[2:]: # neu kich thuoc khong giong nhau thi can chinh lai
                e4 = F.interpolate(e4, size=u4.shape[2:], mode="bilinear", align_corners=True) # chinh kich thuoc e4 de gop noi voi up4
            d4 = self.dec4(torch.cat([u4, e4], dim=1)) # gop noi up4 va e4, sau do dua vao dec4
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

            out = self.conv_last(d1) # lop conv cuoi cung de chuyen ve so kenh mong muon
            return out

## DataSet Class
class ColorizationDataset(Dataset):
    """
        raw: dataset/raw/{name}.jpg
        gray: dataset/colorized/{name}.jpg
        edge: dataset/edge/{name}.jpg
    """
    def __ init__(self, list_file, root_dir="dataset", img_size=256, use_lab=true, augment=False): # khoi tao lop dataset, dataset xuất L channel làm input và ab làm target
        with open(list_file, "r") as f:
            self.file_names = [line.strip() for line in f if line.strip()] # đọc tên file từ list_file và lưu vào self.file_names
        self.root_dir = Path(root_dir) # thư mục gốc chứa dữ liệu
        self.img_size = img_size # kích thước ảnh
        self.use_lab = use_lab # sử dụng không gian màu LAB hay RGB
        self.augment = augment # có tăng cường dữ liệu hay không

        # albumentations pipeline cho tăng cường dữ liệu
        base = [
            A.Resize(img_size, img_size), # thay đổi kích thước ảnh
        ]

        if augment:
            aug_ops = [
                A.HorizontalFlip(p=0.5), # lật ngang với xác suất 0.5
                A.RandomRotate(limit=20, p=0.5), # xoay ảnh ngẫu nhiên trong giới hạn 20 độ với xác suất 0.5
                A.RandomBrightnessContrast(p=0.2), # thay đổi độ sáng và tương phản với xác suất 0.2
                A.GaussianNoise(var_limit=(5.0, 30.0), p=0.2), # thêm nhiễu Gaussian với xác suất 0.2
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5), # dịch chuyển, phóng to/thu nhỏ, xoay ảnh với xác suất 0.5
            ]
            base = base + aug_ops
        base += [A.Normalize(mean=(0.0,), std=(1.0, ToTensorV2())] # chuẩn hóa ảnh và chuyển sang tensor PyTorch
        self.transform = A.Compose(base) # tạo pipeline tăng cường dữ liệu

        def __len__(self): return len(self.names) # trả về số lượng ảnh trong dataset

        def __getitem__(self, idx):
            name = self.names[idx] # lấy tên file ảnh tại vị trí idx

            raw_p = self.root_dir / "raw" / f"{name}.jpg" # đường dẫn đến ảnh gốc
            gray_p = self.root_dir / "gray" / f"{name}.jpg" # đường dẫn đến ảnh xám
            edge_p = self.root_dir / "edge" / f"{name}.jpg" # đường dẫn đến ban do canh

            img_rgb = cv2.imread(str(raw_p)) # đọc ảnh gốc
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB) # chuyển đổi từ BGR sang RGB
            img_gray = cv2.imread(str(gray_p), cv2.IMREAD_GRAYSCALE) # đọc ảnh xám
            img_edge = cv2.imread(str(edge_p), cv2.IMREAD_GRAYSCALE) # đọc ảnh bản đồ cạnh

            if self.use_lab:
                lab = color.rgb2lab(img_rgb).astype("float32") # chuyển đổi ảnh RGB sang LAB: L in [0,100], a,b ~[-128,127]
                L = lab[...,0:1] # shape (H,W,1) lấy kênh L làm input 
                ab = lab[...,1:3] # shape (H,W,2) lấy kênh ab làm target

                L = L / 100.0 # chuẩn hóa kênh L về [0,1]
                ab = ab / 128.0 # chuẩn hóa kênh ab về [-1,1]
                tanget = ab # target là kênh ab
                inp_gray = (img_gray.astype("float32") / 255.0)[...,None] # chuẩn hóa ảnh xám về [0,1], shape (H,W,1)
            else:
                inp_gray = (img_gray.astype("float32") / 255.0)[...,None]
                target = img_rgb.astype("float32") / 255.0

            inp = np.concatenate([inp_gray, (img_edge.astype("float32") / 255.0)[...,None]], axis=2) # kết hợp ảnh xám và bản đồ cạnh làm input, shape (H,W,2)