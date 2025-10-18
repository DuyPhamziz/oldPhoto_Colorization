import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
DATA_DIR = "datasets/landscape Images"
GRAY_DIR = os.path.join(DATA_DIR, "gray")
COLOR_DIR = os.path.join(DATA_DIR, "color")
SAVE_DIR = "preprocessed"
IMG_SIZE = 256

os.makedirs(SAVE_DIR, exist_ok=True)

def preprocess_pair(gray_path, color_path):
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(color_path)
    
    if gray is None or color is None:
        print(f"Bỏ qua ảnh : {gray_path}")
        return None, None
    
    #thong nhat kich thuoc 256x256
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))   
    color = cv2.resize(color, (IMG_SIZE, IMG_SIZE))
    
    #lam muot anh
    color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
    
    #tach kenh do sang va mau
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0]
    ab = lab[:,:,1:]
    
    #cang bang sang va tuong phan
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    
    #chuan hoa gia tri fixel ve khoang[0,1]
    L = L / 255.0
    ab = (ab + 128) / 255.0
    
    #them chieu cho L
    L = np.expand_dims(L, axis=-1)
    return L, ab

#duyet toan bo anh
gray_files = sorted(os.listdir(GRAY_DIR))
color_files = sorted(os.listdir(COLOR_DIR))

L_list , ab_list = [], []

#duyet tung cap ten tuong ung
for g, c in zip(gray_files, color_files):
    gray_path = os.path.join(GRAY_DIR, g)
    color_path = os.path.join(COLOR_DIR, c)
    L, ab = preprocess_pair(gray_path, color_path)
    if L is not None:
        L_list.append(L)
        ab_list.append(ab)
        
        
#chuyen danh sach sang mang numpy de luu
L_array = np.array(L_list)
ab_array = np.array(ab_list)

# luu du lieu da su ly
np.save(os.path.join(SAVE_DIR, "L.npy"), L_array)
np.save(os.path.join(SAVE_DIR, "ab.npy"), ab_array)
print(f"✅ Đã xử lý {len(L_array)} ảnh và lưu vào thư mục '{SAVE_DIR}'")

#tang cung du lieu
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)