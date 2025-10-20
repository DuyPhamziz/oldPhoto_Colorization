import os
import cv2

GRAY_DIR = "dataset/landscape Images/gray"
EDGE_DIR = "dataset/landscape Images/edges"
os.makedirs(EDGE_DIR, exist_ok=True)

#Duyet qua toan bo anh trong thu muc
for filename in os.listdir(GRAY_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(GRAY_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
        if img is None:
            print(f"Bo qua anh: {filename}")
            continue
        #lam muot de giam nhieu
        blur = cv2.GaussianBlur(img, (3,3), 0)
        #phat hien bien bang canny
        edges = cv2.Canny(blur, threshold1=50, threshold2=150)
        #Luu anh bien 
        save_path = os.path.join(EDGE_DIR, filename)
        cv2.imwrite(save_path, edges)
    
print("da tao ban do bien cho toan bo anh.")