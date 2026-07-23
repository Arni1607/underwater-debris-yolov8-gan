import os
import numpy as np
import cv2
from skimage import color

def compute_uciqe(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = color.rgb2lab(img_rgb / 255.0)
    l = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    chroma = (a**2 + b**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5
    top = int(round(0.01 * l.shape[0] * l.shape[1]))
    sl = np.sort(l, axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top]) - np.mean(sl[:top])
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0 or l1[i] == 0:
            satur.append(0)
        else:
            satur.append(chroma1[i] / l1[i])
    us = np.mean(satur)
    uciqe = c1 * sc + c2 * conl + c3 * us
    return uciqe

def compute_uiqm(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    rg = r - g
    yb = 0.5*(r + g) - b
    uicm = -0.0268 * np.sqrt(np.mean(rg**2) + np.mean(yb**2)) + 0.1586
    def sobel_sharpness(channel):
        sob = cv2.Sobel(channel, cv2.CV_64F, 1, 0) + cv2.Sobel(channel, cv2.CV_64F, 0, 1)
        return np.mean(np.abs(sob))
    uism = 0.299*sobel_sharpness(r) + 0.587*sobel_sharpness(g) + 0.114*sobel_sharpness(b)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    uiconm = np.std(gray) / (np.mean(gray) + 1e-6)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uiqm = c1*uicm + c2*uism + c3*uiconm
    return uiqm

def compute_entropy(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def compute_variance(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    variance = np.var(gray)
    return variance

def evaluate_folder(folder_path, folder_name):
    uciqe_scores = []
    uiqm_scores = []
    entropy_scores = []
    variance_scores = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filename in files:
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is None:
            continue
        uciqe_scores.append(compute_uciqe(img))
        uiqm_scores.append(compute_uiqm(img))
        entropy_scores.append(compute_entropy(img))
        variance_scores.append(compute_variance(img))
    print(f"\n{folder_name}")
    print(f"  Images evaluated: {len(uciqe_scores)}")
    print(f"  Average UCIQE:    {np.mean(uciqe_scores):.4f}")
    print(f"  Average UIQM:     {np.mean(uiqm_scores):.4f}")
    print(f"  Average Entropy:  {np.mean(entropy_scores):.4f}")
    print(f"  Average Variance: {np.mean(variance_scores):.4f}")
    return np.mean(uciqe_scores), np.mean(uiqm_scores), np.mean(entropy_scores), np.mean(variance_scores)

raw_folder = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\valid\images"
clahe_folder = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\valid_clahe\images"
gan_folder = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\valid_enhanced_resized\images"

print("=" * 60)
print("UNDERWATER IMAGE QUALITY ASSESSMENT")
print("=" * 60)

raw_uciqe, raw_uiqm, raw_ent, raw_var = evaluate_folder(raw_folder, "RAW IMAGES")
clahe_uciqe, clahe_uiqm, clahe_ent, clahe_var = evaluate_folder(clahe_folder, "CLAHE ENHANCED")
gan_uciqe, gan_uiqm, gan_ent, gan_var = evaluate_folder(gan_folder, "FUnIE-GAN ENHANCED")

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Condition':<25} {'UCIQE':>8} {'UIQM':>8} {'Entropy':>8} {'Variance':>10}")
print("-" * 60)
print(f"{'Raw images':<25} {raw_uciqe:>8.4f} {raw_uiqm:>8.4f} {raw_ent:>8.4f} {raw_var:>10.2f}")
print(f"{'CLAHE enhanced':<25} {clahe_uciqe:>8.4f} {clahe_uiqm:>8.4f} {clahe_ent:>8.4f} {clahe_var:>10.2f}")
print(f"{'FUnIE-GAN enhanced':<25} {gan_uciqe:>8.4f} {gan_uiqm:>8.4f} {gan_ent:>8.4f} {gan_var:>10.2f}")