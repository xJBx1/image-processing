import cv2
import numpy as np

# --- Load image ---
img = cv2.imread("drill.jpg")
if img is None:
    raise ValueError("Could not load image. Check path/filename.")

# --- Convert to grayscale ---
# h1,w1 = img.shape[:2]
# ix = w1//2
# iy = h1//2
# img = img[iy-500:iy+500, ix-500:ix+500]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Focus measure: Laplacian ---
lap = cv2.Laplacian(gray, cv2.CV_64F)
lap_abs = cv2.convertScaleAbs(lap)

# --- Normalize and threshold ---
# Pixels with high Laplacian = sharp (drill), low = blurry (background)
lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX)
_, mask = cv2.threshold(lap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Clean up mask ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# --- Keep largest sharp region (the drill bit) ---
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise ValueError("No sharp region found.")
c = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(c)

# --- Crop to drill bit ---
h_img, w_img = img.shape[:2]
pad_h = int(0.22 * h_img)
pad_w = int(0.22 * w_img)
if w > pad_h or h > pad_w:
    crop = img[y:y+h, x:x+w]
else:
    cx = x + w // 2
    cy = y + h // 2
    x1 = max(0, cx - pad_w // 2)
    y1 = max(0, cy - pad_h // 2)
    x2 = min(w_img, cx + pad_w // 2)
    y2 = min(h_img, cy + pad_h // 2)
    crop = img[y1:y2, x1:x2]
#crop = img[y:y+h, x:x+w]
print("image size", img.shape)
print("crop size", crop.shape)
# --- Save & show ---
cv2.imwrite("drill_cropped.jpg", crop)
cv2.imshow("Cropped Drill Bit", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
