import cv2
import numpy as np

# --- Load original image ---
img = cv2.imread("drill.jpg")
if img is None:
    raise ValueError("Could not load image.")

H_orig, W_orig = img.shape[:2]

# --- Downscale for detection ---
max_side = 1000  # downscale largest side to 1000 px
scale = max_side / max(H_orig, W_orig)
if scale < 1:
    img_small = cv2.resize(img, (int(W_orig*scale), int(H_orig*scale)), interpolation=cv2.INTER_AREA)
else:
    img_small = img.copy()
H_small, W_small = img_small.shape[:2]

# --- Focus detection on small image ---
gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(gray, cv2.CV_64F)
lap_abs = cv2.convertScaleAbs(lap)
lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX)
_, mask = cv2.threshold(lap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise ValueError("No drill bit found.")

c = max(contours, key=cv2.contourArea)
x_s, y_s, w_s, h_s = cv2.boundingRect(c)

# Drill bit center in small image
cx_s = x_s + w_s // 2
cy_s = y_s + h_s // 2

# Map back to original image coordinates
cx = int(cx_s / scale)
cy = int(cy_s / scale)

# --- Crop original image around center ---
crop_w = int(0.22 * W_orig)
crop_h = int(0.22 * H_orig)

x1 = max(cx - crop_w // 2, 0)
y1 = max(cy - crop_h // 2, 0)
x2 = min(cx + crop_w // 2, W_orig)
y2 = min(cy + crop_h // 2, H_orig)

crop = img[y1:y2, x1:x2]

cv2.imwrite("drill_bit_crop_highres.jpg", crop)
cv2.imshow("Cropped Drill Bit", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
