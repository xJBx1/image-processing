import cv2
import numpy as np
import os

# --- Input & output folders ---
input_folder = "images/"
square_folder = "processed_square/"
circle_folder = "processed_circle/"
os.makedirs(square_folder, exist_ok=True)
os.makedirs(circle_folder, exist_ok=True)

# --- Process all images ---
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {filename}")
        continue

    H_orig, W_orig = img.shape[:2]

    # --- Downscale for detection ---
    max_side = 1000
    scale = max_side / max(H_orig, W_orig)
    if scale < 1:
        img_small = cv2.resize(img, (int(W_orig*scale), int(H_orig*scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()
    H_small, W_small = img_small.shape[:2]

    # --- Focus detection ---
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = cv2.convertScaleAbs(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(lap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No drill bit found in {filename}")
        continue

    # --- Largest contour = drill bit ---
    c = max(contours, key=cv2.contourArea)
    x_s, y_s, w_s, h_s = cv2.boundingRect(c)
    cx_s = x_s + w_s // 2
    cy_s = y_s + h_s // 2

    # Map back to original image
    cx = int(cx_s / scale)
    cy = int(cy_s / scale)

    # --- Crop size = 22% of original image (square) ---
    crop_size = int(0.22 * min(W_orig, H_orig))

    # --- Center the drill bit in the crop ---
    x1 = cx - crop_size // 2
    y1 = cy - crop_size // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # --- Clamp to image boundaries ---
    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    if x2 > W_orig:
        x1 -= x2 - W_orig
        x2 = W_orig
    if y2 > H_orig:
        y1 -= y2 - H_orig
        y2 = H_orig

    # --- Square crop ---
    crop_square = img[y1:y2, x1:x2]
    save_square = os.path.join(square_folder, os.path.splitext(filename)[0] + "_square.png")
    cv2.imwrite(save_square, crop_square)

    # --- Circular crop ---
    h_crop, w_crop = crop_square.shape[:2]
    mask_circle = np.zeros((h_crop, w_crop), dtype=np.uint8)
    center = (w_crop // 2, h_crop // 2)
    radius = min(h_crop, w_crop) // 2
    cv2.circle(mask_circle, center, radius, 255, -1)

    # Add alpha channel for transparency
    b, g, r = cv2.split(crop_square)
    alpha = mask_circle
    crop_circle = cv2.merge([b, g, r, alpha])

    save_circle = os.path.join(circle_folder, os.path.splitext(filename)[0] + "_circle.png")
    cv2.imwrite(save_circle, crop_circle)

    print(f"Processed {filename}")
