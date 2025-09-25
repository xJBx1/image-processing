import cv2
import numpy as np

# --- Load your test image ---
frame = cv2.imread("drill.jpg")   # replace with your file

if frame is None:
    raise ValueError("Could not load image. Check filename/path.")

# --- Convert to grayscale ---
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)

# --- Threshold to isolate drill bit ---
_, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

result = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow("Masked", result)

# --- Find largest contour (drill bit) ---
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise ValueError("No drill bit found in image.")

c = max(contours, key=cv2.contourArea)

# --- Crop ---
x, y, w, h = cv2.boundingRect(c)
crop = result[y:y+h, x:x+w]
cv2.imshow("Cropped", crop)

# --- Rotate ---
rect = cv2.minAreaRect(c)
angle = rect[-1]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(crop, M, (w, h))

# --- Save & show ---
cv2.imwrite("drill_bit_processed.jpg", rotated)

cv2.imshow("Original", frame)
cv2.imshow("Processed", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
