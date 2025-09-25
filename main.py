import cv2
import numpy as np
import serial
import time

# --- 1. Setup serial communication with ESP32 ---
ser = serial.Serial('COM3', 115200, timeout=1)  

time.sleep(2)  # Wait for serial to initialize

while True:
    if ser.in_waiting > 0:
        msg = ser.readline().decode('utf-8').strip()
        print(f"Received: {msg}")

        if msg == "CAPTURE":   # Trigger word from ESP32
            print("Capturing drill bit...")

            # --- 2. Capture image from Pi Camera ---
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("Error: Failed to capture image")
                continue

            cv2.imwrite("drill_bit_raw.jpg", frame)

            # --- 3. Convert to grayscale ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- 4. Threshold to isolate drill bit ---
            _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

            result = cv2.bitwise_and(frame, frame, mask=mask)

            # --- 5. Find largest contour ---
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("Error: No drill bit found")
                continue

            c = max(contours, key=cv2.contourArea)

            # --- 6. Crop ---
            x, y, w, h = cv2.boundingRect(c)
            crop = result[y:y+h, x:x+w]

            # --- 7. Rotate ---
            rect = cv2.minAreaRect(c)
            angle = rect[-1]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(crop, M, (w, h))

            # --- 8. Save final processed image ---
            cv2.imwrite("drill_bit_final.jpg", rotated)

            print("Drill bit captured, cropped, rotated, and saved as drill_bit_final.jpg")
            ser.write(b"DONE\n")  # Notify ESP32 that processing is done
