import cv2
from services.detector import detect_seatbelt

img = cv2.imread("path_to_a_vehicle_image.jpg")  # replace with your image
results = detect_seatbelt(img)
print(results[0].boxes)
if results[0].boxes:
    for box in results[0].boxes:
        print(f"Class: {results[0].names[int(box.cls[0])]}, Confidence: {box.conf[0]}")