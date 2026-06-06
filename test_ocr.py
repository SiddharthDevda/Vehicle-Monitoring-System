import cv2
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load a plate crop (if you have any saved debug image)
img = cv2.imread("outputs/debug_plate_car.jpg")  # or use any test image

if img is None:
    print("No debug image found. Please run the app first to generate debug_plate_*.jpg in outputs/ folder")
    exit()

result = ocr.ocr(img, cls=True)
print("OCR Result:", result)

if result and result[0]:
    texts = [line[1][0] for line in result[0]]
    print("Extracted texts:", texts)
else:
    print("No text detected")