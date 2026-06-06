import cv2
import sys
import os
from services.detector import detect_plate, detect_vehicle

# ── Change this to your actual truck image path ──────────────────────────────
IMAGE_PATH = "truck.jpg"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(
        f"Could not load image at '{IMAGE_PATH}'. "
        "Please update IMAGE_PATH in debug_truck.py to a valid image file."
    )

print(f"Image loaded: {image.shape[1]}x{image.shape[0]} px")

# ── Vehicle detection ─────────────────────────────────────────────────────────
vehicle_results = detect_vehicle(image)
annotated = image.copy()

if vehicle_results[0].boxes and len(vehicle_results[0].boxes) > 0:
    for box in vehicle_results[0].boxes:
        cls_id = int(box.cls[0])
        name = vehicle_results[0].names[cls_id]
        conf = float(box.conf[0])
        print(f"Detected vehicle: {name} (confidence {conf:.2f})")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{name} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:
    print("No vehicle detected!")

# ── Plate detection on full image (low confidence) ────────────────────────────
for conf_thresh in [0.25, 0.15, 0.10]:
    plate_results = detect_plate(image, conf=conf_thresh)
    if plate_results[0].boxes and len(plate_results[0].boxes) > 0:
        print(f"Found {len(plate_results[0].boxes)} plate(s) at conf={conf_thresh}")
        for i, pbox in enumerate(plate_results[0].boxes):
            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
            pconf = float(pbox.conf[0])
            print(f"  Plate {i}: ({px1},{py1}) -> ({px2},{py2})  conf={pconf:.2f}")
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(annotated, f"plate_{i} {pconf:.2f}", (px1, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Save each plate crop for manual inspection
            crop = image[py1:py2, px1:px2]
            crop_path = f"outputs/debug_plate_crop_{i}.jpg"
            cv2.imwrite(crop_path, crop)
            print(f"  Saved crop: {crop_path}")
        break  # Stop once we find plates at some confidence level
    else:
        print(f"No plates found at conf={conf_thresh}, trying lower...")
else:
    print("No plates found even at conf=0.10")

# ── Save annotated output ─────────────────────────────────────────────────────
out_path = "outputs/truck_debug_annotated.jpg"
cv2.imwrite(out_path, annotated)
print(f"\nSaved annotated image -> {out_path}")