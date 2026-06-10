import cv2
import os
import re
import numpy as np

from services.detector import detect_vehicle, detect_plate, detect_seatbelt
from services.ocr import read_plate  as tesseract_read_plate, correct_indian_plate, PLATE_PATTERN, INDIAN_STATE_CODES
try:
    import easyocr
    reader = easyocr.Reader(['en'])
    EASYOCR_AVAILABLE = True

except Exception:
    EASYOCR_AVAILABLE = False
    reader = None



os.makedirs("outputs", exist_ok=True)

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]
YOLO_CAR_ID   = 2
YOLO_BUS_ID   = 5
YOLO_TRUCK_ID = 7
YOLO_MOTO_ID  = 3


# ── Reclassify bus → car ──────────────────────────────────────────────────────

def reclassify_vehicle(cls_id, conf, box, image_shape):
    name = {YOLO_CAR_ID: "car", YOLO_BUS_ID: "bus",
            YOLO_TRUCK_ID: "truck", YOLO_MOTO_ID: "motorcycle"}.get(cls_id)
    if name != "bus":
        return name
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    is_real_bus = (
        bw / img_w  >= 0.55 and
        bh / img_h  >= 0.40 and
        bw / max(bh, 1) >= 1.6 and
        conf        >= 0.60
    )
    if not is_real_bus:
        print(f"  Reclassifying 'bus' -> 'car' "
              f"(w={bw/img_w:.2f} h={bh/img_h:.2f} asp={bw/max(bh,1):.2f} conf={conf:.2f})")
        return "car"
    return "bus"


# ── Contour-based plate localizer ─────────────────────────────────────────────

def find_plate_by_contour(vehicle_crop, save_name=""):
    """
    Locate a licence-plate-shaped white/yellow rectangle in the crop
    using edge detection + contour filtering.
    Returns the plate sub-image, or None if not found.

    Works well for Indian plates (white background, black text, rectangular).
    """
    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    # Bilateral filter: smooths noise but keeps hard edges (plate border)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort by area descending; plate is usually a mid-to-large rectangle
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    img_h, img_w = vehicle_crop.shape[:2]
    img_area = img_h * img_w
    best_crop = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Plate should be between 0.5% and 25% of the vehicle crop area
        if area < img_area * 0.005 or area > img_area * 0.25:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

        # Accept 4-sided polygons (rectangles)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / max(h, 1)

        # Indian plate aspect ratio: roughly 2:1 to 5:1
        if aspect < 1.5 or aspect > 6.0:
            continue

        # Plate should sit in lower 75% of crop (not at the very top)
        if (y + h) < img_h * 0.20:
            continue

        # Prefer larger plates closer to expected aspect ~3.5
        score = area * (1 - abs(aspect - 3.5) / 3.5)
        if score > best_score:
            best_score = score
            padding = 8
            cx1 = max(0, x - padding)
            cy1 = max(0, y - padding)
            cx2 = min(img_w, x + w + padding)
            cy2 = min(img_h, y + h + padding)
            best_crop = vehicle_crop[cy1:cy2, cx1:cx2]

    if best_crop is not None and best_crop.size > 0:
        if save_name:
            cv2.imwrite(f"outputs/debug_contour_{save_name}.jpg", best_crop)
        print("  Plate found via contour detection")
        return best_crop

    return None


# ── EasyOCR helper ────────────────────────────────────────────────────────────

def easyocr_on_crop(crop, label=""):
    """Run EasyOCR; return best plate-pattern token or 'Not Found'."""
    if not EASYOCR_AVAILABLE or crop is None or crop.size == 0:
        return "Not Found"
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb, detail=0, paragraph=False)
    if not results:
        return "Not Found"

    best, best_score = "Not Found", 0
    for token in results:
        t = re.sub(r'[^A-Z0-9]', '', token.upper())
        if len(t) < 6:
            continue
        corrected = correct_indian_plate(t)
        if PLATE_PATTERN.match(corrected) and corrected[:2] in INDIAN_STATE_CODES:
            if label:
                print(f"  EasyOCR({label}): {corrected}")
            return corrected
        score = (100 if corrected[:2] in INDIAN_STATE_CODES else 0) + len(corrected)
        if score > best_score:
            best_score, best = score, corrected

    if label:
        print(f"  EasyOCR({label}) best-effort: {best}")
    return best

def read_plate_with_fallback(plate_crop, label=""):
    """Tesseract first, then EasyOCR."""

    text = tesseract_read_plate(plate_crop)

    if text != "Not Found" and len(text) >= 6:
        return text

    return easyocr_on_crop(plate_crop, label=label)



# ── YOLO plate detection ──────────────────────────────────────────────────────

def detect_plate_crop(vehicle_crop, vehicle_name):
    """
    Try YOLO plate detector at decreasing confidence.
    Returns the plate sub-image or None.
    """
    start_conf = 0.20 if vehicle_name == "bus" else 0.25
    for conf in [start_conf, 0.15, 0.10]:
        results = detect_plate(vehicle_crop, conf=conf)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        boxes = results[0].boxes.xyxy
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        px1, py1, px2, py2 = map(int, boxes[int(np.argmax(areas))])
        pad = 20
        py1 = max(0, py1-pad); px1 = max(0, px1-pad)
        py2 = min(vehicle_crop.shape[0], py2+pad)
        px2 = min(vehicle_crop.shape[1], px2+pad)
        crop = vehicle_crop[py1:py2, px1:px2]
        if crop.size > 0:
            cv2.imwrite(f"outputs/debug_plate_{vehicle_name}.jpg", crop)
            return crop
    return None


# ── Truck-specific OCR ────────────────────────────────────────────────────────

def ocr_truck(vehicle_crop):
    # 1. YOLO plate box
    for conf in [0.25, 0.15, 0.10]:
        results = detect_plate(vehicle_crop, conf=conf)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            px1, py1, px2, py2 = map(int, boxes[int(np.argmax(areas))])
            pad = 15
            crop = vehicle_crop[max(0,py1-pad):min(vehicle_crop.shape[0],py2+pad),
                                 max(0,px1-pad):min(vehicle_crop.shape[1],px2+pad)]
            if crop.size > 0:
                cv2.imwrite("outputs/debug_plate_truck.jpg", crop)
                text = read_plate_with_fallback(crop, "truck_yolo")
                if text != "Not Found":
                    return text
            break

    # 2. Contour on bottom strip
    h = vehicle_crop.shape[0]
    bottom = vehicle_crop[int(h * 0.55):, :]
    cv2.imwrite("outputs/debug_truck_bottom.jpg", bottom)

    contour_crop = find_plate_by_contour(bottom, save_name="truck_bottom")
    if contour_crop is not None:
        text = read_plate_with_fallback(contour_crop, "truck_contour")
        if text != "Not Found":
            return text

    # 3. EasyOCR on bottom strip with token filtering
    if EASYOCR_AVAILABLE:
        rgb = cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB)
        tokens = reader.readtext(rgb, detail=0, paragraph=False,
                                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        best = max((re.sub(r'[^A-Z0-9]','',t.upper()) for t in tokens),
                   key=lambda x: len(x) if 6 <= len(x) <= 11 else 0,
                   default="")
        if best:
            return correct_indian_plate(best)

    return "Not Found"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_image(image, video_mode=False):
    output = image.copy()
    results_data = []
    vehicle_results = detect_vehicle(image)

    if vehicle_results[0].boxes is None or len(vehicle_results[0].boxes) == 0:
        return {"image": output, "data": results_data}

    for box in vehicle_results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        coords = tuple(map(int, box.xyxy[0]))

        raw_name = vehicle_results[0].names[cls_id]
        if raw_name not in VEHICLE_CLASSES:
            continue

        vehicle_name = reclassify_vehicle(cls_id, conf, coords, image.shape)
        if vehicle_name is None:
            continue

        x1, y1, x2, y2 = coords
        vehicle_crop    = image[y1:y2, x1:x2]
        plate_text      = "Not Found"
        seatbelt_status = "Not Detected"

        # ── Plate reading ─────────────────────────────────────────────────
        # ── Plate reading ──

        if video_mode:
            plate_text = "Skipped"

            if vehicle_name in ["car", "truck", "bus"]:
                plate_crop = detect_plate_crop(vehicle_crop, vehicle_name)

                if plate_crop is not None:
                    plate_text = read_plate_with_fallback(
                        plate_crop,
                        vehicle_name
                    )

        elif vehicle_name == "truck":
            plate_text = ocr_truck(vehicle_crop)

        else:
            # Step 1: YOLO plate detection
            plate_crop = detect_plate_crop(vehicle_crop, vehicle_name)
            if plate_crop is not None:
                plate_text = read_plate_with_fallback(plate_crop, vehicle_name)

            # Step 2: YOLO missed → contour-based plate localization
            if plate_text == "Not Found":
                print(f"  YOLO plate miss for {vehicle_name} — trying contour localization")
                contour_crop = find_plate_by_contour(vehicle_crop,
                                                     save_name=vehicle_name)
                if contour_crop is not None:
                    plate_text = read_plate_with_fallback(contour_crop,
                                                          f"{vehicle_name}_contour")

            # Step 3: Contour also failed → EasyOCR on bottom half of vehicle
            if plate_text == "Not Found":
                print(f"  Contour miss — EasyOCR on bottom half")
                h = vehicle_crop.shape[0]
                bottom = vehicle_crop[h // 2:, :]
                cv2.imwrite(f"outputs/debug_{vehicle_name}_bottom.jpg", bottom)
                plate_text = easyocr_on_crop(bottom,
                                             label=f"{vehicle_name}_bottom")

        # ── Seatbelt ──────────────────────────────────────────────────────
        seatbelt_results = detect_seatbelt(vehicle_crop)
        if seatbelt_results[0].boxes is not None:
            for sb_box in seatbelt_results[0].boxes:
                sb_cls_id = int(sb_box.cls[0])
                sb_label  = seatbelt_results[0].names[sb_cls_id]
                if sb_cls_id == 1 or sb_label.lower() == "seatbelt":
                    seatbelt_status = "Detected"
                    break

        # ── Draw ──────────────────────────────────────────────────────────
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{vehicle_name} | {plate_text}"
        cv2.putText(output, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        results_data.append({
            "vehicle":  vehicle_name,
            "plate":    plate_text,
            "seatbelt": seatbelt_status,
        })

    return {"image": output, "data": results_data}