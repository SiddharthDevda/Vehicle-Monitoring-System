import cv2
import re
import numpy as np
import pytesseract

import os

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Substitution tables ───────────────────────────────────────────────────────
# When a character is in a LETTER position but looks like a digit
DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'E',
    '4': 'A', '5': 'S', '6': 'G', '8': 'B'
}
# When a character is in a DIGIT position but looks like a letter
LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'Z': '2', 'E': '3',
    'A': '4', 'S': '5', 'G': '6', 'B': '8',
    'D': '0', 'Q': '0', 'U': '0', 'J': '1'
}

# Valid Indian state codes (helps validate the first two chars)
INDIAN_STATE_CODES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HP","HR",
    "JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","PB",
    "PY","RJ","SK","TN","TR","TS","UK","UP","WB"
}

# Indian plate pattern
PLATE_PATTERN = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')


def preprocess_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 600:
        scale = max(2, 600 // w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure dark text on white background (standard Indian plate)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _fix_by_position(text):
    """Fix a string to match Indian plate format using position-based rules."""
    chars = list(text)
    n = len(chars)

    # 10-char: LL DD LL DDDD  e.g. MH20EE7602
    # 9-char:  LL DD L  DDDD  e.g. MH20E7602
    if n == 10:
        pos_types = ['L','L','D','D','L','L','D','D','D','D']
    elif n == 9:
        pos_types = ['L','L','D','D','L','D','D','D','D']
    else:
        return text  # can't fix non-standard lengths here

    corrected = []
    for ch, ptype in zip(chars, pos_types):
        if ptype == 'L' and ch.isdigit():
            corrected.append(DIGIT_TO_LETTER.get(ch, ch))
        elif ptype == 'D' and ch.isalpha():
            corrected.append(LETTER_TO_DIGIT.get(ch, ch))
        else:
            corrected.append(ch)
    return ''.join(corrected)


def correct_indian_plate(text):
    """
    Clean and correct OCR output to match Indian plate format.
    Returns corrected plate string, or "Not Found" if unusable.
    """
    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    if len(text) < 6:
        return "Not Found"

    # Already valid — return immediately
    if PLATE_PATTERN.match(text):
        # Extra validation: check state code is real
        if text[:2] in INDIAN_STATE_CODES:
            return text

    # Try position-based fix for 9 or 10 char strings
    if len(text) in (9, 10):
        fixed = _fix_by_position(text)
        if PLATE_PATTERN.match(fixed) and fixed[:2] in INDIAN_STATE_CODES:
            return fixed

    # If length is off (e.g. EasyOCR merged/split chars), try sliding window
    # to find the best matching 9 or 10-char substring
    for length in (10, 9):
        for start in range(len(text) - length + 1):
            candidate = text[start:start + length]
            fixed = _fix_by_position(candidate)
            if PLATE_PATTERN.match(fixed) and fixed[:2] in INDIAN_STATE_CODES:
                return fixed

    # Final fallback: return as-is if at least 6 chars, else Not Found
    return text if len(text) >= 6 else "Not Found"


def read_plate(image):
    """Run Tesseract OCR on a plate crop and return corrected text."""
    try:
        processed = preprocess_plate(image)
        # PSM 7 = single text line; PSM 8 = single word (try both)
        best = "Not Found"
        for psm in (7, 8, 6):
            config = (
                f'--psm {psm} '
                '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            raw = pytesseract.image_to_string(processed, config=config)
            candidate = correct_indian_plate(raw)
            # Prefer result that matches full pattern
            if PLATE_PATTERN.match(candidate) and candidate[:2] in INDIAN_STATE_CODES:
                return candidate
            if candidate != "Not Found" and len(candidate) > len(best):
                best = candidate
        return best
    except Exception as e:
        print("OCR error:", e)
        return "Not Found"