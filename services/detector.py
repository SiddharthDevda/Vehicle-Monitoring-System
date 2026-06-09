import streamlit as st
from ultralytics import YOLO

# ── Cache models so they load ONCE and stay in memory across all reruns ────────
@st.cache_resource
def _load_models():
    plate_model    = YOLO("models/plate.pt")
    seatbelt_model = YOLO("models/seatbelt.pt")
    vehicle_model  = YOLO("yolov8n.pt")
    return plate_model, seatbelt_model, vehicle_model

def _get_models():
    return _load_models()

def detect_vehicle(image):
    _, _, vehicle_model = _get_models()
    return vehicle_model(image, conf=0.25)

def detect_plate(image, conf=0.25):
    plate_model, _, _ = _get_models()
    return plate_model(image, conf=conf)

def detect_seatbelt(image):
    _, seatbelt_model, _ = _get_models()
    return seatbelt_model(image, conf=0.25)