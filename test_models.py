from services.detector import vehicle_model, plate_model, seatbelt_model
from services.detector import seatbelt_model
import cv2
import numpy as np

print("Vehicle Classes:")
print(vehicle_model.names)

print("\nPlate Classes:")
print(plate_model.names)

print("\nSeatbelt Classes:")
print(seatbelt_model.names)

print("Seatbelt model classes:", seatbelt_model.names)

dummy = np.zeros((640, 640, 3), dtype=np.uint8)
try:
    vehicle_model(dummy, verbose=False)
    plate_model(dummy, verbose=False)
    seatbelt_model(dummy, verbose=False)
    print("\n✅ All models loaded successfully!")
except Exception as e:
    print(f"\n❌ Error: {e}")