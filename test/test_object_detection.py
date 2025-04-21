import numpy as np
from backend_service.object_detection import preprocess_image, detect_objects
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_preprocess_image_shape():
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed = preprocess_image(dummy_image)
    assert processed.shape == (1, 3, 640, 640), "Processed image shape mismatch"

def test_detect_objects_blank_image():
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    result_image, detections = detect_objects(blank)
    assert result_image is not None, "No image returned"
    assert isinstance(detections, list), "Detections should be a list"

def test_detect_objects_random_image():
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    result_image, detections = detect_objects(dummy_image)
    assert result_image is not None, "No image returned from detection"
    assert isinstance(detections, list), "Detections must be a list"
