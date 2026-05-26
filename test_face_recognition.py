import os
import sys
import numpy as np
from PIL import Image
import urllib.request

# Ensure we can import modules from the Face_Detection directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'Face_Detection'))

print("Starting Face Recognition App Integration Tests...")

# URL of a sample face image from the official face_recognition test suite
TEST_IMAGE_URL = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/tests/test_images/obama.jpg"
TEMP_IMAGE_PATH = "obama_test.jpg"

try:
    print(f"Downloading sample test image from: {TEST_IMAGE_URL}")
    urllib.request.urlretrieve(TEST_IMAGE_URL, TEMP_IMAGE_PATH)
    print("Download successful.")
except Exception as e:
    print(f"Error downloading test image: {e}")
    sys.exit(1)

try:
    # 1. Test imports and environment
    import face_recognition
    import cv2
    from app import to_rgb_numpy, add_new_face, recognize_faces, known_face_encodings, known_face_names, save_known_faces
    
    print("\n--- Test 1: Image Conversion & Verification ---")
    img = Image.open(TEMP_IMAGE_PATH)
    print(f"Loaded image format: {img.format}, size: {img.size}, mode: {img.mode}")
    
    # Convert image
    img_np = to_rgb_numpy(img)
    print(f"Converted numpy shape: {img_np.shape}, dtype: {img_np.dtype}")
    assert img_np.ndim == 3 and img_np.shape[2] == 3, "Converted image must be 3D with 3 channels (RGB)"
    assert img_np.dtype == np.uint8, "Converted image must be uint8"
    print("Test 1 Passed: Image correctly loaded and converted to RGB uint8.")
    
    print("\n--- Test 2: Face Registration ---")
    name = "Barack Obama"
    # Ensure database starts empty in memory
    known_face_encodings.clear()
    known_face_names.clear()
    
    success = add_new_face(img, name)
    print(f"Registration result: {success}")
    assert success is True, "Face should be successfully detected and registered in the sample image"
    assert len(known_face_names) == 1, "Should have 1 registered face name"
    assert known_face_names[0] == name, f"Registered name should be {name}"
    print(f"Test 2 Passed: Face successfully detected and registered for {name}.")
    
    print("\n--- Test 3: Face Database Serialization ---")
    save_success = save_known_faces()
    print(f"Save database successful: {save_success}")
    assert save_success is True, "Database serialization should succeed"
    assert os.path.exists("Face_Detection/known_faces.pkl"), "Database file should exist on disk"
    print("Test 3 Passed: Database file successfully written to disk.")
    
    print("\n--- Test 4: Face Recognition & Bounding Boxes ---")
    # Recognize face in the same image
    annotated_image, count = recognize_faces(img_np, tolerance=0.6)
    print(f"Face recognized count: {count}")
    assert count == 1, "Should recognize 1 face in the image"
    print("Test 4 Passed: Face successfully recognized.")

    print("\nAll integration tests passed successfully!")
    
except AssertionError as ae:
    print(f"\n[FAIL] Test Assertion Failed: {ae}")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Test Failed with Exception: {e}")
    sys.exit(1)
finally:
    # Cleanup temp file
    if os.path.exists(TEMP_IMAGE_PATH):
        try:
            os.remove(TEMP_IMAGE_PATH)
        except Exception:
            pass
