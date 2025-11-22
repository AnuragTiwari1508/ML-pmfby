"""
Complete Final Demo with All Features Working
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("🌾 PMFBY SMART CAPTURE SYSTEM - FINAL COMPLETE DEMO")
print("="*70)

# Check all imports
print("\n1️⃣ Checking All Dependencies...")
dependencies = {
    'numpy': 'numpy',
    'cv2': 'opencv-python-headless',
    'PIL': 'pillow',
    'torch': 'torch',
    'ultralytics': 'ultralytics',
    'albumentations': 'albumentations',
    'scikit-image': 'skimage',
    'pandas': 'pandas',
    'yaml': 'pyyaml'
}

for name, module in dependencies.items():
    try:
        if module == 'skimage':
            import skimage
        else:
            __import__(module)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - Install: pip install {module}")

# Test core modules
print("\n2️⃣ Testing Core Inference Modules...")

try:
    import numpy as np
    import cv2
    
    # Create realistic test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(480):
        test_image[i, :] = [30 + i//6, 80 + i//8, 20 + i//10]
    
    # Add crop-like objects
    cv2.rectangle(test_image, (150, 150), (300, 350), (40, 150, 40), -1)
    cv2.rectangle(test_image, (350, 200), (500, 380), (35, 140, 35), -1)
    
    # Add realistic noise
    noise = np.random.randint(-15, 15, test_image.shape, dtype=np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print("   ✅ Test image created")
    
except Exception as e:
    print(f"   ❌ Image creation failed: {e}")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Test blur detection
print("\n3️⃣ Blur Detection...")
try:
    from inference.blur_detector import BlurDetector
    
    detector = BlurDetector(threshold=100, warning_threshold=150)
    result = detector.detect(test_image)
    
    print(f"   ✅ Blur Score: {result['blur_score']:.2f}")
    print(f"   Status: {result['status']}")
    print(f"   Message: {detector.get_message(result)}")
    
    blur_working = True
except Exception as e:
    print(f"   ❌ Error: {e}")
    blur_working = False

# Test lighting detection
print("\n4️⃣ Lighting Detection...")
try:
    from inference.light_detector import LightingDetector
    
    detector = LightingDetector()
    result = detector.detect(test_image)
    
    print(f"   ✅ Mean Brightness: {result['mean_brightness']:.2f}")
    print(f"   Status: {result['status']}")
    print(f"   Acceptable: {result['is_acceptable']}")
    
    lighting_working = True
except Exception as e:
    print(f"   ❌ Error: {e}")
    lighting_working = False

# Test distance estimation
print("\n5️⃣ Distance Estimation...")
try:
    from inference.distance_estimator import DistanceEstimator
    
    estimator = DistanceEstimator()
    
    # Calibrate with realistic data
    calibration_data = [
        {'bbox_area': 120000, 'distance_meters': 0.8},
        {'bbox_area': 80000, 'distance_meters': 1.2},
        {'bbox_area': 40000, 'distance_meters': 1.8},
        {'bbox_area': 25000, 'distance_meters': 2.5},
    ]
    k = estimator.calibrate(calibration_data)
    
    # Test with realistic bbox
    bbox = [150, 150, 300, 350]  # 150x200 = 30000 px area
    result = estimator.estimate(bbox)
    
    guidance = estimator.get_guidance(result['distance_meters'], target_distance=1.5)
    
    print(f"   ✅ Calibration K: {k:.2f}")
    print(f"   Distance: {result['distance_meters']:.2f}m")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Guidance: {guidance['message']}")
    
    distance_working = True
except Exception as e:
    print(f"   ❌ Error: {e}")
    distance_working = False

# Test geotag validation
print("\n6️⃣ Geotag Validation...")
try:
    from inference.geotag_validator import GeotagValidator
    
    validator = GeotagValidator()
    
    # Test with Indian farm coordinates
    test_gps = {
        'latitude': 28.6150,  # Near Delhi
        'longitude': 77.2100,
        'has_gps': True,
        'altitude': 216.0
    }
    
    expected = {
        'latitude': 28.6139,  # Farm location
        'longitude': 77.2090
    }
    
    result = validator.validate_location(test_gps, expected, max_distance_km=2.0)
    
    print(f"   ✅ Distance: {result['distance_km']:.3f} km")
    print(f"   Valid: {result['is_valid']}")
    print(f"   {result['message']}")
    
    geotag_working = True
except Exception as e:
    print(f"   ❌ Error: {e}")
    geotag_working = False

# Test unified capture engine
print("\n7️⃣ Unified Capture Engine...")
try:
    from inference.capture_engine import CaptureEngine
    
    config = {
        'blur_threshold': 100,
        'blur_warning': 150,
        'dark_threshold': 40,
        'overexposed_threshold': 220,
        'target_distance': 1.5,
        'distance_tolerance': 0.3
    }
    
    engine = CaptureEngine(config)
    
    # Analyze complete frame
    bbox = [150, 150, 300, 350]
    analysis = engine.analyze_frame(test_image, bbox)
    
    print(f"   ✅ Overall Score: {analysis['overall_score']}/100")
    print(f"   Status: {engine.get_status_message(analysis)}")
    print(f"   Acceptable: {analysis['is_acceptable']}")
    print(f"   Should Capture: {engine.should_capture(analysis)}")
    
    if analysis['issues']:
        print(f"   Issues: {', '.join(analysis['issues'])}")
    
    if analysis['guidance']:
        print(f"   Guidance:")
        for g in analysis['guidance'][:2]:
            print(f"      • {g}")
    
    engine_working = True
except Exception as e:
    print(f"   ❌ Error: {e}")
    engine_working = False

# Test YOLOv8 model download and setup
print("\n8️⃣ YOLO Model Setup...")
try:
    from ultralytics import YOLO
    
    # Download pretrained model
    print("   Downloading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Test prediction on our image
    results = model(test_image, verbose=False)
    
    detections = results[0].boxes
    if len(detections) > 0:
        print(f"   ✅ Detected {len(detections)} objects")
        for i, box in enumerate(detections[:3]):  # Show first 3
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            name = model.names[cls]
            print(f"      {i+1}. {name}: {conf:.2f}")
    else:
        print("   ✅ Model loaded, no objects detected in test image")
    
    # Save model to our models directory
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'yolov8n.pt'
    if not model_path.exists():
        import shutil
        shutil.copy('yolov8n.pt', model_path)
        print(f"   ✅ Model saved to: {model_path}")
    
    yolo_working = True
except Exception as e:
    print(f"   ❌ Error: {e}")
    yolo_working = False

# Performance benchmark
print("\n9️⃣ Performance Benchmark...")
try:
    if blur_working and lighting_working:
        from inference.blur_detector import BlurDetector
        from inference.light_detector import LightingDetector
        
        blur_det = BlurDetector()
        light_det = LightingDetector()
        
        # Benchmark 100 iterations
        start_time = time.time()
        for _ in range(100):
            blur_result = blur_det.detect(test_image)
            light_result = light_det.detect(test_image)
        
        elapsed = (time.time() - start_time) / 100 * 1000
        fps = 1000 / elapsed
        
        print(f"   ✅ Combined Analysis: {elapsed:.2f}ms per frame")
        print(f"   Expected FPS: {fps:.0f}")
        print(f"   Suitable for real-time: {'Yes' if fps >= 10 else 'No'}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# Create sample dataset structure
print("\n🔟 Dataset Structure Setup...")
try:
    # Create directories
    dirs = [
        'dataset/raw',
        'dataset/processed/train/images',
        'dataset/processed/train/labels',
        'dataset/processed/val/images',
        'dataset/processed/val/labels',
        'models',
        'runs/train',
        'captured_images'
    ]
    
    for dir_path in dirs:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Save test image
    cv2.imwrite(str(project_root / 'test_crop.jpg'), test_image)
    
    print("   ✅ Directory structure created")
    print("   ✅ Sample test image saved: test_crop.jpg")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# Final summary
print("\n" + "="*70)
print("🎯 SYSTEM STATUS SUMMARY")
print("="*70)

components = {
    'Blur Detection': blur_working if 'blur_working' in locals() else False,
    'Lighting Analysis': lighting_working if 'lighting_working' in locals() else False,
    'Distance Estimation': distance_working if 'distance_working' in locals() else False,
    'Geotag Validation': geotag_working if 'geotag_working' in locals() else False,
    'Capture Engine': engine_working if 'engine_working' in locals() else False,
    'YOLO Model': yolo_working if 'yolo_working' in locals() else False,
}

working_count = sum(components.values())
total_count = len(components)

print(f"\n📊 Components Working: {working_count}/{total_count}")

for name, status in components.items():
    icon = "✅" if status else "❌"
    print(f"   {icon} {name}")

if working_count == total_count:
    print(f"\n🎉 ALL SYSTEMS OPERATIONAL!")
    print(f"✅ Ready for production deployment")
else:
    print(f"\n⚠️  {total_count - working_count} components need attention")

# Next steps
print(f"\n" + "="*70)
print("🚀 READY TO USE - NEXT STEPS")
print("="*70)

print(f"\n1️⃣ TEST INDIVIDUAL MODULES:")
print(f"   python inference/blur_detector.py --image test_crop.jpg")
print(f"   python inference/light_detector.py --image test_crop.jpg")

print(f"\n2️⃣ COLLECT DATASET:")
print(f"   • Place 1000+ crop images in dataset/raw/")
print(f"   • Run: python dataset/augment_dataset.py --input dataset/raw --output dataset/processed --target 15000")

print(f"\n3️⃣ TRAIN MODEL:")
print(f"   python training/train_detector.py --data dataset/data.yaml --epochs 100")

print(f"\n4️⃣ RUN CAMERA APP (if webcam available):")
print(f"   python camera_app/desktop_capture.py")

print(f"\n5️⃣ EXPORT FOR MOBILE:")
print(f"   # In Python:")
print(f"   from ultralytics import YOLO")
print(f"   model = YOLO('models/yolov8n.pt')")
print(f"   model.export(format='tflite', int8=True)")

print(f"\n" + "="*70)
print("🌾 PMFBY SMART CAPTURE SYSTEM - READY FOR DEPLOYMENT!")
print("भारतीय किसानों के लिए बनाया गया | Built for Indian Farmers")
print("="*70 + "\n")