# 🚀 PMFBY Smart Capture - Step-by-Step Implementation Guide

## 📋 Table of Contents
1. [Setup & Installation](#setup)
2. [Quick Start (5 minutes)](#quick-start)
3. [Training Your Own Model](#training)
4. [Dataset Preparation](#dataset)
5. [Mobile Integration](#mobile)
6. [Production Deployment](#deployment)

---

## 🛠️ Setup & Installation {#setup}

### Prerequisites
- Python 3.8+
- Webcam/Camera access
- 8GB RAM minimum
- GPU optional (for training)

### Installation Steps

```bash
# 1. Clone/Navigate to project
cd /workspaces/ML-pmfby

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import cv2, numpy, torch; print('✅ All imports successful')"
```

---

## ⚡ Quick Start (5 minutes) {#quick-start}

### Test Individual Modules

#### 1. Test Blur Detection
```bash
# Download a test image first
curl -o test_crop.jpg "https://source.unsplash.com/800x600/?crop,field"

# Run blur detection
python inference/blur_detector.py --image test_crop.jpg --show
```

**Expected Output:**
```
🔍 Analyzing: test_crop.jpg

✅ Image is sharp (score: 234.5)

Details:
  Blur Score: 234.52
  Status: sharp
  Is Blurry: False
  Needs Warning: False
```

#### 2. Test Lighting Detection
```bash
python inference/light_detector.py --image test_crop.jpg --show
```

#### 3. Launch Desktop Camera App (WITHOUT trained model)
```bash
# Works immediately - uses blur and lighting checks only
python camera_app/desktop_capture.py

# Controls:
#   SPACE - Capture image
#   M     - Toggle multi-angle mode
#   G     - Toggle guidance
#   Q     - Quit
```

You'll see real-time overlay showing:
- ✓ Blur score
- ✓ Lighting status
- ✓ Capture guidance

---

## 🎓 Training Your Own Model {#training}

### Phase 1: Prepare Dataset (2-3 days)

#### Option A: Use Public Crop Dataset (Fastest)

```bash
# 1. Download Kaggle crop disease dataset
# Visit: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
# Download and extract to dataset/raw/

# 2. Or use sample images
mkdir -p dataset/raw/train/images
mkdir -p dataset/raw/train/labels

# Collect 100-500 images manually
# Take photos of crops at different angles, distances, lighting
```

#### Option B: Auto-collect from Camera

```bash
# Create collection script
python -c "
import cv2
import time
from pathlib import Path

output_dir = Path('dataset/raw/collected')
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print('Press SPACE to capture, Q to quit')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Collect Images', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        filename = f'img_{count:04d}.jpg'
        cv2.imwrite(str(output_dir / filename), frame)
        print(f'Saved: {filename}')
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f'Collected {count} images')
"
```

#### Step 2: Annotate Images

```bash
# Install LabelImg
pip install labelImg

# Launch annotation tool
labelimg dataset/raw/train/images/ --labels crop,damage,plant,field

# Instructions:
# 1. Press 'w' to create box
# 2. Draw around crop/plant
# 3. Select class (crop, damage, plant, field)
# 4. Press 'd' for next image
# 5. Save in YOLO format
```

#### Step 3: Augment to 15k

```bash
# Create annotations CSV first
python -c "
import pandas as pd
from pathlib import Path
import cv2

# Example: Convert YOLO labels to CSV
data = []
img_dir = Path('dataset/raw/train/images')
label_dir = Path('dataset/raw/train/labels')

for img_path in img_dir.glob('*.jpg'):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    label_path = label_dir / f'{img_path.stem}.txt'
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                
                # Convert YOLO format to Pascal VOC
                xmin = int((x_center - width/2) * w)
                ymin = int((y_center - height/2) * h)
                xmax = int((x_center + width/2) * w)
                ymax = int((y_center + height/2) * h)
                
                data.append({
                    'filename': img_path.name,
                    'width': w,
                    'height': h,
                    'class': int(cls),
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })

df = pd.DataFrame(data)
df.to_csv('dataset/raw/annotations.csv', index=False)
print(f'Created annotations for {len(df)} boxes')
"

# Now augment
python dataset/augment_dataset.py \
    --input dataset/raw/train/images \
    --output dataset/processed/train \
    --annotations dataset/raw/annotations.csv \
    --target 15000
```

#### Step 4: Prepare YOLO Dataset

```bash
# Split into train/val
python -c "
from pathlib import Path
import shutil
import random

processed = Path('dataset/processed/train')
images = list(processed.glob('*.jpg'))
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

# Create directories
Path('dataset/yolo/images/train').mkdir(parents=True, exist_ok=True)
Path('dataset/yolo/images/val').mkdir(parents=True, exist_ok=True)
Path('dataset/yolo/labels/train').mkdir(parents=True, exist_ok=True)
Path('dataset/yolo/labels/val').mkdir(parents=True, exist_ok=True)

print(f'Moving {len(train_imgs)} train, {len(val_imgs)} val images...')
# TODO: Move images and labels to YOLO structure
"

# Create dataset YAML
python training/train_detector.py create-yaml \
    --train dataset/yolo/images/train \
    --val dataset/yolo/images/val \
    --classes crop damage plant field \
    --output dataset/crop_data.yaml
```

#### Step 5: Train Model

```bash
# Start training (will take 2-8 hours depending on GPU)
python training/train_detector.py train \
    --data dataset/crop_data.yaml \
    --epochs 100 \
    --batch 16 \
    --model-size n \
    --device 0 \
    --name pmfby_crop_v1

# Training will save to: runs/train/pmfby_crop_v1/
# Best model: runs/train/pmfby_crop_v1/weights/best.pt
```

#### Step 6: Export to TFLite

```bash
# Export for mobile
python training/train_detector.py export \
    --weights runs/train/pmfby_crop_v1/weights/best.pt \
    --format tflite \
    --int8 \
    --img-size 640

# Exported model: runs/train/pmfby_crop_v1/weights/best.tflite
```

#### Step 7: Test Trained Model

```bash
# Copy model to models directory
cp runs/train/pmfby_crop_v1/weights/best.pt models/yolov8_crop.pt

# Test on image
python inference/object_detector.py \
    --image test_crop.jpg \
    --model models/yolov8_crop.pt \
    --show

# Launch camera with detection
python camera_app/desktop_capture.py \
    --model models/yolov8_crop.pt
```

---

## 📊 Dataset Preparation {#dataset}

### Recommended Dataset Structure

```
dataset/
├── raw/                        # Original images
│   ├── train/
│   │   ├── images/            # Raw training images
│   │   └── labels/            # YOLO format labels
│   └── annotations.csv        # Or CSV annotations
├── processed/                 # Augmented images
│   ├── train/
│   └── val/
└── yolo/                      # YOLO training format
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── crop_data.yaml         # Dataset config
```

### Annotation CSV Format

```csv
filename,width,height,class,xmin,ymin,xmax,ymax,latitude,longitude,timestamp_utc,device_model,blur_score,light_flag,angle_pitch,angle_yaw,distance_m,multi_angle_group
IMG_001.jpg,4032,3024,0,400,800,3200,2600,22.7196,75.8577,2025-11-22T07:12:03Z,MiA3,356.2,ok,2.5,-1.2,1.8,group_001
IMG_002.jpg,4032,3024,1,500,900,3100,2500,22.7198,75.8579,2025-11-22T07:12:15Z,MiA3,412.8,ok,1.8,-0.8,1.5,group_001
```

### Labeling Tools

1. **LabelImg** (Desktop, Free)
   ```bash
   pip install labelImg
   labelimg
   ```

2. **CVAT** (Web-based, Team collaboration)
   - Visit: https://www.cvat.ai/
   - Create project, upload images, annotate

3. **Roboflow** (Cloud, Auto-augmentation)
   - Visit: https://roboflow.com/
   - Upload dataset, annotate, export YOLO format

---

## 📱 Mobile Integration {#mobile}

### Android Integration

#### 1. Export TFLite Model
```bash
python training/train_detector.py export \
    --weights models/yolov8_crop.pt \
    --format tflite \
    --int8
```

#### 2. Android Project Setup (Kotlin + CameraX)

Create `SmartCaptureSDK.kt`:

```kotlin
import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class SmartCaptureEngine(context: Context) {
    private val tflite: Interpreter
    
    init {
        val model = loadModelFile(context, "yolov8_crop.tflite")
        tflite = Interpreter(model)
    }
    
    fun validateCapture(bitmap: Bitmap): CaptureResult {
        // 1. Check blur
        val blurScore = checkBlur(bitmap)
        
        // 2. Check lighting
        val lightStatus = checkLighting(bitmap)
        
        // 3. Run detection
        val detections = runDetection(bitmap)
        
        return CaptureResult(
            isValid = blurScore > 100 && lightStatus == "ok" && detections.isNotEmpty(),
            blurScore = blurScore,
            lightStatus = lightStatus,
            detections = detections
        )
    }
    
    private fun checkBlur(bitmap: Bitmap): Double {
        // Implement Laplacian variance
        // (Convert Python blur_detector.py logic)
        return 150.0
    }
    
    private fun runDetection(bitmap: Bitmap): List<Detection> {
        // Run TFLite inference
        val input = preprocessImage(bitmap)
        val output = Array(1) { FloatArray(25200 * 85) }
        tflite.run(input, output)
        return postprocess(output)
    }
}
```

#### 3. Camera Activity
```kotlin
class CaptureActivity : AppCompatActivity() {
    private lateinit var captureEngine: SmartCaptureEngine
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        captureEngine = SmartCaptureEngine(this)
        
        startCamera()
    }
    
    private fun analyzeFrame(image: ImageProxy) {
        val bitmap = image.toBitmap()
        val result = captureEngine.validateCapture(bitmap)
        
        runOnUiThread {
            updateUI(result)
        }
    }
}
```

### iOS Integration (Swift + AVFoundation)

Similar approach - export CoreML model:
```bash
python training/train_detector.py export \
    --weights models/yolov8_crop.pt \
    --format coreml
```

---

## 🚀 Production Deployment {#deployment}

### Server-side Processing (Optional)

```python
# api/server.py
from flask import Flask, request, jsonify
from inference.capture_engine import CaptureEngine
import cv2
import numpy as np

app = Flask(__name__)
engine = CaptureEngine('config.yaml')

@app.route('/validate', methods=['POST'])
def validate():
    file = request.files['image']
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    result = engine.validate_capture(img)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 🎯 Performance Optimization

### 1. Reduce Model Size
```bash
# Use nano model
--model-size n

# INT8 quantization
--int8
```

### 2. Optimize Inference
```python
# Run at reduced FPS
inference_fps = 5  # Instead of 30

# Downscale images
img = cv2.resize(img, (640, 480))
```

### 3. Multi-threading
```python
import threading

def inference_thread():
    while True:
        if frame_queue.not_empty():
            frame = frame_queue.get()
            result = engine.validate_capture(frame)
            result_queue.put(result)
```

---

## 📚 Additional Resources

- YOLOv8 Docs: https://docs.ultralytics.com
- OpenCV Tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- TFLite Android: https://www.tensorflow.org/lite/android
- CoreML iOS: https://developer.apple.com/documentation/coreml

---

## 🆘 Troubleshooting

### Issue: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Issue: Camera not opening
```bash
# Try different camera ID
python camera_app/desktop_capture.py --camera 1
```

### Issue: Low FPS during inference
- Reduce `inference_fps` in config
- Use smaller model (`yolov8n`)
- Enable GPU if available

---

**Ready to start? Run:**
```bash
python camera_app/desktop_capture.py
```
