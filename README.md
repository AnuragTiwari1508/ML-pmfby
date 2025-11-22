# 🌾 Smart Image Capture Guidance System for PMFBY

**Pradhan Mantri Fasal Bima Yojana (PMFBY) - AI-Powered Crop Insurance Image Capture**

## 📋 Overview

Complete ML-based smart capture system with:
- ✅ Real-time blur detection
- ✅ Lighting quality analysis
- ✅ Object detection & bounding boxes (YOLOv8)
- ✅ Distance estimation (bbox-based)
- ✅ Angle & multi-capture guidance
- ✅ Automatic geotag validation
- ✅ On-device inference (TFLite/ONNX ready)
- ✅ 15k+ dataset support

**100% self-contained - NO external APIs needed!**

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Test Individual Modules

```bash
# Test blur detection
python inference/blur_detector.py --image test.jpg

# Test lighting analysis
python inference/light_detector.py --image test.jpg
```

### 3️⃣ Run Desktop Capture App

```bash
# Launch real-time capture with all checks
python camera_app/desktop_capture.py
```

---

## 📊 Dataset Format (CSV)

```csv
filename,width,height,class,xmin,ymin,xmax,ymax,latitude,longitude,timestamp_utc,device_model,blur_score,light_flag,angle_pitch,angle_yaw,distance_m,multi_angle_group
IMG_001.jpg,4032,3024,crop,400,800,3200,2600,22.7196,75.8577,2025-11-22T07:12:03Z,MiA3,356.2,ok,2.5,-1.2,1.8,group_001
```

---

**Built with ❤️ for Indian farmers | भारतीय किसानों के लिए बनाया गया**