#!/bin/bash

# PMFBY Smart Capture - Quick Setup Script
# Run this to verify everything is working

echo "=================================================="
echo "🌾 PMFBY Smart Capture - Quick Setup"
echo "=================================================="

# Check Python
echo ""
echo "🐍 Checking Python..."
python3 --version || { echo "❌ Python not found!"; exit 1; }

# Check dependencies
echo ""
echo "📦 Checking dependencies..."
python3 -c "import cv2" 2>/dev/null || { 
    echo "⚠️ OpenCV not installed. Installing..."; 
    pip install opencv-python opencv-contrib-python; 
}

python3 -c "import numpy" 2>/dev/null || { 
    echo "⚠️ NumPy not installed. Installing..."; 
    pip install numpy; 
}

echo "✅ Core dependencies OK"

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p models
mkdir -p dataset/{raw,processed,yolo}/{train,val}/{images,labels}
mkdir -p captures
mkdir -p runs/train
echo "✅ Directories created"

# Run quick demo
echo ""
echo "🚀 Running quick demo..."
python3 tests/quick_demo.py

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "📋 What you can do now:"
echo ""
echo "1️⃣ Test camera (no training needed):"
echo "   python camera_app/desktop_capture.py"
echo ""
echo "2️⃣ Test on an image:"
echo "   python inference/blur_detector.py --image <YOUR_IMAGE>"
echo ""
echo "3️⃣ Start training pipeline:"
echo "   See IMPLEMENTATION_GUIDE.md"
echo ""
echo "💡 Tip: Read IMPLEMENTATION_GUIDE.md for complete walkthrough"
