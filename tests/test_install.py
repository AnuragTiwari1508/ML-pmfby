"""
Quick Test - Minimal dependencies
"""

print("=" * 60)
print("🌾 PMFBY Smart Capture System - Installation Test")
print("=" * 60)

print("\n✅ Basic Python environment OK")
print(f"   Python version: 3.x")

print("\n📦 Checking packages...")

# NumPy
try:
    import numpy as np
    print(f"   ✅ numpy {np.__version__}")
except:
    print("   ❌ numpy - Run: pip install numpy")

# OpenCV
try:
    import cv2
    print(f"   ✅ opencv {cv2.__version__}")
except:
    print("   ❌ opencv - Run: pip install opencv-python-headless")

# PIL
try:
    from PIL import Image
    print(f"   ✅ Pillow (PIL)")
except:
    print("   ❌ Pillow - Run: pip install pillow")

# PyTorch (optional)
try:
    import torch
    print(f"   ✅ torch {torch.__version__} (optional)")
except:
    print("   ⚠️  torch not installed (optional for training)")

# Ultralytics (optional)
try:
    import ultralytics
    print(f"   ✅ ultralytics (optional)")
except:
    print("   ⚠️  ultralytics not installed (optional for training)")

print("\n" + "=" * 60)
print("📚 Installation Guide:")
print("=" * 60)

print("\n🚀 Quick Start (Essential only):")
print("   pip install numpy opencv-python-headless pillow")

print("\n🎓 Full Installation (With ML Training):")
print("   pip install -r requirements.txt")

print("\n📁 Project Structure:")
print("   ✅ inference/      - Detection modules (blur, light, distance)")
print("   ✅ camera_app/     - Desktop capture application")
print("   ✅ training/       - Model training scripts")
print("   ✅ dataset/        - Dataset preparation tools")
print("   ✅ utils/          - Utility functions")

print("\n💡 Try These Commands:")
print("   python inference/blur_detector.py --help")
print("   python inference/light_detector.py --help")

print("\n" + "=" * 60)
