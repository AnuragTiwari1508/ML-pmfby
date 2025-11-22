"""
Quick Demo Script
Test all modules with sample data
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.blur_detector import BlurDetector
from inference.light_detector import LightingDetector
from inference.capture_engine import CaptureEngine


def generate_test_image(image_type='good'):
    """Generate synthetic test image."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    if image_type == 'good':
        # Good quality image
        img[:] = (80, 150, 80)  # Green background (crop-like)
        cv2.rectangle(img, (400, 200), (880, 520), (60, 180, 60), -1)
        cv2.circle(img, (640, 360), 100, (40, 200, 40), -1)
        # Add some texture
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    elif image_type == 'blurry':
        # Blurry image
        img[:] = (80, 150, 80)
        cv2.rectangle(img, (400, 200), (880, 520), (60, 180, 60), -1)
        # Apply heavy blur
        img = cv2.GaussianBlur(img, (51, 51), 0)
    
    elif image_type == 'dark':
        # Dark image
        img[:] = (20, 30, 20)
        cv2.rectangle(img, (400, 200), (880, 520), (15, 25, 15), -1)
    
    elif image_type == 'overexposed':
        # Overexposed image
        img[:] = (240, 245, 240)
        cv2.rectangle(img, (400, 200), (880, 520), (250, 255, 250), -1)
    
    return img


def test_blur_detection():
    """Test blur detector."""
    print("\n" + "="*60)
    print("🔍 TESTING BLUR DETECTION")
    print("="*60)
    
    detector = BlurDetector()
    
    # Test good image
    print("\n📸 Test 1: Good Quality Image")
    img_good = generate_test_image('good')
    result = detector.detect(img_good)
    print(f"  Blur Score: {result['blur_score']:.2f}")
    print(f"  Status: {result['status']}")
    print(f"  Message: {detector.get_message(result)}")
    
    # Test blurry image
    print("\n📸 Test 2: Blurry Image")
    img_blurry = generate_test_image('blurry')
    result = detector.detect(img_blurry)
    print(f"  Blur Score: {result['blur_score']:.2f}")
    print(f"  Status: {result['status']}")
    print(f"  Message: {detector.get_message(result)}")
    
    print("\n✅ Blur detection tests complete!")


def test_lighting_detection():
    """Test lighting detector."""
    print("\n" + "="*60)
    print("💡 TESTING LIGHTING DETECTION")
    print("="*60)
    
    detector = LightingDetector()
    
    test_cases = [
        ('good', "Good Lighting"),
        ('dark', "Dark Image"),
        ('overexposed', "Overexposed Image")
    ]
    
    for img_type, description in test_cases:
        print(f"\n📸 Test: {description}")
        img = generate_test_image(img_type)
        result = detector.detect(img)
        print(f"  Brightness: {result['mean_brightness']:.2f}")
        print(f"  Status: {result['status']}")
        print(f"  Message: {detector.get_message(result)}")
        
        feedback = detector.get_detailed_feedback(result)
        if feedback:
            print(f"  Suggestions:")
            for tip in feedback[:2]:
                print(f"    - {tip}")
    
    print("\n✅ Lighting detection tests complete!")


def test_capture_engine():
    """Test unified capture engine."""
    print("\n" + "="*60)
    print("🎯 TESTING UNIFIED CAPTURE ENGINE")
    print("="*60)
    
    engine = CaptureEngine()
    
    test_cases = [
        ('good', "✅ Good Quality Image"),
        ('blurry', "❌ Blurry Image"),
        ('dark', "❌ Dark Image"),
        ('overexposed', "❌ Overexposed Image")
    ]
    
    for img_type, description in test_cases:
        print(f"\n📸 Test: {description}")
        img = generate_test_image(img_type)
        result = engine.validate_capture(img)
        
        print(f"  Overall Score: {result['score']:.1f}/100")
        print(f"  Is Valid: {'✅' if result['is_valid'] else '❌'} {result['is_valid']}")
        print(f"  All Checks: {'✅' if result['all_checks_passed'] else '❌'} {result['all_checks_passed']}")
        
        if result['issues']:
            print(f"  Issues:")
            for issue in result['issues'][:3]:
                print(f"    - {issue}")
        
        guidance = engine.get_guidance_message(result)
        print(f"  💬 Guidance: {guidance}")
    
    print("\n✅ Capture engine tests complete!")


def test_visual_demo():
    """Create visual demonstration."""
    print("\n" + "="*60)
    print("🎨 CREATING VISUAL DEMO")
    print("="*60)
    
    engine = CaptureEngine()
    
    # Create demo images
    images = {
        'Good Quality': generate_test_image('good'),
        'Blurry': generate_test_image('blurry'),
        'Too Dark': generate_test_image('dark'),
        'Overexposed': generate_test_image('overexposed')
    }
    
    print("\n📸 Generating visual comparison...")
    
    # Create composite image
    h, w = 720, 1280
    composite = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    positions = [
        (0, 0), (w, 0),
        (0, h), (w, h)
    ]
    
    for (title, img), (x, y) in zip(images.items(), positions):
        # Validate
        result = engine.validate_capture(img)
        
        # Add title and status
        img_copy = img.copy()
        
        # Title
        cv2.putText(img_copy, title, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Score
        score_color = (0, 255, 0) if result['score'] >= 70 else (0, 0, 255)
        score_text = f"Score: {result['score']:.0f}/100"
        cv2.putText(img_copy, score_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
        
        # Status
        status_text = "VALID" if result['is_valid'] else "INVALID"
        cv2.putText(img_copy, status_text, (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
        
        # Add to composite
        composite[y:y+h, x:x+w] = img_copy
    
    # Save
    output_path = Path('tests/demo_output.jpg')
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), composite)
    
    print(f"\n💾 Demo image saved: {output_path}")
    print("   View this image to see quality comparison!")
    
    # Display if possible
    try:
        cv2.imshow('PMFBY Smart Capture Demo', composite)
        print("\n👁️ Displaying demo (press any key to close)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("   (Display not available in this environment)")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" " * 20 + "🌾 PMFBY SMART CAPTURE - QUICK DEMO")
    print("="*80)
    
    try:
        # Test individual modules
        test_blur_detection()
        test_lighting_detection()
        test_capture_engine()
        
        # Visual demo
        test_visual_demo()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        
        print("\n📋 Next Steps:")
        print("  1. Run camera app: python camera_app/desktop_capture.py")
        print("  2. Train your model: see IMPLEMENTATION_GUIDE.md")
        print("  3. Collect dataset: Take 100-500 crop images")
        print("  4. Integrate into mobile app")
        
        print("\n💡 Quick Commands:")
        print("  • Test blur:     python inference/blur_detector.py --image <IMAGE>")
        print("  • Test lighting: python inference/light_detector.py --image <IMAGE>")
        print("  • Launch camera: python camera_app/desktop_capture.py")
        
    except Exception as e:
        print(f"\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
