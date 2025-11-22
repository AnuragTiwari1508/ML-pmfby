"""
Blur Detection Module using Laplacian Variance
Fast on-device blur detection without external APIs
"""

import numpy as np
from pathlib import Path
import argparse

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, using PIL fallback")


class BlurDetector:
    """
    Real-time blur detection using Laplacian variance method.
    Fast enough for real-time mobile inference (~2ms per frame).
    """
    
    def __init__(self, threshold=100.0, warning_threshold=150.0):
        """
        Args:
            threshold: Minimum blur score (below = rejected)
            warning_threshold: Warning threshold (below = warning shown)
        """
        self.threshold = threshold
        self.warning_threshold = warning_threshold
    
    def detect(self, image):
        """
        Calculate blur score for an image.
        
        Args:
            image: numpy array (BGR or grayscale)
        
        Returns:
            dict: {
                'blur_score': float,
                'is_blurry': bool,
                'needs_warning': bool,
                'status': str ('sharp', 'warning', 'blurry')
            }
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                # Manual RGB to grayscale
                gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = image
        
        # Calculate Laplacian variance
        if CV2_AVAILABLE:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        else:
            # Manual Laplacian (3x3 kernel)
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            from scipy import signal
            laplacian = signal.convolve2d(gray, kernel, mode='same')
        blur_score = laplacian.var()
        
        # Determine status
        if blur_score < self.threshold:
            status = 'blurry'
            is_blurry = True
            needs_warning = True
        elif blur_score < self.warning_threshold:
            status = 'warning'
            is_blurry = False
            needs_warning = True
        else:
            status = 'sharp'
            is_blurry = False
            needs_warning = False
        
        return {
            'blur_score': float(blur_score),
            'is_blurry': is_blurry,
            'needs_warning': needs_warning,
            'status': status
        }
    
    def detect_from_file(self, image_path):
        """
        Detect blur from image file path.
        
        Args:
            image_path: Path to image file
        
        Returns:
            dict: Blur detection results
        """
        if CV2_AVAILABLE:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
        else:
            from PIL import Image
            pil_img = Image.open(image_path)
            image = np.array(pil_img)
            if len(image.shape) == 2:
                pass  # already grayscale
            elif image.shape[2] == 3:
                image = image[:, :, ::-1]  # RGB to BGR
        return self.detect(image)
    
    def get_message(self, result):
        """
        Get user-friendly message based on blur detection result.
        
        Args:
            result: Result dict from detect()
        
        Returns:
            str: User message
        """
        score = result['blur_score']
        status = result['status']
        
        if status == 'blurry':
            return f"❌ Image too blurry (score: {score:.1f}). Hold phone steady!"
        elif status == 'warning':
            return f"⚠️ Image may be blurry (score: {score:.1f}). Try to stabilize."
        else:
            return f"✅ Image is sharp (score: {score:.1f})"


def main():
    """Command-line interface for testing blur detection."""
    parser = argparse.ArgumentParser(description='Blur Detection for PMFBY')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--threshold', type=float, default=100.0, help='Blur threshold')
    parser.add_argument('--warning', type=float, default=150.0, help='Warning threshold')
    parser.add_argument('--show', action='store_true', help='Display image with result')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BlurDetector(
        threshold=args.threshold,
        warning_threshold=args.warning
    )
    
    # Detect blur
    print(f"\n🔍 Analyzing: {args.image}")
    result = detector.detect_from_file(args.image)
    message = detector.get_message(result)
    
    print(f"\n{message}")
    print(f"\nDetails:")
    print(f"  Blur Score: {result['blur_score']:.2f}")
    print(f"  Status: {result['status']}")
    print(f"  Is Blurry: {result['is_blurry']}")
    print(f"  Needs Warning: {result['needs_warning']}")
    
    # Show image if requested
    if args.show:
        image = cv2.imread(args.image)
        if image is not None:
            # Add text overlay
            color = (0, 255, 0) if result['status'] == 'sharp' else \
                    (0, 165, 255) if result['status'] == 'warning' else (0, 0, 255)
            
            text = f"Blur: {result['blur_score']:.1f} ({result['status']})"
            cv2.putText(image, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Blur Detection', image)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
