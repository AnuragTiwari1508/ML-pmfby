"""
Lighting Quality Detector
Detects dark, overexposed, or acceptable lighting conditions
No external APIs - pure OpenCV histogram analysis
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


class LightingDetector:
    """
    Real-time lighting quality detector using histogram analysis.
    Classifies images as: dark, ok, overexposed
    """
    
    def __init__(self, dark_threshold=40, overexposed_threshold=220):
        """
        Args:
            dark_threshold: Mean brightness below this = dark
            overexposed_threshold: Mean brightness above this = overexposed
        """
        self.dark_threshold = dark_threshold
        self.overexposed_threshold = overexposed_threshold
    
    def detect(self, image):
        """
        Analyze lighting quality of an image.
        
        Args:
            image: numpy array (BGR or grayscale)
        
        Returns:
            dict: {
                'mean_brightness': float,
                'std_brightness': float,
                'status': str ('dark', 'ok', 'overexposed'),
                'is_acceptable': bool,
                'histogram': dict with channel histograms
            }
        """
        # Convert to grayscale for brightness analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Determine status
        if mean_brightness < self.dark_threshold:
            status = 'dark'
            is_acceptable = False
        elif mean_brightness > self.overexposed_threshold:
            status = 'overexposed'
            is_acceptable = False
        else:
            status = 'ok'
            is_acceptable = True
        
        # Additional analysis: check for overexposure in highlights
        bright_pixels = np.sum(gray > 240)
        total_pixels = gray.shape[0] * gray.shape[1]
        overexposed_ratio = bright_pixels / total_pixels
        
        # Check for underexposure in shadows
        dark_pixels = np.sum(gray < 20)
        underexposed_ratio = dark_pixels / total_pixels
        
        return {
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'status': status,
            'is_acceptable': is_acceptable,
            'overexposed_ratio': float(overexposed_ratio),
            'underexposed_ratio': float(underexposed_ratio),
            'histogram': hist.flatten().tolist()
        }
    
    def detect_from_file(self, image_path):
        """
        Detect lighting quality from image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            dict: Lighting detection results
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect(image)
    
    def get_message(self, result):
        """
        Get user-friendly message based on lighting detection.
        
        Args:
            result: Result dict from detect()
        
        Returns:
            str: User message
        """
        brightness = result['mean_brightness']
        status = result['status']
        
        if status == 'dark':
            return f"❌ Too dark (brightness: {brightness:.1f}). Move to better lighting!"
        elif status == 'overexposed':
            return f"❌ Overexposed (brightness: {brightness:.1f}). Reduce light or move to shade!"
        else:
            return f"✅ Good lighting (brightness: {brightness:.1f})"
    
    def get_detailed_feedback(self, result):
        """
        Get detailed feedback for improving lighting.
        
        Args:
            result: Result dict from detect()
        
        Returns:
            list: List of feedback strings
        """
        feedback = []
        
        if result['status'] == 'dark':
            feedback.append("📱 Turn on flash or move to brighter area")
            feedback.append("☀️ Try capturing in daylight")
        
        if result['status'] == 'overexposed':
            feedback.append("🌳 Move to shade")
            feedback.append("📱 Disable flash if enabled")
            feedback.append("☁️ Wait for clouds to diffuse sunlight")
        
        if result['overexposed_ratio'] > 0.1 and result['status'] != 'overexposed':
            feedback.append("⚠️ Some areas are too bright")
        
        if result['underexposed_ratio'] > 0.1 and result['status'] != 'dark':
            feedback.append("⚠️ Some areas are too dark")
        
        if result['std_brightness'] < 20:
            feedback.append("⚠️ Low contrast - try different angle")
        
        return feedback


def main():
    """Command-line interface for testing lighting detection."""
    parser = argparse.ArgumentParser(description='Lighting Quality Detector for PMFBY')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--dark', type=float, default=40, help='Dark threshold')
    parser.add_argument('--overexposed', type=float, default=220, help='Overexposed threshold')
    parser.add_argument('--show', action='store_true', help='Display image with result')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = LightingDetector(
        dark_threshold=args.dark,
        overexposed_threshold=args.overexposed
    )
    
    # Detect lighting
    print(f"\n💡 Analyzing lighting: {args.image}")
    result = detector.detect_from_file(args.image)
    message = detector.get_message(result)
    
    print(f"\n{message}")
    print(f"\nDetails:")
    print(f"  Mean Brightness: {result['mean_brightness']:.2f}")
    print(f"  Std Brightness: {result['std_brightness']:.2f}")
    print(f"  Status: {result['status']}")
    print(f"  Acceptable: {result['is_acceptable']}")
    print(f"  Overexposed Ratio: {result['overexposed_ratio']:.2%}")
    print(f"  Underexposed Ratio: {result['underexposed_ratio']:.2%}")
    
    # Get feedback
    feedback = detector.get_detailed_feedback(result)
    if feedback:
        print(f"\n📋 Suggestions:")
        for tip in feedback:
            print(f"  {tip}")
    
    # Show image if requested
    if args.show:
        image = cv2.imread(args.image)
        if image is not None:
            # Add text overlay
            color = (0, 255, 0) if result['is_acceptable'] else (0, 0, 255)
            
            text = f"Light: {result['mean_brightness']:.1f} ({result['status']})"
            cv2.putText(image, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Lighting Detection', image)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
