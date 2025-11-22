"""
Unified Capture Engine
Combines all detection modules into single inference pipeline
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent))

from inference.blur_detector import BlurDetector
from inference.light_detector import LightingDetector
from inference.geotag_validator import GeotagValidator

try:
    from inference.object_detector import ObjectDetector
    from inference.distance_estimator import DistanceEstimator
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False


class CaptureEngine:
    """
    Unified capture engine combining all quality checks.
    Single entry point for real-time capture validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize capture engine with all detectors.
        
        Args:
            config_path: Path to config YAML file (optional)
        """
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Initialize detectors
        self.blur_detector = BlurDetector(
            threshold=self.config['blur']['threshold'],
            warning_threshold=self.config['blur']['warning_threshold']
        )
        
        self.light_detector = LightingDetector(
            dark_threshold=self.config['lighting']['dark_threshold'],
            overexposed_threshold=self.config['lighting']['overexposed_threshold']
        )
        
        self.geotag_validator = GeotagValidator(
            expected_bounds=self.config['geotag'].get('bounds')
        )
        
        # Initialize object detector if available
        self.object_detector = None
        self.distance_estimator = None
        
        if DETECTION_AVAILABLE:
            model_path = self.config['models'].get('detector')
            if model_path and Path(model_path).exists():
                try:
                    self.object_detector = ObjectDetector(
                        model_path=model_path,
                        confidence=self.config['detection']['confidence'],
                        iou_threshold=self.config['detection']['iou_threshold']
                    )
                    
                    calib_path = self.config['models'].get('calibration')
                    self.distance_estimator = DistanceEstimator(calib_path)
                except Exception as e:
                    print(f"⚠️ Object detector initialization failed: {e}")
    
    def validate_capture(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None
    ) -> Dict:
        """
        Run all quality checks on an image.
        
        Args:
            image: Input image (numpy array)
            image_path: Optional path to image file (for EXIF reading)
        
        Returns:
            dict: {
                'is_valid': bool,
                'all_checks_passed': bool,
                'blur': dict,
                'lighting': dict,
                'detection': dict (optional),
                'distance': dict (optional),
                'geotag': dict (optional),
                'issues': list,
                'score': float (0-100)
            }
        """
        result = {
            'is_valid': False,
            'all_checks_passed': False,
            'issues': [],
            'score': 0.0
        }
        
        scores = []
        
        # 1. Blur detection
        blur_result = self.blur_detector.detect(image)
        result['blur'] = blur_result
        
        if blur_result['is_blurry']:
            result['issues'].append(f"Image is blurry (score: {blur_result['blur_score']:.1f})")
            scores.append(0)
        elif blur_result['needs_warning']:
            result['issues'].append(f"Image may be blurry (score: {blur_result['blur_score']:.1f})")
            scores.append(50)
        else:
            scores.append(100)
        
        # 2. Lighting detection
        light_result = self.light_detector.detect(image)
        result['lighting'] = light_result
        
        if not light_result['is_acceptable']:
            result['issues'].append(f"Poor lighting ({light_result['status']})")
            scores.append(0)
        else:
            scores.append(100)
        
        # 3. Object detection (if available)
        if self.object_detector:
            try:
                detections = self.object_detector.detect(image)
                result['detection'] = detections
                
                if detections['count'] == 0:
                    result['issues'].append("No objects detected")
                    scores.append(0)
                else:
                    scores.append(100)
                    
                    # 4. Distance estimation
                    if self.distance_estimator and detections['count'] > 0:
                        largest = self.object_detector.get_largest_detection(detections)
                        if largest:
                            dist_result = self.distance_estimator.estimate(largest['box'])
                            result['distance'] = dist_result
                            
                            if not dist_result['is_in_range']:
                                result['issues'].append(dist_result['guidance'])
                                scores.append(50)
                            else:
                                scores.append(100)
            
            except Exception as e:
                result['issues'].append(f"Detection error: {e}")
                scores.append(0)
        
        # 5. Geotag validation (if image path provided)
        if image_path and self.config['geotag']['required']:
            gps_data = self.geotag_validator.extract_gps(image_path)
            if gps_data:
                geotag_result = self.geotag_validator.validate(gps_data)
                result['geotag'] = geotag_result
                
                if not geotag_result['is_valid']:
                    result['issues'].extend(geotag_result['issues'])
                    scores.append(0)
                else:
                    scores.append(100)
            else:
                result['issues'].append("No GPS data found")
                scores.append(0)
        
        # Calculate overall score
        if scores:
            result['score'] = np.mean(scores)
        
        # Determine if valid
        result['all_checks_passed'] = len(result['issues']) == 0
        result['is_valid'] = result['score'] >= 70  # 70% threshold
        
        return result
    
    def get_guidance_message(self, validation_result: Dict) -> str:
        """
        Get human-readable guidance message.
        
        Args:
            validation_result: Result from validate_capture()
        
        Returns:
            str: Guidance message
        """
        if validation_result['all_checks_passed']:
            return "✅ All checks passed! Ready to capture."
        
        if not validation_result['issues']:
            return "✅ Image quality acceptable"
        
        # Priority messages
        messages = []
        
        # Critical issues first
        for issue in validation_result['issues']:
            if 'blurry' in issue.lower():
                messages.append("📱 Hold phone steady")
            elif 'dark' in issue.lower():
                messages.append("💡 Move to brighter area")
            elif 'overexposed' in issue.lower():
                messages.append("🌳 Move to shade")
            elif 'no objects' in issue.lower():
                messages.append("🎯 Point at crop/plant")
            elif 'distance' in issue.lower():
                messages.append(f"📏 {validation_result['distance']['guidance']}")
        
        return " | ".join(messages[:3]) if messages else "⚠️ Adjust capture"
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'blur': {'threshold': 100.0, 'warning_threshold': 150.0},
            'lighting': {'dark_threshold': 40, 'overexposed_threshold': 220},
            'detection': {'confidence': 0.5, 'iou_threshold': 0.45},
            'distance': {'min_meters': 0.5, 'max_meters': 3.0},
            'geotag': {'required': False, 'bounds': None},
            'models': {'detector': None, 'calibration': None}
        }


def main():
    """Test capture engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Capture Engine Test')
    parser.add_argument('--image', type=str, required=True, help='Test image path')
    parser.add_argument('--config', type=str, help='Config YAML file')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = CaptureEngine(args.config)
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Cannot load image: {args.image}")
        return
    
    # Validate
    print(f"\n🔍 Validating: {args.image}")
    result = engine.validate_capture(image, args.image)
    
    # Display results
    print(f"\n📊 Validation Results:")
    print(f"  Overall Score: {result['score']:.1f}/100")
    print(f"  Is Valid: {result['is_valid']}")
    print(f"  All Checks Passed: {result['all_checks_passed']}")
    
    if result['issues']:
        print(f"\n⚠️ Issues:")
        for issue in result['issues']:
            print(f"  - {issue}")
    
    # Get guidance
    guidance = engine.get_guidance_message(result)
    print(f"\n💬 Guidance: {guidance}")


if __name__ == '__main__':
    main()
