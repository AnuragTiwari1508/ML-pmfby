"""
Desktop Camera Capture with Real-time Guidance
Complete smart capture system with all quality checks
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inference.blur_detector import BlurDetector
from inference.light_detector import LightingDetector
try:
    from inference.object_detector import ObjectDetector
    from inference.distance_estimator import DistanceEstimator
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    print("⚠️ Object detection not available. Run: pip install ultralytics")


class SmartCaptureApp:
    """
    Complete camera capture application with real-time guidance.
    Features:
    - Real-time blur detection
    - Lighting quality check
    - Object detection & bounding boxes
    - Distance estimation
    - Multi-angle capture guidance
    - Visual overlay & feedback
    """
    
    def __init__(self, model_path=None, calibration_file=None):
        """
        Initialize capture app.
        
        Args:
            model_path: Path to trained YOLO model (optional)
            calibration_file: Path to distance calibration file (optional)
        """
        # Initialize detectors
        self.blur_detector = BlurDetector(threshold=100, warning_threshold=150)
        self.light_detector = LightingDetector(dark_threshold=40, overexposed_threshold=220)
        
        # Initialize object detector if available
        self.detector = None
        self.distance_estimator = None
        
        if DETECTION_AVAILABLE and model_path and Path(model_path).exists():
            try:
                self.detector = ObjectDetector(model_path)
                from inference.distance_estimator import DistanceEstimator
                self.distance_estimator = DistanceEstimator(calibration_file)
                print("✅ Object detection enabled")
            except Exception as e:
                print(f"⚠️ Object detection failed: {e}")
        
        # Capture settings
        self.capture_mode = 'single'  # or 'multi-angle'
        self.multi_angle_captures = []
        self.max_multi_captures = 3
        
        # UI settings
        self.show_guidance = True
        self.show_overlay = True
        self.inference_fps = 5  # Run inference every N frames
        self.frame_count = 0
        
        # Cache for last results
        self.last_blur_result = None
        self.last_light_result = None
        self.last_detection = None
        
        # Colors
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_WHITE = (255, 255, 255)
    
    def run(self, camera_id=0, save_dir='captures'):
        """
        Start camera capture loop.
        
        Args:
            camera_id: Camera device ID
            save_dir: Directory to save captured images
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n🎥 Camera opened successfully")
        print("\n📋 Controls:")
        print("  SPACE - Capture image")
        print("  M     - Toggle multi-angle mode")
        print("  G     - Toggle guidance overlay")
        print("  Q/ESC - Quit")
        print("\nPress SPACE to capture when all checks pass ✓")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            self.frame_count += 1
            
            # Run inference at reduced rate
            if self.frame_count % self.inference_fps == 0:
                self._update_analysis(frame)
            
            # Draw overlay
            display_frame = self._draw_overlay(frame.copy())
            
            # Show frame
            cv2.imshow('PMFBY Smart Capture', display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            
            elif key == ord(' '):  # SPACE - capture
                self._capture_image(frame, save_path)
            
            elif key == ord('m'):  # M - toggle multi-angle
                self.capture_mode = 'multi-angle' if self.capture_mode == 'single' else 'single'
                print(f"\n📷 Mode: {self.capture_mode}")
            
            elif key == ord('g'):  # G - toggle guidance
                self.show_guidance = not self.show_guidance
                print(f"\n👁️ Guidance: {'ON' if self.show_guidance else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Camera closed")
    
    def _update_analysis(self, frame):
        """Run quality checks on frame."""
        # Blur detection
        self.last_blur_result = self.blur_detector.detect(frame)
        
        # Lighting detection
        self.last_light_result = self.light_detector.detect(frame)
        
        # Object detection
        if self.detector:
            try:
                detections = self.detector.detect(frame)
                if detections['count'] > 0:
                    self.last_detection = self.detector.get_largest_detection(detections)
                else:
                    self.last_detection = None
            except Exception as e:
                self.last_detection = None
    
    def _draw_overlay(self, frame):
        """Draw guidance overlay on frame."""
        h, w = frame.shape[:2]
        
        if not self.show_guidance:
            return frame
        
        # Draw center crosshair
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), 
                self.COLOR_WHITE, 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), 
                self.COLOR_WHITE, 1)
        
        # Draw bounding box guidance
        margin = 100
        guide_rect = (margin, margin, w - margin, h - margin)
        cv2.rectangle(frame, (guide_rect[0], guide_rect[1]), 
                     (guide_rect[2], guide_rect[3]), self.COLOR_YELLOW, 2)
        
        # Draw detected object box
        if self.last_detection:
            box = self.last_detection['box']
            xmin, ymin, xmax, ymax = map(int, box)
            
            # Determine box color based on position
            box_center_x = (xmin + xmax) // 2
            box_center_y = (ymin + ymax) // 2
            
            offset_x = abs(box_center_x - center_x)
            offset_y = abs(box_center_y - center_y)
            
            # Color based on alignment
            if offset_x < 50 and offset_y < 50:
                box_color = self.COLOR_GREEN
            elif offset_x < 100 and offset_y < 100:
                box_color = self.COLOR_YELLOW
            else:
                box_color = self.COLOR_RED
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 3)
            
            # Draw label
            label = f"{self.last_detection['class_name']} {self.last_detection['confidence']:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        # Draw status panel
        self._draw_status_panel(frame)
        
        return frame
    
    def _draw_status_panel(self, frame):
        """Draw status information panel."""
        h, w = frame.shape[:2]
        panel_h = 200
        panel_y = h - panel_h
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y_offset = panel_y + 30
        x_offset = 20
        
        # Check results and display
        all_pass = True
        
        # 1. Blur check
        if self.last_blur_result:
            blur_status = self.last_blur_result['status']
            blur_text = f"Blur: {self.last_blur_result['blur_score']:.1f}"
            
            if blur_status == 'sharp':
                blur_color = self.COLOR_GREEN
                blur_icon = "✓"
            elif blur_status == 'warning':
                blur_color = self.COLOR_YELLOW
                blur_icon = "⚠"
                all_pass = False
            else:
                blur_color = self.COLOR_RED
                blur_icon = "✗"
                all_pass = False
            
            cv2.putText(frame, f"{blur_icon} {blur_text}", 
                       (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, blur_color, 2)
            y_offset += 30
        
        # 2. Lighting check
        if self.last_light_result:
            light_status = self.last_light_result['status']
            light_text = f"Light: {self.last_light_result['mean_brightness']:.1f}"
            
            if light_status == 'ok':
                light_color = self.COLOR_GREEN
                light_icon = "✓"
            else:
                light_color = self.COLOR_RED
                light_icon = "✗"
                all_pass = False
            
            cv2.putText(frame, f"{light_icon} {light_text}", 
                       (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, light_color, 2)
            y_offset += 30
        
        # 3. Detection check
        if self.detector:
            if self.last_detection:
                detect_text = f"Object: {self.last_detection['class_name']}"
                detect_color = self.COLOR_GREEN
                detect_icon = "✓"
            else:
                detect_text = "Object: Not detected"
                detect_color = self.COLOR_RED
                detect_icon = "✗"
                all_pass = False
            
            cv2.putText(frame, f"{detect_icon} {detect_text}", 
                       (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, detect_color, 2)
            y_offset += 30
        
        # 4. Distance check (if available)
        if self.distance_estimator and self.last_detection:
            dist_result = self.distance_estimator.estimate(self.last_detection['box'])
            dist_text = f"Distance: {dist_result['distance_m']:.2f}m"
            
            if dist_result['is_in_range']:
                dist_color = self.COLOR_GREEN
                dist_icon = "✓"
            else:
                dist_color = self.COLOR_YELLOW
                dist_icon = "⚠"
                all_pass = False
            
            cv2.putText(frame, f"{dist_icon} {dist_text}", 
                       (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, dist_color, 2)
            y_offset += 30
        
        # 5. Multi-angle status
        if self.capture_mode == 'multi-angle':
            multi_text = f"Captures: {len(self.multi_angle_captures)}/{self.max_multi_captures}"
            cv2.putText(frame, multi_text, 
                       (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.COLOR_WHITE, 2)
            y_offset += 30
        
        # Overall status
        if all_pass:
            status_text = "READY TO CAPTURE ✓"
            status_color = self.COLOR_GREEN
        else:
            status_text = "ADJUST POSITION ⚠"
            status_color = self.COLOR_YELLOW
        
        cv2.putText(frame, status_text, 
                   (w // 2 - 150, panel_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    def _capture_image(self, frame, save_path):
        """Capture and save image with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.capture_mode == 'single':
            filename = f"capture_{timestamp}.jpg"
        else:
            angle_idx = len(self.multi_angle_captures) + 1
            filename = f"capture_{timestamp}_angle{angle_idx}.jpg"
        
        filepath = save_path / filename
        cv2.imwrite(str(filepath), frame)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'timestamp': timestamp,
            'mode': self.capture_mode,
            'blur_score': self.last_blur_result['blur_score'] if self.last_blur_result else None,
            'light_status': self.last_light_result['status'] if self.last_light_result else None,
            'light_brightness': self.last_light_result['mean_brightness'] if self.last_light_result else None,
        }
        
        if self.last_detection:
            metadata['detection'] = {
                'class': self.last_detection['class_name'],
                'confidence': self.last_detection['confidence'],
                'bbox': self.last_detection['box']
            }
            
            if self.distance_estimator:
                dist_result = self.distance_estimator.estimate(self.last_detection['box'])
                metadata['distance_m'] = dist_result['distance_m']
        
        meta_file = save_path / f"{filename}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Captured: {filename}")
        
        # Handle multi-angle mode
        if self.capture_mode == 'multi-angle':
            self.multi_angle_captures.append(filepath)
            
            if len(self.multi_angle_captures) >= self.max_multi_captures:
                print(f"\n🎉 Multi-angle capture complete! {len(self.multi_angle_captures)} images saved.")
                self.multi_angle_captures = []


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PMFBY Smart Capture Desktop App')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--model', type=str, help='Path to trained YOLO model')
    parser.add_argument('--calibration', type=str, help='Distance calibration file')
    parser.add_argument('--save-dir', type=str, default='captures', help='Save directory')
    
    args = parser.parse_args()
    
    # Create app
    app = SmartCaptureApp(
        model_path=args.model,
        calibration_file=args.calibration
    )
    
    # Run
    try:
        app.run(camera_id=args.camera, save_dir=args.save_dir)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
