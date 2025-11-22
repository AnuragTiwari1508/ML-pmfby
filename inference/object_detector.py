"""
YOLOv8 Object Detector Wrapper
Provides inference interface for crop detection with bounding boxes
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics not installed. Run: pip install ultralytics")


class ObjectDetector:
    """
    YOLOv8-based object detector for crop/plant detection.
    Supports real-time inference and bounding box extraction.
    """
    
    def __init__(self, model_path: str, confidence: float = 0.5, iou_threshold: float = 0.45):
        """
        Args:
            model_path: Path to trained YOLO model (.pt file)
            confidence: Minimum confidence threshold
            iou_threshold: NMS IOU threshold
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLOv8 not installed")
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou_threshold = iou_threshold
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect objects in an image.
        
        Args:
            image: numpy array (BGR format)
        
        Returns:
            dict: {
                'boxes': List of [xmin, ymin, xmax, ymax],
                'confidences': List of confidence scores,
                'classes': List of class IDs,
                'class_names': List of class names,
                'count': Number of detections
            }
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Extract detections
        boxes = []
        confidences = []
        classes = []
        class_names = []
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                # Get coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append(xyxy.tolist())
                
                # Get confidence
                conf = float(box.conf[0].cpu().numpy())
                confidences.append(conf)
                
                # Get class
                cls_id = int(box.cls[0].cpu().numpy())
                classes.append(cls_id)
                class_names.append(results.names[cls_id])
        
        return {
            'boxes': boxes,
            'confidences': confidences,
            'classes': classes,
            'class_names': class_names,
            'count': len(boxes)
        }
    
    def detect_from_file(self, image_path: str) -> Dict:
        """
        Detect objects from image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            dict: Detection results
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect(image)
    
    def draw_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Original image
            detections: Detection results from detect()
        
        Returns:
            np.ndarray: Image with drawn boxes
        """
        img_draw = image.copy()
        
        for i, box in enumerate(detections['boxes']):
            xmin, ymin, xmax, ymax = map(int, box)
            conf = detections['confidences'][i]
            class_name = detections['class_names'][i]
            
            # Draw box
            color = self._get_color(detections['classes'][i])
            cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for text
            cv2.rectangle(
                img_draw,
                (xmin, ymin - label_size[1] - 10),
                (xmin + label_size[0], ymin),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                img_draw,
                label,
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return img_draw
    
    def _get_color(self, class_id: int) -> tuple:
        """Generate consistent color for each class."""
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
    
    def get_largest_detection(self, detections: Dict) -> Optional[Dict]:
        """
        Get the largest detection (by area).
        
        Args:
            detections: Detection results
        
        Returns:
            dict or None: Largest detection info
        """
        if detections['count'] == 0:
            return None
        
        # Calculate areas
        areas = []
        for box in detections['boxes']:
            xmin, ymin, xmax, ymax = box
            area = (xmax - xmin) * (ymax - ymin)
            areas.append(area)
        
        # Get largest
        max_idx = np.argmax(areas)
        
        return {
            'box': detections['boxes'][max_idx],
            'confidence': detections['confidences'][max_idx],
            'class': detections['classes'][max_idx],
            'class_name': detections['class_names'][max_idx],
            'area': areas[max_idx]
        }


def main():
    """Command-line interface for object detection."""
    parser = argparse.ArgumentParser(description='Object Detection for PMFBY')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--show', action='store_true', help='Display result')
    parser.add_argument('--save', type=str, help='Save result to path')
    
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("❌ Ultralytics not installed. Install: pip install ultralytics")
        return
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"⚠️ Model not found: {args.model}")
        print("Using pretrained YOLOv8n. Download your trained model for better results.")
        args.model = 'yolov8n.pt'  # Use pretrained
    
    try:
        # Initialize detector
        print(f"\n🔍 Loading model: {args.model}")
        detector = ObjectDetector(
            model_path=args.model,
            confidence=args.conf,
            iou_threshold=args.iou
        )
        
        # Detect objects
        print(f"📸 Analyzing: {args.image}")
        detections = detector.detect_from_file(args.image)
        
        # Print results
        print(f"\n✅ Found {detections['count']} objects:")
        for i in range(detections['count']):
            print(f"  {i+1}. {detections['class_names'][i]} "
                  f"(conf: {detections['confidences'][i]:.2f})")
        
        # Get largest detection
        largest = detector.get_largest_detection(detections)
        if largest:
            print(f"\n🎯 Largest detection:")
            print(f"  Class: {largest['class_name']}")
            print(f"  Confidence: {largest['confidence']:.2f}")
            print(f"  Area: {largest['area']:.0f} pixels²")
        
        # Show or save result
        if args.show or args.save:
            image = cv2.imread(args.image)
            result_image = detector.draw_detections(image, detections)
            
            if args.save:
                cv2.imwrite(args.save, result_image)
                print(f"\n💾 Saved to: {args.save}")
            
            if args.show:
                cv2.imshow('Object Detection', result_image)
                print("\nPress any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == '__main__':
    main()
