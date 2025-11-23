#!/usr/bin/env python3
"""
Simple PMFBY ML System - YOLO Training without OpenCV dependencies
Uses PIL instead of OpenCV to avoid display issues
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import json
from datetime import datetime
import random
import shutil

class SimplePMFBYSystem:
    """Simplified ML system using PIL instead of OpenCV"""
    
    def __init__(self, base_path="/workspaces/ML-pmfby"):
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / "simple_dataset"
        self.models_path = self.base_path / "trained_models"
        self.results_path = self.base_path / "results"
        
        # Create directories
        for path in [self.dataset_path, self.models_path, self.results_path]:
            path.mkdir(exist_ok=True)
        
        # Dataset structure for YOLO
        self.train_images = self.dataset_path / "train" / "images"
        self.train_labels = self.dataset_path / "train" / "labels"
        self.val_images = self.dataset_path / "val" / "images"
        self.val_labels = self.dataset_path / "val" / "labels"
        
        for path in [self.train_images, self.train_labels, self.val_images, self.val_labels]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Agricultural classes
        self.classes = ['healthy_crop', 'damaged_crop', 'pest_damage', 'disease', 'weed']
        self.num_classes = len(self.classes)
        
        print(f"🌾 Simple PMFBY ML System initialized")
        print(f"📂 Dataset: {self.dataset_path}")
        print(f"🤖 Models: {self.models_path}")
        print(f"📊 Results: {self.results_path}")
    
    def create_synthetic_dataset(self, num_images=100):
        """Create synthetic agricultural dataset using PIL"""
        print(f"📸 Creating {num_images} synthetic images using PIL...")
        
        # Clear existing dataset
        for path in [self.train_images, self.train_labels, self.val_images, self.val_labels]:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        
        # Split into train/val
        train_count = int(num_images * 0.8)
        val_count = num_images - train_count
        
        print(f"📊 Creating {train_count} training and {val_count} validation images...")
        
        # Generate training images
        for i in range(train_count):
            img, annotations = self._generate_simple_image(i)
            
            # Save image
            img_path = self.train_images / f"crop_{i:04d}.jpg"
            img.save(str(img_path), 'JPEG')
            
            # Save YOLO annotation
            label_path = self.train_labels / f"crop_{i:04d}.txt"
            self._save_yolo_annotation(annotations, label_path)
        
        # Generate validation images
        for i in range(val_count):
            img, annotations = self._generate_simple_image(i + train_count)
            
            # Save image
            img_path = self.val_images / f"crop_{i:04d}.jpg"
            img.save(str(img_path), 'JPEG')
            
            # Save YOLO annotation
            label_path = self.val_labels / f"crop_{i:04d}.txt"
            self._save_yolo_annotation(annotations, label_path)
        
        # Create data.yaml
        self._create_data_yaml()
        
        print(f"✅ Dataset created with {num_images} images!")
        return {
            "total_images": num_images,
            "train_images": train_count,
            "val_images": val_count,
            "classes": self.classes
        }
    
    def _generate_simple_image(self, seed):
        """Generate simple agricultural image using PIL"""
        random.seed(seed)
        np.random.seed(seed)
        
        width, height = 640, 640
        
        # Create base image with ground color
        ground_colors = [
            (34, 139, 34),   # Forest green
            (85, 107, 47),   # Dark olive green
            (124, 252, 0),   # Lawn green
            (139, 69, 19),   # Saddle brown (soil)
        ]
        
        base_color = random.choice(ground_colors)
        img = Image.new('RGB', (width, height), base_color)
        draw = ImageDraw.Draw(img)
        
        # Add texture patches
        for _ in range(20):
            x = random.randint(0, width-50)
            y = random.randint(0, height-50)
            w = random.randint(10, 50)
            h = random.randint(10, 50)
            
            # Slight color variation
            color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in base_color)
            draw.rectangle([x, y, x+w, y+h], fill=color)
        
        # Generate objects and annotations
        annotations = []
        num_objects = random.randint(1, 4)
        
        for _ in range(num_objects):
            # Random object properties
            obj_class = random.randint(0, self.num_classes - 1)
            
            # Object position and size
            x_center = random.randint(50, width-50)
            y_center = random.randint(50, height-50)
            obj_width = random.randint(30, 80)
            obj_height = random.randint(40, 100)
            
            # Draw object based on class
            self._draw_simple_object(draw, obj_class, x_center, y_center, obj_width, obj_height)
            
            # Convert to YOLO format (normalized coordinates)
            x_norm = x_center / width
            y_norm = y_center / height
            w_norm = obj_width / width
            h_norm = obj_height / height
            
            annotations.append([obj_class, x_norm, y_norm, w_norm, h_norm])
        
        return img, annotations
    
    def _draw_simple_object(self, draw, obj_class, x, y, w, h):
        """Draw simple objects representing different crop conditions"""
        
        if obj_class == 0:  # healthy_crop - green
            color = (50, 200, 50)
            draw.ellipse([x-w//2, y-h//2, x+w//2, y+h//2], fill=color)
            # Add some lines for plant structure
            draw.line([x, y+h//2, x, y-h//2], fill=(30, 150, 30), width=3)
            
        elif obj_class == 1:  # damaged_crop - yellow/brown
            color = (200, 150, 50)
            draw.ellipse([x-w//2, y-h//2, x+w//2, y+h//2], fill=color)
            # Add damage marks
            for _ in range(3):
                dx = random.randint(-w//4, w//4)
                dy = random.randint(-h//4, h//4)
                draw.ellipse([x+dx-5, y+dy-5, x+dx+5, y+dy+5], fill=(150, 100, 30))
                
        elif obj_class == 2:  # pest_damage - brown spots
            color = (120, 80, 40)
            # Multiple small spots
            for _ in range(4):
                dx = random.randint(-w//3, w//3)
                dy = random.randint(-h//3, h//3)
                size = random.randint(5, 15)
                draw.ellipse([x+dx-size, y+dy-size, x+dx+size, y+dy+size], fill=color)
                
        elif obj_class == 3:  # disease - dark patches
            color = (80, 50, 20)
            # Irregular patch
            points = []
            for _ in range(6):
                dx = random.randint(-w//2, w//2)
                dy = random.randint(-h//2, h//2)
                points.extend([x+dx, y+dy])
            draw.polygon(points, fill=color)
            
        elif obj_class == 4:  # weed - different green
            color = (100, 180, 60)
            draw.rectangle([x-w//2, y-h//2, x+w//2, y+h//2], fill=color)
            # Add spiky edges
            for _ in range(3):
                dx = random.randint(-w//2, w//2)
                dy = random.randint(-h//2, h//2)
                draw.line([x, y, x+dx, y+dy], fill=(80, 160, 40), width=2)
    
    def _save_yolo_annotation(self, annotations, file_path):
        """Save annotations in YOLO format"""
        with open(file_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    def _create_data_yaml(self):
        """Create data.yaml file for YOLO training"""
        yaml_content = f"""path: {self.dataset_path}
train: train/images
val: val/images
nc: {self.num_classes}
names: {self.classes}
"""
        
        yaml_path = self.dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"📝 Created data.yaml at: {yaml_path}")
    
    def train_yolo_model(self, epochs=30, batch_size=8):
        """Train YOLO model on the synthetic dataset"""
        print(f"🎯 Training YOLO model for {epochs} epochs...")
        
        # Check if dataset exists
        data_yaml = self.dataset_path / 'data.yaml'
        if not data_yaml.exists():
            print("❌ Dataset not found! Please create dataset first.")
            return False
        
        try:
            # Initialize YOLO model (nano for faster training)
            print("📥 Loading YOLO model...")
            model = YOLO('yolov8n.pt')
            
            print(f"🚀 Starting training...")
            print(f"   - Dataset: {data_yaml}")
            print(f"   - Epochs: {epochs}")
            print(f"   - Batch size: {batch_size}")
            print(f"   - Classes: {self.classes}")
            
            # Train the model
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                patience=10,
                save=True,
                project=str(self.models_path),
                name='pmfby_simple',
                plots=True,
                verbose=True,
                device='cpu'  # Force CPU to avoid CUDA issues
            )
            
            # Save training results
            self._save_training_results(results, epochs, batch_size)
            
            print("🎉 Training completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def _save_training_results(self, results, epochs, batch_size):
        """Save training results and info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training info
        training_info = {
            'timestamp': timestamp,
            'epochs': epochs,
            'batch_size': batch_size,
            'model_type': 'YOLOv8n',
            'classes': self.classes,
            'dataset_path': str(self.dataset_path),
            'model_path': str(self.models_path / 'pmfby_simple'),
        }
        
        # Try to extract metrics if available
        try:
            if hasattr(results, 'results_dict'):
                training_info['metrics'] = {
                    'map50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                    'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
                }
        except:
            training_info['metrics'] = {'note': 'Metrics extraction failed, but training completed'}
        
        # Save info
        info_file = self.results_path / f'training_info_{timestamp}.json'
        with open(info_file, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"💾 Training info saved to: {info_file}")
    
    def run_inference(self, image_path):
        """Run inference on an image"""
        # Find the trained model
        model_path = self.models_path / 'pmfby_simple' / 'weights' / 'best.pt'
        
        if not model_path.exists():
            print("❌ No trained model found! Please train the model first.")
            return None
        
        try:
            print(f"🔮 Running inference on: {image_path}")
            
            # Load model
            model = YOLO(str(model_path))
            
            # Run inference
            results = model(image_path, device='cpu')
            
            # Extract results
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    detection = {
                        'class_id': int(boxes.cls[i].item()),
                        'class': self.classes[int(boxes.cls[i].item())],
                        'confidence': float(boxes.conf[i].item()),
                        'bbox': boxes.xyxy[i].cpu().numpy().tolist()
                    }
                    detections.append(detection)
            
            result = {
                'success': True,
                'image_path': image_path,
                'detections': detections,
                'total_detections': len(detections)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_demo_images(self):
        """Create demo images for testing"""
        print("📷 Creating demo images for inference...")
        
        demo_dir = self.results_path / 'demo_images'
        demo_dir.mkdir(exist_ok=True)
        
        for i in range(3):
            img, _ = self._generate_simple_image(1000 + i)  # Different seed
            demo_path = demo_dir / f'demo_{i+1}.jpg'
            img.save(str(demo_path), 'JPEG')
        
        print(f"✅ Demo images created in: {demo_dir}")
        return demo_dir
    
    def get_system_status(self):
        """Get current system status"""
        status = {
            'dataset_exists': (self.dataset_path / 'data.yaml').exists(),
            'model_exists': (self.models_path / 'pmfby_simple' / 'weights' / 'best.pt').exists(),
            'train_images': len(list(self.train_images.glob('*.jpg'))) if self.train_images.exists() else 0,
            'val_images': len(list(self.val_images.glob('*.jpg'))) if self.val_images.exists() else 0,
        }
        
        return status
    
    def full_pipeline(self, num_images=50, epochs=20):
        """Run complete ML pipeline"""
        print("🚀 Starting Complete PMFBY ML Pipeline (Simple Version)")
        print("=" * 60)
        
        try:
            # Step 1: Create dataset
            print(f"\n📊 Step 1: Creating synthetic dataset ({num_images} images)")
            dataset_info = self.create_synthetic_dataset(num_images)
            
            # Step 2: Train model
            print(f"\n🎯 Step 2: Training YOLO model ({epochs} epochs)")
            training_success = self.train_yolo_model(epochs=epochs, batch_size=4)
            
            if not training_success:
                print("❌ Training failed, stopping pipeline")
                return False
            
            # Step 3: Test inference
            print(f"\n🔮 Step 3: Testing inference")
            demo_dir = self.create_demo_images()
            demo_images = list(demo_dir.glob('*.jpg'))
            
            successful_tests = 0
            for img_path in demo_images:
                result = self.run_inference(str(img_path))
                if result and result.get('success'):
                    successful_tests += 1
                    print(f"   ✅ {img_path.name}: {result['total_detections']} objects detected")
                    for det in result['detections'][:2]:  # Show first 2
                        print(f"      - {det['class']}: {det['confidence']:.3f}")
                else:
                    print(f"   ❌ {img_path.name}: Failed")
            
            # Final summary
            print(f"\n🎉 Pipeline Complete!")
            print(f"   📊 Dataset: {dataset_info['total_images']} images")
            print(f"   🎯 Training: {epochs} epochs completed")
            print(f"   🔮 Inference: {successful_tests}/{len(demo_images)} successful")
            print(f"   📂 Results: {self.results_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            return False

def main():
    """Main function"""
    print("🌾 PMFBY Simple ML System")
    print("🤖 Agricultural Object Detection with YOLO")
    print("=" * 50)
    
    try:
        # Initialize system
        system = SimplePMFBYSystem()
        
        print("\n🎯 Choose an option:")
        print("1. 🔥 Complete Pipeline (Dataset + Training + Testing)")
        print("2. 📊 Create Dataset Only")
        print("3. 🎯 Train Model")
        print("4. 🔮 Test Inference")
        print("5. ⚡ Quick Demo (20 images, 10 epochs)")
        print("6. 📋 System Status")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        start_time = time.time()
        
        if choice == "1":
            print("\n🔥 Running Complete Pipeline...")
            num_images = int(input("Number of images (20-100, default 50): ") or "50")
            epochs = int(input("Number of epochs (10-50, default 20): ") or "20")
            system.full_pipeline(num_images=num_images, epochs=epochs)
            
        elif choice == "2":
            print("\n📊 Creating Dataset...")
            num_images = int(input("Number of images (default 50): ") or "50")
            result = system.create_synthetic_dataset(num_images=num_images)
            print(f"✅ Dataset created: {result}")
            
        elif choice == "3":
            print("\n🎯 Training Model...")
            epochs = int(input("Number of epochs (default 20): ") or "20")
            batch_size = int(input("Batch size (default 4): ") or "4")
            system.train_yolo_model(epochs=epochs, batch_size=batch_size)
            
        elif choice == "4":
            print("\n🔮 Testing Inference...")
            demo_dir = system.create_demo_images()
            demo_images = list(demo_dir.glob('*.jpg'))
            
            for img_path in demo_images:
                result = system.run_inference(str(img_path))
                if result and result.get('success'):
                    print(f"\n📸 {img_path.name}: {result['total_detections']} detections")
                    for det in result['detections']:
                        print(f"   - {det['class']}: {det['confidence']:.3f}")
                else:
                    print(f"\n📸 {img_path.name}: No detections or error")
        
        elif choice == "5":
            print("\n⚡ Quick Demo...")
            system.full_pipeline(num_images=20, epochs=10)
            
        elif choice == "6":
            print("\n📋 System Status:")
            status = system.get_system_status()
            for key, value in status.items():
                print(f"   {key}: {value}")
            
        else:
            print("❌ Invalid choice!")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Execution time: {elapsed/60:.1f} minutes")
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()