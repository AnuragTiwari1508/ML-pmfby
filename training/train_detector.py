"""
YOLOv8 Training Pipeline for Crop Detection
Complete training setup with augmentation and export to TFLite
"""

import yaml
from pathlib import Path
import argparse
from datetime import datetime
import shutil

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ Ultralytics not installed. Run: pip install ultralytics")


class YOLOTrainer:
    """
    YOLOv8 training pipeline for PMFBY crop detection.
    """
    
    def __init__(self, model_size='n'):
        """
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
                       'n' = nano (fastest), 'x' = extra large (most accurate)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics not installed")
        
        self.model_size = model_size
        self.model = None
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        project: str = 'runs/train',
        name: str = 'pmfby_crop_detector',
        pretrained: bool = True,
        device: str = '0',
        workers: int = 8,
        patience: int = 50,
        save_period: int = 10
    ):
        """
        Train YOLOv8 model.
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            batch_size: Batch size (-1 for auto)
            img_size: Input image size
            project: Project directory
            name: Experiment name
            pretrained: Use pretrained weights
            device: Device ('0', 'cpu', '0,1,2,3' for multi-GPU)
            workers: Number of data loading workers
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
        
        Returns:
            dict: Training results
        """
        # Load model
        if pretrained:
            model_name = f'yolov8{self.model_size}.pt'
            print(f"\n🔄 Loading pretrained model: {model_name}")
        else:
            model_name = f'yolov8{self.model_size}.yaml'
            print(f"\n🔄 Creating model from scratch: {model_name}")
        
        self.model = YOLO(model_name)
        
        # Train
        print(f"\n🚀 Starting training...")
        print(f"  Dataset: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        print(f"  Device: {device}")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=project,
            name=name,
            device=device,
            workers=workers,
            patience=patience,
            save_period=save_period,
            pretrained=pretrained,
            # Augmentation settings
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            # Optimization
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # Validation
            val=True,
            save=True,
            save_json=False,
            plots=True
        )
        
        print(f"\n✅ Training complete!")
        print(f"  Best weights: {self.model.trainer.best}")
        print(f"  Last weights: {self.model.trainer.last}")
        
        return results
    
    def validate(self, data_yaml: str, weights: str = None):
        """
        Validate model on test set.
        
        Args:
            data_yaml: Path to dataset YAML
            weights: Path to model weights (optional)
        
        Returns:
            dict: Validation metrics
        """
        if weights:
            self.model = YOLO(weights)
        elif not self.model:
            raise ValueError("No model loaded. Provide weights or train first.")
        
        print(f"\n🔍 Validating model...")
        results = self.model.val(data=data_yaml)
        
        print(f"\n📊 Validation Results:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        
        return results
    
    def export(
        self,
        weights: str,
        format: str = 'tflite',
        int8: bool = True,
        img_size: int = 640
    ):
        """
        Export model to mobile format.
        
        Args:
            weights: Path to trained weights
            format: Export format ('tflite', 'onnx', 'coreml', 'saved_model')
            int8: Use INT8 quantization (smaller size, faster inference)
            img_size: Input image size
        
        Returns:
            str: Path to exported model
        """
        self.model = YOLO(weights)
        
        print(f"\n📦 Exporting model to {format.upper()}...")
        print(f"  Input size: {img_size}")
        print(f"  INT8 quantization: {int8}")
        
        exported_model = self.model.export(
            format=format,
            imgsz=img_size,
            int8=int8,
            optimize=True
        )
        
        print(f"\n✅ Export complete: {exported_model}")
        
        return exported_model


def create_dataset_yaml(
    train_path: str,
    val_path: str,
    test_path: str = None,
    class_names: list = None,
    output_path: str = 'dataset.yaml'
):
    """
    Create dataset YAML file for YOLOv8.
    
    Args:
        train_path: Path to training images directory
        val_path: Path to validation images directory
        test_path: Path to test images directory (optional)
        class_names: List of class names
        output_path: Output YAML file path
    """
    if class_names is None:
        class_names = ['crop', 'damage', 'plant', 'field']
    
    data = {
        'path': str(Path(train_path).parent),
        'train': str(Path(train_path).name),
        'val': str(Path(val_path).name),
        'nc': len(class_names),
        'names': class_names
    }
    
    if test_path:
        data['test'] = str(Path(test_path).name)
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✅ Created dataset YAML: {output_path}")
    return output_path


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description='YOLOv8 Training for PMFBY')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--img-size', type=int, default=640, help='Image size')
    train_parser.add_argument('--model-size', type=str, default='n', 
                             choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    train_parser.add_argument('--device', type=str, default='0', help='Device (0, cpu, 0,1,2,3)')
    train_parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    train_parser.add_argument('--name', type=str, default='pmfby_crop', help='Experiment name')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate model')
    val_parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    val_parser.add_argument('--weights', type=str, required=True, help='Model weights')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--weights', type=str, required=True, help='Model weights')
    export_parser.add_argument('--format', type=str, default='tflite', 
                               choices=['tflite', 'onnx', 'coreml', 'saved_model'],
                               help='Export format')
    export_parser.add_argument('--int8', action='store_true', help='Use INT8 quantization')
    export_parser.add_argument('--img-size', type=int, default=640, help='Image size')
    
    # Create YAML command
    yaml_parser = subparsers.add_parser('create-yaml', help='Create dataset YAML')
    yaml_parser.add_argument('--train', type=str, required=True, help='Train images path')
    yaml_parser.add_argument('--val', type=str, required=True, help='Val images path')
    yaml_parser.add_argument('--test', type=str, help='Test images path')
    yaml_parser.add_argument('--classes', type=str, nargs='+', 
                            default=['crop', 'damage', 'plant', 'field'],
                            help='Class names')
    yaml_parser.add_argument('--output', type=str, default='dataset.yaml', 
                            help='Output YAML path')
    
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("❌ Please install: pip install ultralytics")
        return
    
    if args.command == 'train':
        trainer = YOLOTrainer(model_size=args.model_size)
        trainer.train(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            project=args.project,
            name=args.name,
            device=args.device
        )
    
    elif args.command == 'validate':
        trainer = YOLOTrainer()
        trainer.validate(data_yaml=args.data, weights=args.weights)
    
    elif args.command == 'export':
        trainer = YOLOTrainer()
        trainer.export(
            weights=args.weights,
            format=args.format,
            int8=args.int8,
            img_size=args.img_size
        )
    
    elif args.command == 'create-yaml':
        create_dataset_yaml(
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            class_names=args.classes,
            output_path=args.output
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
