"""
Real-time ML Training Pipeline for PMFBY
Supports online learning, incremental training, and automatic model updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from ultralytics import YOLO
import numpy as np
import cv2
import json
import time
from pathlib import Path
import threading
import queue
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import pickle
import shutil
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RealTimeTrainingPipeline:
    """Complete real-time training system"""
    
    def __init__(self, base_path="/workspaces/ML-pmfby"):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.logs_path = self.base_path / "logs"
        self.checkpoints_path = self.base_path / "checkpoints"
        
        # Create directories
        for path in [self.models_path, self.logs_path, self.checkpoints_path]:
            path.mkdir(exist_ok=True)
        
        # Training queue for real-time updates
        self.training_queue = queue.Queue()
        self.is_training = False
        self.training_thread = None
        
        # Models
        self.models = {
            'yolo': None,
            'classifier': None,
            'feature_extractor': None
        }
        
        # Training history
        self.training_history = {
            'yolo': {'epochs': [], 'loss': [], 'map': []},
            'classifier': {'epochs': [], 'loss': [], 'accuracy': []},
            'overall': {'training_sessions': 0, 'total_images_processed': 0}
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_path / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.logs_path / "tensorboard"))
        
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all models"""
        print("🤖 Initializing models...")
        
        # 1. YOLO for object detection
        try:
            self.models['yolo'] = YOLO('yolov8n.pt')  # Start with nano model
            print("✅ YOLO model initialized")
        except Exception as e:
            print(f"⚠️  YOLO initialization failed: {e}")
        
        # 2. Custom classifier for quality assessment
        self.models['classifier'] = QualityClassifier()
        print("✅ Quality classifier initialized")
        
        # 3. Feature extractor
        self.models['feature_extractor'] = FeatureExtractor()
        print("✅ Feature extractor initialized")
        
        # Load existing models if available
        self.load_existing_models()
    
    def load_existing_models(self):
        """Load previously trained models"""
        model_files = {
            'yolo': self.models_path / 'best_yolo.pt',
            'classifier': self.models_path / 'classifier.pth',
            'feature_extractor': self.models_path / 'feature_extractor.pth'
        }
        
        for model_name, model_path in model_files.items():
            if model_path.exists():
                try:
                    if model_name == 'yolo':
                        self.models[model_name] = YOLO(str(model_path))
                    else:
                        self.models[model_name].load_state_dict(torch.load(model_path))
                    print(f"✅ Loaded existing {model_name} model")
                except Exception as e:
                    print(f"⚠️  Failed to load {model_name}: {e}")
    
    def start_real_time_training(self):
        """Start background training thread"""
        if not self.is_training:
            self.is_training = True
            self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
            self.training_thread.start()
            print("🔄 Real-time training started!")
    
    def stop_real_time_training(self):
        """Stop background training"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join()
        print("⏹️  Real-time training stopped!")
    
    def add_training_data(self, image_path: str, annotations: List[Dict]):
        """Add new training data to queue"""
        training_item = {
            'image_path': image_path,
            'annotations': annotations,
            'timestamp': datetime.now().isoformat()
        }
        self.training_queue.put(training_item)
        print(f"➕ Added training data to queue (size: {self.training_queue.qsize()})")
    
    def _training_worker(self):
        """Background training worker"""
        batch_size = 10
        training_batch = []
        
        while self.is_training:
            try:
                # Collect batch of training data
                if not self.training_queue.empty():
                    item = self.training_queue.get(timeout=1)
                    training_batch.append(item)
                    
                    # Process batch when full or after timeout
                    if len(training_batch) >= batch_size:
                        self._process_training_batch(training_batch)
                        training_batch = []
                
                # Process smaller batches periodically
                elif training_batch and len(training_batch) > 0:
                    time.sleep(5)  # Wait a bit more
                    self._process_training_batch(training_batch)
                    training_batch = []
                
                else:
                    time.sleep(1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Training worker error: {e}")
    
    def _process_training_batch(self, batch: List[Dict]):
        """Process a batch of training data"""
        try:
            print(f"🔄 Processing training batch of {len(batch)} items...")
            
            # Prepare data
            images, yolo_annotations, quality_labels = self._prepare_batch_data(batch)
            
            # Train YOLO (incremental)
            if len(yolo_annotations) > 0:
                self._train_yolo_incremental(images, yolo_annotations)
            
            # Train quality classifier
            if len(quality_labels) > 0:
                self._train_classifier_incremental(images, quality_labels)
            
            # Update training history
            self.training_history['overall']['training_sessions'] += 1
            self.training_history['overall']['total_images_processed'] += len(batch)
            
            # Save updated models
            self.save_models()
            
            print(f"✅ Training batch completed!")
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
    
    def _prepare_batch_data(self, batch: List[Dict]) -> Tuple[List, List, List]:
        """Prepare batch data for training"""
        images = []
        yolo_annotations = []
        quality_labels = []
        
        for item in batch:
            # Load image
            img = cv2.imread(item['image_path'])
            if img is None:
                continue
            
            images.append(img)
            
            # Process annotations
            yolo_ann = []
            for ann in item['annotations']:
                # Convert to YOLO format if needed
                yolo_ann.append({
                    'class': ann.get('class', 0),
                    'bbox': ann.get('bbox', [0.5, 0.5, 0.1, 0.1])
                })
            yolo_annotations.append(yolo_ann)
            
            # Generate quality label based on annotations
            quality_label = self._generate_quality_label(item['annotations'])
            quality_labels.append(quality_label)
        
        return images, yolo_annotations, quality_labels
    
    def _generate_quality_label(self, annotations: List[Dict]) -> int:
        """Generate quality label from annotations"""
        # Simple heuristic: more objects = higher quality
        num_objects = len(annotations)
        
        if num_objects >= 3:
            return 2  # High quality
        elif num_objects >= 1:
            return 1  # Medium quality
        else:
            return 0  # Low quality
    
    def _train_yolo_incremental(self, images: List, annotations: List):
        """Incremental YOLO training"""
        try:
            print("🎯 Training YOLO incrementally...")
            
            # Create temporary training data
            temp_dataset_path = self.base_path / "temp_training"
            temp_dataset_path.mkdir(exist_ok=True)
            
            # Save images and annotations
            for i, (img, anns) in enumerate(zip(images, annotations)):
                img_path = temp_dataset_path / f"temp_{i}.jpg"
                ann_path = temp_dataset_path / f"temp_{i}.txt"
                
                cv2.imwrite(str(img_path), img)
                
                with open(ann_path, 'w') as f:
                    for ann in anns:
                        bbox = ann['bbox']
                        f.write(f"{ann['class']} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            
            # Create temporary data.yaml
            temp_yaml = {
                'path': str(temp_dataset_path),
                'train': '.',
                'val': '.',
                'names': {0: 'crop', 1: 'damage', 2: 'plant', 3: 'field', 4: 'other'},
                'nc': 5
            }
            
            with open(temp_dataset_path / "data.yaml", 'w') as f:
                yaml.dump(temp_yaml, f)
            
            # Train for few epochs
            results = self.models['yolo'].train(
                data=str(temp_dataset_path / "data.yaml"),
                epochs=5,
                batch=2,
                imgsz=640,
                patience=2,
                save=True,
                project=str(self.models_path),
                name='incremental_yolo'
            )
            
            # Update training history
            self.training_history['yolo']['epochs'].append(len(self.training_history['yolo']['epochs']) + 1)
            self.training_history['yolo']['loss'].append(results.results_dict.get('train/box_loss', 0))
            self.training_history['yolo']['map'].append(results.results_dict.get('metrics/mAP50', 0))
            
            # Log to TensorBoard
            epoch = len(self.training_history['yolo']['epochs'])
            self.writer.add_scalar('YOLO/Loss', results.results_dict.get('train/box_loss', 0), epoch)
            self.writer.add_scalar('YOLO/mAP', results.results_dict.get('metrics/mAP50', 0), epoch)
            
            # Cleanup
            shutil.rmtree(temp_dataset_path)
            
            print("✅ YOLO incremental training completed")
            
        except Exception as e:
            self.logger.error(f"YOLO training error: {e}")
    
    def _train_classifier_incremental(self, images: List, labels: List):
        """Incremental classifier training"""
        try:
            print("📊 Training classifier incrementally...")
            
            # Prepare data
            X = []
            y = labels
            
            for img in images:
                # Extract features
                features = self._extract_image_features(img)
                X.append(features)
            
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            
            # Train
            self.models['classifier'].train()
            optimizer = optim.Adam(self.models['classifier'].parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = self.models['classifier'](X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            # Update history
            self.training_history['classifier']['epochs'].append(len(self.training_history['classifier']['epochs']) + 1)
            self.training_history['classifier']['loss'].append(loss.item())
            
            # Calculate accuracy
            with torch.no_grad():
                outputs = self.models['classifier'](X)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y).float().mean().item()
                self.training_history['classifier']['accuracy'].append(accuracy)
            
            # Log to TensorBoard
            epoch = len(self.training_history['classifier']['epochs'])
            self.writer.add_scalar('Classifier/Loss', loss.item(), epoch)
            self.writer.add_scalar('Classifier/Accuracy', accuracy, epoch)
            
            print("✅ Classifier incremental training completed")
            
        except Exception as e:
            self.logger.error(f"Classifier training error: {e}")
    
    def _extract_image_features(self, img: np.ndarray) -> List[float]:
        """Extract features from image for classification"""
        # Simple feature extraction
        features = []
        
        # Color statistics
        for channel in range(3):
            channel_data = img[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.max(channel_data),
                np.min(channel_data)
            ])
        
        # Texture features (simplified)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Contrast
        contrast = np.std(gray)
        features.append(contrast)
        
        # Brightness
        brightness = np.mean(gray)
        features.append(brightness)
        
        return features
    
    def train_full_pipeline(self, dataset_path: str, epochs: int = 50):
        """Train complete pipeline from scratch"""
        print("🚀 Starting full pipeline training...")
        
        # 1. Train YOLO
        print("🎯 Training YOLO model...")
        yolo_results = self._train_yolo_full(dataset_path, epochs)
        
        # 2. Train classifier
        print("📊 Training quality classifier...")
        classifier_results = self._train_classifier_full(dataset_path, epochs)
        
        # 3. Generate training report
        self._generate_training_report(yolo_results, classifier_results)
        
        # 4. Save all models
        self.save_models()
        
        print("🎉 Full pipeline training completed!")
        
        return {
            'yolo_results': yolo_results,
            'classifier_results': classifier_results
        }
    
    def _train_yolo_full(self, dataset_path: str, epochs: int):
        """Full YOLO training"""
        try:
            data_yaml = Path(dataset_path) / "data.yaml"
            
            results = self.models['yolo'].train(
                data=str(data_yaml),
                epochs=epochs,
                batch=16,
                imgsz=640,
                patience=10,
                save=True,
                project=str(self.models_path),
                name='full_yolo_training',
                plots=True
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Full YOLO training error: {e}")
            return None
    
    def _train_classifier_full(self, dataset_path: str, epochs: int):
        """Full classifier training"""
        try:
            # Create dataset for classifier training
            train_loader = self._create_classifier_dataloader(dataset_path, 'train')
            val_loader = self._create_classifier_dataloader(dataset_path, 'val')
            
            optimizer = optim.Adam(self.models['classifier'].parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            
            best_accuracy = 0
            training_losses = []
            validation_accuracies = []
            
            for epoch in range(epochs):
                # Training
                self.models['classifier'].train()
                epoch_loss = 0
                
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.models['classifier'](batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Validation
                self.models['classifier'].eval()
                val_accuracy = self._evaluate_classifier(val_loader)
                
                # Record metrics
                avg_loss = epoch_loss / len(train_loader)
                training_losses.append(avg_loss)
                validation_accuracies.append(val_accuracy)
                
                # Log to TensorBoard
                self.writer.add_scalar('FullTrain/Classifier_Loss', avg_loss, epoch)
                self.writer.add_scalar('FullTrain/Classifier_Accuracy', val_accuracy, epoch)
                
                # Save best model
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.models['classifier'].state_dict(), 
                             self.models_path / 'best_classifier.pth')
                
                scheduler.step()
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={val_accuracy:.4f}")
            
            return {
                'training_losses': training_losses,
                'validation_accuracies': validation_accuracies,
                'best_accuracy': best_accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Full classifier training error: {e}")
            return None
    
    def _create_classifier_dataloader(self, dataset_path: str, split: str):
        """Create dataloader for classifier training"""
        # This is a placeholder - implement based on your dataset structure
        # For now, return empty dataloader
        return []
    
    def _evaluate_classifier(self, dataloader):
        """Evaluate classifier on validation data"""
        if not dataloader:
            return 0.0
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                outputs = self.models['classifier'](batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def predict(self, image_path: str) -> Dict:
        """Make prediction using trained models"""
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not load image'}
        
        results = {}
        
        # YOLO detection
        if self.models['yolo']:
            yolo_results = self.models['yolo'](img)
            results['detection'] = {
                'boxes': yolo_results[0].boxes.xyxy.cpu().numpy().tolist() if yolo_results[0].boxes is not None else [],
                'classes': yolo_results[0].boxes.cls.cpu().numpy().tolist() if yolo_results[0].boxes is not None else [],
                'confidence': yolo_results[0].boxes.conf.cpu().numpy().tolist() if yolo_results[0].boxes is not None else []
            }
        
        # Quality classification
        if self.models['classifier']:
            features = torch.tensor([self._extract_image_features(img)], dtype=torch.float32)
            with torch.no_grad():
                quality_pred = self.models['classifier'](features)
                quality_class = torch.argmax(quality_pred, dim=1).item()
                quality_conf = torch.softmax(quality_pred, dim=1).max().item()
            
            results['quality'] = {
                'class': quality_class,
                'confidence': quality_conf,
                'label': ['Low', 'Medium', 'High'][quality_class]
            }
        
        return results
    
    def save_models(self):
        """Save all trained models"""
        try:
            # Save YOLO
            if self.models['yolo']:
                yolo_path = self.models_path / 'best_yolo.pt'
                self.models['yolo'].save(str(yolo_path))
            
            # Save classifier
            if self.models['classifier']:
                classifier_path = self.models_path / 'classifier.pth'
                torch.save(self.models['classifier'].state_dict(), classifier_path)
            
            # Save feature extractor
            if self.models['feature_extractor']:
                fe_path = self.models_path / 'feature_extractor.pth'
                torch.save(self.models['feature_extractor'].state_dict(), fe_path)
            
            # Save training history
            history_path = self.models_path / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            print("💾 All models saved successfully!")
            
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
    
    def _generate_training_report(self, yolo_results, classifier_results):
        """Generate comprehensive training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'yolo_performance': {
                'final_map': yolo_results.results_dict.get('metrics/mAP50', 0) if yolo_results else 0,
                'final_loss': yolo_results.results_dict.get('train/box_loss', 0) if yolo_results else 0
            },
            'classifier_performance': {
                'best_accuracy': classifier_results['best_accuracy'] if classifier_results else 0,
                'final_loss': classifier_results['training_losses'][-1] if classifier_results and classifier_results['training_losses'] else 0
            },
            'training_stats': self.training_history['overall']
        }
        
        with open(self.logs_path / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("📋 Training report generated!")

class QualityClassifier(nn.Module):
    """Simple quality classifier for crop images"""
    
    def __init__(self, input_dim=15, hidden_dim=64, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class FeatureExtractor(nn.Module):
    """Feature extractor for crop images"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(128 * 8 * 8, 256)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return x

if __name__ == "__main__":
    # Initialize training pipeline
    pipeline = RealTimeTrainingPipeline()
    
    # Start real-time training
    pipeline.start_real_time_training()
    
    print("🚀 Real-time training pipeline ready!")
    print("📝 Use pipeline.add_training_data() to add new training samples")
    print("🎯 Use pipeline.train_full_pipeline() for complete training")
    print("🔮 Use pipeline.predict() for inference")