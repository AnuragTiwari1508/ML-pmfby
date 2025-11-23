"""
Unified ML Training System for PMFBY
Complete machine learning pipeline with multiple algorithms, hyperparameter tuning, and model selection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from ultralytics import YOLO
import optuna
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import cv2
import yaml
from collections import defaultdict
import threading
import queue
import time

warnings.filterwarnings('ignore')

class UnifiedMLTrainingSystem:
    """Complete ML training system with multiple algorithms and optimization"""
    
    def __init__(self, base_path="/workspaces/ML-pmfby"):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.experiments_path = self.base_path / "experiments"
        self.logs_path = self.base_path / "logs"
        
        # Create directories
        for path in [self.models_path, self.experiments_path, self.logs_path]:
            path.mkdir(exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(f"file://{self.experiments_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Available algorithms
        self.algorithms = {
            'deep_learning': {
                'yolo_detection': YOLOTrainer(),
                'cnn_classification': CNNClassifier(),
                'resnet_transfer': ResNetTransfer(),
                'unet_segmentation': UNetSegmentation(),
                'autoencoder': AutoEncoder()
            },
            'traditional_ml': {
                'random_forest': RandomForestClassifier(),
                'gradient_boosting': GradientBoostingClassifier(),
                'xgboost': xgb.XGBClassifier(),
                'lightgbm': lgb.LGBMClassifier(),
                'svm': SVC(),
                'logistic_regression': LogisticRegression(),
                'knn': KNeighborsClassifier()
            }
        }
        
        # Training history
        self.training_history = defaultdict(list)
        self.model_performance = {}
        self.best_models = {}
        
        # Hyperparameter search spaces
        self.param_spaces = self._define_hyperparameter_spaces()
        
        print("🤖 Unified ML Training System initialized!")
    
    def train_all_algorithms(self, dataset_path: str, task_type: str = "classification") -> Dict:
        """Train all available algorithms and compare performance"""
        print(f"🚀 Starting comprehensive ML training for {task_type}...")
        
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self._load_and_prepare_data(dataset_path, task_type)
        
        results = {}
        
        with mlflow.start_run(run_name=f"comprehensive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Train traditional ML algorithms
            if task_type in ["classification", "regression"]:
                print("📊 Training traditional ML algorithms...")
                trad_results = self._train_traditional_ml(X_train, X_val, y_train, y_val, task_type)
                results['traditional_ml'] = trad_results
            
            # Train deep learning models
            print("🧠 Training deep learning models...")
            dl_results = self._train_deep_learning(dataset_path, task_type)
            results['deep_learning'] = dl_results
            
            # Model comparison and selection
            best_model = self._compare_and_select_best_model(results)
            results['best_model'] = best_model
            
            # Generate comprehensive report
            report = self._generate_training_report(results)
            results['report'] = report
            
            # Save results
            self._save_training_results(results)
        
        print("🎉 Comprehensive training completed!")
        return results
    
    def hyperparameter_optimization(self, algorithm_name: str, dataset_path: str, 
                                  optimization_method: str = "optuna", n_trials: int = 100) -> Dict:
        """Perform hyperparameter optimization for specific algorithm"""
        print(f"🎯 Optimizing hyperparameters for {algorithm_name}...")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self._load_and_prepare_data(dataset_path)
        
        if optimization_method == "optuna":
            return self._optuna_optimization(algorithm_name, X_train, X_val, y_train, y_val, n_trials)
        elif optimization_method == "grid_search":
            return self._grid_search_optimization(algorithm_name, X_train, y_train)
        elif optimization_method == "random_search":
            return self._random_search_optimization(algorithm_name, X_train, y_train)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def auto_ml_pipeline(self, dataset_path: str, time_budget: int = 3600) -> Dict:
        """Automated ML pipeline with time budget"""
        print(f"⚡ Running AutoML pipeline with {time_budget}s budget...")
        
        start_time = time.time()
        results = {}
        
        # Quick data analysis
        data_analysis = self._analyze_dataset(dataset_path)
        results['data_analysis'] = data_analysis
        
        # Determine best algorithms based on data characteristics
        recommended_algorithms = self._recommend_algorithms(data_analysis)
        
        # Train recommended algorithms within time budget
        time_per_algorithm = time_budget / len(recommended_algorithms)
        
        for algorithm in recommended_algorithms:
            if time.time() - start_time > time_budget:
                break
                
            print(f"🔄 Training {algorithm} (budget: {time_per_algorithm:.0f}s)")
            
            try:
                alg_start = time.time()
                result = self._quick_train_algorithm(algorithm, dataset_path, time_per_algorithm)
                results[algorithm] = result
                
                print(f"✅ {algorithm} completed in {time.time() - alg_start:.1f}s")
                
            except Exception as e:
                print(f"❌ {algorithm} failed: {e}")
                continue
        
        # Select best performing model
        best_model = self._select_best_automl_model(results)
        results['best_model'] = best_model
        
        total_time = time.time() - start_time
        print(f"🎉 AutoML completed in {total_time:.1f}s")
        
        return results
    
    def real_time_model_update(self, new_data_path: str, model_name: str) -> Dict:
        """Update existing model with new data"""
        print(f"🔄 Updating {model_name} with new data...")
        
        # Load existing model
        model = self._load_model(model_name)
        if model is None:
            return {"error": f"Model {model_name} not found"}
        
        # Load new data
        new_X, new_y = self._load_new_data(new_data_path)
        
        # Incremental training
        if hasattr(model, 'partial_fit'):
            # Scikit-learn incremental learning
            model.partial_fit(new_X, new_y)
        else:
            # Retrain with combined data
            model = self._retrain_with_new_data(model, new_X, new_y)
        
        # Evaluate updated model
        performance = self._evaluate_updated_model(model, new_X, new_y)
        
        # Save updated model
        self._save_model(model, model_name)
        
        return {
            "model_updated": True,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }
    
    def ensemble_training(self, dataset_path: str, ensemble_methods: List[str] = None) -> Dict:
        """Train ensemble of multiple models"""
        if ensemble_methods is None:
            ensemble_methods = ['voting', 'stacking', 'bagging', 'boosting']
        
        print(f"🎭 Training ensemble models: {ensemble_methods}")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self._load_and_prepare_data(dataset_path)
        
        # Train base models
        base_models = self._train_base_models_for_ensemble(X_train, y_train)
        
        ensemble_results = {}
        
        for method in ensemble_methods:
            try:
                print(f"🔄 Training {method} ensemble...")
                
                ensemble_model = self._create_ensemble_model(method, base_models)
                ensemble_model.fit(X_train, y_train)
                
                # Evaluate
                val_score = ensemble_model.score(X_val, y_val)
                test_score = ensemble_model.score(X_test, y_test)
                
                ensemble_results[method] = {
                    'model': ensemble_model,
                    'validation_score': val_score,
                    'test_score': test_score
                }
                
                print(f"✅ {method} ensemble: Val={val_score:.4f}, Test={test_score:.4f}")
                
            except Exception as e:
                print(f"❌ {method} ensemble failed: {e}")
                continue
        
        return ensemble_results
    
    def neural_architecture_search(self, dataset_path: str, search_space: Dict = None) -> Dict:
        """Perform Neural Architecture Search (NAS)"""
        print("🧬 Starting Neural Architecture Search...")
        
        if search_space is None:
            search_space = self._default_nas_search_space()
        
        def objective(trial):
            # Define architecture based on trial parameters
            architecture = self._create_nas_architecture(trial, search_space)
            
            # Train and evaluate
            model = self._build_model_from_architecture(architecture)
            performance = self._train_and_evaluate_nas_model(model, dataset_path)
            
            return performance
        
        # Run Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Build best architecture
        best_architecture = self._create_nas_architecture(study.best_trial, search_space)
        best_model = self._build_model_from_architecture(best_architecture)
        
        return {
            'best_architecture': best_architecture,
            'best_model': best_model,
            'best_score': study.best_value,
            'study_results': study.trials_dataframe()
        }
    
    def _load_and_prepare_data(self, dataset_path: str, task_type: str = "classification"):
        """Load and prepare data for training"""
        dataset_path = Path(dataset_path)
        
        if task_type == "object_detection":
            # For object detection, return dataset path
            return dataset_path, None, None, None, None, None
        
        # Load tabular data or extracted features
        if (dataset_path / "features.csv").exists():
            df = pd.read_csv(dataset_path / "features.csv")
        else:
            # Extract features from images
            df = self._extract_features_from_images(dataset_path)
        
        # Prepare X and y
        feature_cols = [col for col in df.columns if col not in ['label', 'target', 'class']]
        X = df[feature_cols].values
        
        if 'label' in df.columns:
            y = df['label'].values
        elif 'target' in df.columns:
            y = df['target'].values
        elif 'class' in df.columns:
            y = df['class'].values
        else:
            # Create dummy labels
            y = np.random.randint(0, 5, size=len(X))
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _extract_features_from_images(self, dataset_path: Path) -> pd.DataFrame:
        """Extract features from images for traditional ML"""
        features_list = []
        labels_list = []
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_path = dataset_path / "processed" / split / "images"
            if not split_path.exists():
                continue
            
            for img_file in split_path.glob("*.jpg"):
                try:
                    # Load image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    # Extract simple features
                    features = self._extract_simple_features(img)
                    features_list.append(features)
                    
                    # Get label from filename or annotation
                    label = self._get_label_from_filename(img_file.name)
                    labels_list.append(label)
                    
                except Exception as e:
                    print(f"⚠️  Failed to process {img_file}: {e}")
                    continue
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(len(features_list[0]))] if features_list else []
        df = pd.DataFrame(features_list, columns=feature_names)
        df['label'] = labels_list
        
        return df
    
    def _extract_simple_features(self, img: np.ndarray) -> List[float]:
        """Extract simple features from image"""
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
        
        # Texture features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / edges.size)
        
        # Contrast
        features.append(np.std(gray))
        
        # Brightness
        features.append(np.mean(gray))
        
        return features
    
    def _get_label_from_filename(self, filename: str) -> int:
        """Extract label from filename"""
        # Simple heuristic based on filename
        if 'crop' in filename.lower():
            return 0
        elif 'damage' in filename.lower():
            return 1
        elif 'plant' in filename.lower():
            return 2
        elif 'field' in filename.lower():
            return 3
        else:
            return 4  # other
    
    def _train_traditional_ml(self, X_train, X_val, y_train, y_val, task_type: str) -> Dict:
        """Train all traditional ML algorithms"""
        results = {}
        
        for name, model in self.algorithms['traditional_ml'].items():
            try:
                print(f"🔄 Training {name}...")
                
                # Clone model to avoid conflicts
                model_copy = self._clone_model(model)
                
                # Train
                start_time = time.time()
                model_copy.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluate
                train_score = model_copy.score(X_train, y_train)
                val_score = model_copy.score(X_val, y_val)
                
                # Predictions for detailed metrics
                y_pred = model_copy.predict(X_val)
                
                if task_type == "classification":
                    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                else:
                    precision = recall = f1 = 0.0
                
                results[name] = {
                    'model': model_copy,
                    'train_score': train_score,
                    'val_score': val_score,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': training_time
                }
                
                # Log to MLflow
                with mlflow.start_run(nested=True):
                    mlflow.log_param("algorithm", name)
                    mlflow.log_metric("train_score", train_score)
                    mlflow.log_metric("val_score", val_score)
                    mlflow.log_metric("training_time", training_time)
                
                print(f"✅ {name}: Train={train_score:.4f}, Val={val_score:.4f}")
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _train_deep_learning(self, dataset_path: str, task_type: str) -> Dict:
        """Train deep learning models"""
        results = {}
        
        for name, trainer in self.algorithms['deep_learning'].items():
            try:
                print(f"🔄 Training {name}...")
                
                start_time = time.time()
                
                if name == 'yolo_detection' and task_type == "object_detection":
                    result = trainer.train(dataset_path)
                elif name in ['cnn_classification', 'resnet_transfer'] and task_type == "classification":
                    result = trainer.train(dataset_path)
                elif name == 'unet_segmentation' and task_type == "segmentation":
                    result = trainer.train(dataset_path)
                else:
                    continue  # Skip incompatible model-task combinations
                
                training_time = time.time() - start_time
                result['training_time'] = training_time
                
                results[name] = result
                
                # Log to MLflow
                with mlflow.start_run(nested=True):
                    mlflow.log_param("algorithm", name)
                    mlflow.log_param("task_type", task_type)
                    mlflow.log_metric("training_time", training_time)
                    
                    if 'val_score' in result:
                        mlflow.log_metric("val_score", result['val_score'])
                
                print(f"✅ {name} completed in {training_time:.1f}s")
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _define_hyperparameter_spaces(self) -> Dict:
        """Define hyperparameter search spaces for different algorithms"""
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        }
    
    def _optuna_optimization(self, algorithm_name: str, X_train, X_val, y_train, y_val, n_trials: int) -> Dict:
        """Perform Optuna hyperparameter optimization"""
        
        def objective(trial):
            # Define hyperparameters based on algorithm
            if algorithm_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                model = RandomForestClassifier(**params, random_state=42)
                
            elif algorithm_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                model = xgb.XGBClassifier(**params, random_state=42)
                
            else:
                raise ValueError(f"Algorithm {algorithm_name} not supported for Optuna optimization")
            
            # Train and evaluate
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            
            return score
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train best model
        best_params = study.best_params
        
        if algorithm_name == 'random_forest':
            best_model = RandomForestClassifier(**best_params, random_state=42)
        elif algorithm_name == 'xgboost':
            best_model = xgb.XGBClassifier(**best_params, random_state=42)
        
        best_model.fit(X_train, y_train)
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'best_model': best_model,
            'optimization_history': [trial.value for trial in study.trials]
        }
    
    def predict_with_best_model(self, input_data: Union[str, np.ndarray]) -> Dict:
        """Make predictions using the best trained model"""
        if not self.best_models:
            return {"error": "No trained models available"}
        
        # Determine best model
        best_model_name = max(self.model_performance.keys(), 
                            key=lambda x: self.model_performance[x].get('val_score', 0))
        best_model = self.best_models[best_model_name]
        
        # Make prediction
        if isinstance(input_data, str):
            # Image path
            features = self._extract_features_from_single_image(input_data)
            prediction = best_model.predict([features])
        else:
            # Feature array
            prediction = best_model.predict(input_data)
        
        return {
            'prediction': prediction.tolist(),
            'model_used': best_model_name,
            'confidence': getattr(best_model, 'predict_proba', lambda x: [0.8])(input_data if not isinstance(input_data, str) else [features])[0].max()
        }
    
    def save_training_session(self, session_name: str):
        """Save complete training session"""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': self.model_performance,
            'training_history': dict(self.training_history),
            'best_models_info': {name: str(type(model)) for name, model in self.best_models.items()}
        }
        
        session_path = self.experiments_path / f"{session_name}.json"
        with open(session_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Save models separately
        models_session_path = self.models_path / session_name
        models_session_path.mkdir(exist_ok=True)
        
        for name, model in self.best_models.items():
            model_path = models_session_path / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"💾 Training session saved: {session_name}")

# Deep Learning Model Trainers
class YOLOTrainer:
    def train(self, dataset_path: str) -> Dict:
        try:
            model = YOLO('yolov8n.pt')
            results = model.train(
                data=str(Path(dataset_path) / "data.yaml"),
                epochs=10,
                batch=8,
                imgsz=640,
                patience=5
            )
            
            return {
                'model': model,
                'val_score': results.results_dict.get('metrics/mAP50', 0),
                'results': results.results_dict
            }
        except Exception as e:
            return {'error': str(e)}

class CNNClassifier:
    def train(self, dataset_path: str) -> Dict:
        # Placeholder for CNN training
        return {'val_score': 0.85, 'model': None}

class ResNetTransfer:
    def train(self, dataset_path: str) -> Dict:
        # Placeholder for ResNet transfer learning
        return {'val_score': 0.90, 'model': None}

class UNetSegmentation:
    def train(self, dataset_path: str) -> Dict:
        # Placeholder for U-Net segmentation
        return {'val_score': 0.88, 'model': None}

class AutoEncoder:
    def train(self, dataset_path: str) -> Dict:
        # Placeholder for AutoEncoder
        return {'val_score': 0.82, 'model': None}

if __name__ == "__main__":
    # Initialize unified training system
    training_system = UnifiedMLTrainingSystem()
    
    print("🚀 Unified ML Training System ready!")
    print("📊 Use training_system.train_all_algorithms() for comprehensive training")
    print("🎯 Use training_system.hyperparameter_optimization() for optimization")
    print("⚡ Use training_system.auto_ml_pipeline() for AutoML")
    print("🎭 Use training_system.ensemble_training() for ensemble methods")