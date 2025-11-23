"""
Comprehensive Web Interface for ML-PMFBY
Complete web application with data upload, training, real-time inference, and model management
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, session, redirect, url_for
from flask_socketio import SocketIO, emit
import numpy as np
import base64
import io
from PIL import Image
import json
import time
import threading
from pathlib import Path
import cv2
import pandas as pd
from datetime import datetime
import uuid
import os
from werkzeug.utils import secure_filename
import zipfile
import shutil
from typing import Dict, List, Optional
import plotly.graph_objs as go
import plotly.utils
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
import sys
sys.path.append('/workspaces/ML-pmfby')

from dataset.complete_dataset_manager import ComprehensiveDatasetManager
from training.real_time_training_pipeline import RealTimeTrainingPipeline
from inference.advanced_cv_features import AdvancedCVFeatures
from training.unified_ml_training import UnifiedMLTrainingSystem

app = Flask(__name__, 
            template_folder='/workspaces/ML-pmfby/templates',
            static_folder='/workspaces/ML-pmfby/static')
app.config['SECRET_KEY'] = 'pmfby_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize ML components
dataset_manager = ComprehensiveDatasetManager()
training_pipeline = RealTimeTrainingPipeline()
cv_features = AdvancedCVFeatures()
ml_training = UnifiedMLTrainingSystem()

# Global variables for tracking
active_sessions = {}
training_status = {"is_training": False, "progress": 0, "current_task": ""}

class WebMLInterface:
    """Complete web interface for ML operations"""
    
    def __init__(self):
        self.upload_folder = Path('/workspaces/ML-pmfby/uploads')
        self.results_folder = Path('/workspaces/ML-pmfby/results')
        self.temp_folder = Path('/workspaces/ML-pmfby/temp')
        
        # Create directories
        for folder in [self.upload_folder, self.results_folder, self.temp_folder]:
            folder.mkdir(exist_ok=True)
        
        # Allowed file extensions
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'zip', 'csv', 'json'}
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

# Initialize web interface
web_interface = WebMLInterface()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/upload')
def upload_page():
    """Data upload interface"""
    return render_template('upload.html')

@app.route('/training')
def training_page():
    """Training interface"""
    return render_template('training.html')

@app.route('/inference')
def inference_page():
    """Real-time inference interface"""
    return render_template('inference.html')

@app.route('/analytics')
def analytics_page():
    """Analytics and visualization interface"""
    return render_template('analytics.html')

@app.route('/models')
def models_page():
    """Model management interface"""
    return render_template('models.html')

# API Endpoints

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """Upload single image for analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and web_interface.allowed_file(file.filename):
            # Generate unique filename
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = web_interface.upload_folder / filename
            file.save(filepath)
            
            # Perform comprehensive analysis
            analysis_result = cv_features.comprehensive_analysis(str(filepath))
            
            # Save results
            result_id = str(uuid.uuid4())
            result_path = web_interface.results_folder / f"{result_id}.json"
            with open(result_path, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            return jsonify({
                'success': True,
                'result_id': result_id,
                'filename': filename,
                'analysis': analysis_result
            })
        
        return jsonify({'success': False, 'error': 'Invalid file type'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    """Upload dataset for training"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and file.filename.endswith('.zip'):
            # Generate unique folder name
            dataset_id = str(uuid.uuid4())
            dataset_folder = web_interface.upload_folder / dataset_id
            dataset_folder.mkdir(exist_ok=True)
            
            # Save and extract zip file
            zip_path = dataset_folder / file.filename
            file.save(zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_folder)
            
            # Process dataset
            dataset_stats = dataset_manager.build_complete_dataset()
            
            return jsonify({
                'success': True,
                'dataset_id': dataset_id,
                'stats': dataset_stats,
                'message': 'Dataset uploaded and processed successfully'
            })
        
        return jsonify({'success': False, 'error': 'Please upload a ZIP file containing the dataset'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start ML training process"""
    try:
        data = request.get_json()
        
        training_type = data.get('training_type', 'comprehensive')
        dataset_path = data.get('dataset_path', '/workspaces/ML-pmfby/dataset')
        
        # Start training in background thread
        session_id = str(uuid.uuid4())
        
        def training_worker():
            global training_status
            training_status = {"is_training": True, "progress": 0, "current_task": "Initializing..."}
            
            try:
                if training_type == 'comprehensive':
                    # Train all algorithms
                    training_status["current_task"] = "Training all algorithms..."
                    results = ml_training.train_all_algorithms(dataset_path)
                    
                elif training_type == 'automl':
                    # AutoML pipeline
                    training_status["current_task"] = "Running AutoML pipeline..."
                    results = ml_training.auto_ml_pipeline(dataset_path)
                    
                elif training_type == 'ensemble':
                    # Ensemble training
                    training_status["current_task"] = "Training ensemble models..."
                    results = ml_training.ensemble_training(dataset_path)
                
                # Save results
                result_path = web_interface.results_folder / f"training_{session_id}.json"
                with open(result_path, 'w') as f:
                    # Convert results to JSON-serializable format
                    serializable_results = convert_to_serializable(results)
                    json.dump(serializable_results, f, indent=2)
                
                training_status = {"is_training": False, "progress": 100, "current_task": "Training completed!"}
                
                # Emit completion event
                socketio.emit('training_complete', {
                    'session_id': session_id,
                    'results': serializable_results
                })
                
            except Exception as e:
                training_status = {"is_training": False, "progress": 0, "current_task": f"Error: {e}"}
                socketio.emit('training_error', {'error': str(e)})
        
        # Start training thread
        training_thread = threading.Thread(target=training_worker)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Training started in background'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/training_status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/api/real_time_inference', methods=['POST'])
def real_time_inference():
    """Perform real-time inference on uploaded image"""
    try:
        data = request.get_json()
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        image_array = np.array(pil_image)
        
        # Perform inference using multiple models
        results = {}
        
        # Object detection
        detection_result = cv_features.object_detection(image_array)
        results['detection'] = detection_result
        
        # Quality assessment
        quality_result = cv_features.quality_assessment(image_array)
        results['quality'] = quality_result
        
        # Feature extraction
        features = cv_features.feature_extraction(image_array)
        results['features'] = features
        
        # ML prediction if available
        try:
            ml_prediction = ml_training.predict_with_best_model(image_array.reshape(1, -1))
            results['ml_prediction'] = ml_prediction
        except:
            results['ml_prediction'] = {'error': 'No trained model available'}
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset_stats')
def get_dataset_stats():
    """Get dataset statistics"""
    try:
        stats_file = Path('/workspaces/ML-pmfby/dataset/statistics/dataset_stats.json')
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'No dataset statistics available'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model_list')
def get_model_list():
    """Get list of available models"""
    try:
        models_path = Path('/workspaces/ML-pmfby/models')
        models = []
        
        for model_file in models_path.glob('*.pt'):
            model_info = {
                'name': model_file.stem,
                'path': str(model_file),
                'size_mb': model_file.stat().st_size / (1024*1024),
                'created': datetime.fromtimestamp(model_file.stat().st_ctime).isoformat()
            }
            models.append(model_info)
        
        return jsonify({'success': True, 'models': models})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analytics/performance')
def get_performance_analytics():
    """Get model performance analytics"""
    try:
        # Generate performance visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sample performance data
        models = ['YOLO', 'ResNet', 'Random Forest', 'XGBoost', 'SVM']
        accuracy = [0.92, 0.89, 0.85, 0.88, 0.82]
        precision = [0.91, 0.87, 0.84, 0.86, 0.80]
        recall = [0.90, 0.88, 0.83, 0.87, 0.81]
        f1_score = [0.905, 0.875, 0.835, 0.865, 0.805]
        
        # Accuracy comparison
        axes[0, 0].bar(models, accuracy, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision-Recall
        axes[0, 1].scatter(precision, recall, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[0, 1].annotate(model, (precision[i], recall[i]))
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Precision vs Recall')
        
        # F1 Score
        axes[1, 0].plot(models, f1_score, marker='o', linewidth=2, markersize=8)
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall metrics heatmap
        metrics_data = np.array([accuracy, precision, recall, f1_score])
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        im = axes[1, 1].imshow(metrics_data, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xticks(range(len(models)))
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].set_yticks(range(len(metrics_labels)))
        axes[1, 1].set_yticklabels(metrics_labels)
        axes[1, 1].set_title('Performance Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = web_interface.results_folder / 'performance_analytics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return jsonify({
            'success': True,
            'plot_path': f'/results/performance_analytics.png',
            'metrics': {
                'models': models,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/build_dataset')
def build_dataset():
    """Build complete dataset from scratch"""
    try:
        def dataset_worker():
            global training_status
            training_status = {"is_training": True, "progress": 0, "current_task": "Building dataset..."}
            
            try:
                # Build dataset
                stats = dataset_manager.build_complete_dataset()
                
                training_status = {"is_training": False, "progress": 100, "current_task": "Dataset built successfully!"}
                
                socketio.emit('dataset_complete', {'stats': stats})
                
            except Exception as e:
                training_status = {"is_training": False, "progress": 0, "current_task": f"Error: {e}"}
                socketio.emit('dataset_error', {'error': str(e)})
        
        # Start dataset building thread
        dataset_thread = threading.Thread(target=dataset_worker)
        dataset_thread.daemon = True
        dataset_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Dataset building started in background'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Static file serving
@app.route('/results/<filename>')
def serve_results(filename):
    """Serve result files"""
    return send_from_directory(web_interface.results_folder, filename)

@app.route('/uploads/<filename>')
def serve_uploads(filename):
    """Serve uploaded files"""
    return send_from_directory(web_interface.upload_folder, filename)

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to ML-PMFBY system'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_training_update')
def handle_training_update():
    emit('training_status', training_status)

# Helper functions
def convert_to_serializable(obj):
    """Convert objects to JSON serializable format"""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    else:
        return obj

if __name__ == '__main__':
    print("🚀 Starting ML-PMFBY Web Interface...")
    print("🌐 Access the interface at: http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000/")
    print("📤 Upload: http://localhost:5000/upload")
    print("🎯 Training: http://localhost:5000/training")
    print("🔮 Inference: http://localhost:5000/inference")
    print("📈 Analytics: http://localhost:5000/analytics")
    
    # Start real-time training pipeline
    training_pipeline.start_real_time_training()
    
    # Run Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)