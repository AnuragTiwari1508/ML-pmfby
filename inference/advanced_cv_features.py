"""
Advanced Computer Vision Features for PMFBY
Includes object detection, segmentation, classification, feature extraction, and analysis
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import json
from pathlib import Path
from datetime import datetime
import albumentations as A
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class AdvancedCVFeatures:
    """Complete computer vision pipeline with advanced features"""
    
    def __init__(self, models_path="/workspaces/ML-pmfby/models"):
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        self.initialize_models()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def initialize_models(self):
        """Initialize all CV models"""
        print("🤖 Initializing advanced CV models...")
        
        # 1. Object Detection (YOLO)
        try:
            self.models['detector'] = YOLO('yolov8n.pt')
            print("✅ Object detector initialized")
        except Exception as e:
            print(f"⚠️  Object detector failed: {e}")
        
        # 2. Image Classification (ResNet)
        try:
            self.models['classifier'] = models.resnet50(pretrained=True)
            self.models['classifier'].fc = nn.Linear(2048, 5)  # 5 classes for crops
            self.models['classifier'].to(self.device)
            print("✅ Image classifier initialized")
        except Exception as e:
            print(f"⚠️  Classifier failed: {e}")
        
        # 3. Segmentation Model (U-Net)
        try:
            self.models['segmentation'] = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet", 
                in_channels=3,
                classes=5,
            )
            self.models['segmentation'].to(self.device)
            print("✅ Segmentation model initialized")
        except Exception as e:
            print(f"⚠️  Segmentation model failed: {e}")
        
        # 4. Feature Extractor (Custom CNN)
        try:
            self.models['feature_extractor'] = AdvancedFeatureExtractor()
            self.models['feature_extractor'].to(self.device)
            print("✅ Feature extractor initialized")
        except Exception as e:
            print(f"⚠️  Feature extractor failed: {e}")
        
        # 5. Quality Assessor
        try:
            self.models['quality_assessor'] = ImageQualityAssessor()
            self.models['quality_assessor'].to(self.device)
            print("✅ Quality assessor initialized")
        except Exception as e:
            print(f"⚠️  Quality assessor failed: {e}")
    
    def object_detection(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """Advanced object detection with post-processing"""
        try:
            # Run YOLO detection
            results = self.models['detector'](image)
            
            # Extract results
            boxes = []
            classes = []
            confidences = []
            class_names = ['crop', 'damage', 'plant', 'field', 'other']
            
            if results[0].boxes is not None:
                for i in range(len(results[0].boxes)):
                    conf = results[0].boxes.conf[i].item()
                    if conf >= confidence_threshold:
                        box = results[0].boxes.xyxy[i].cpu().numpy()
                        cls = int(results[0].boxes.cls[i].item())
                        
                        boxes.append(box.tolist())
                        classes.append(cls)
                        confidences.append(conf)
            
            # Advanced post-processing
            filtered_detections = self._post_process_detections(boxes, classes, confidences, image.shape)
            
            # Generate analysis
            analysis = self._analyze_detections(filtered_detections, image.shape)
            
            return {
                'detections': filtered_detections,
                'analysis': analysis,
                'total_objects': len(filtered_detections),
                'class_distribution': self._get_class_distribution(classes, class_names)
            }
            
        except Exception as e:
            return {'error': f"Detection failed: {e}"}
    
    def image_segmentation(self, image: np.ndarray) -> Dict:
        """Advanced image segmentation"""
        try:
            # Preprocess
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Segment
            with torch.no_grad():
                segmentation_output = self.models['segmentation'](img_tensor)
                segmentation_mask = torch.argmax(segmentation_output, dim=1).squeeze().cpu().numpy()
            
            # Post-process segmentation
            processed_mask = self._post_process_segmentation(segmentation_mask, image.shape[:2])
            
            # Analyze segments
            segment_analysis = self._analyze_segments(processed_mask, image)
            
            # Generate colored mask
            colored_mask = self._create_colored_mask(processed_mask)
            
            return {
                'segmentation_mask': processed_mask.tolist(),
                'colored_mask': colored_mask,
                'segment_analysis': segment_analysis,
                'class_areas': self._calculate_class_areas(processed_mask)
            }
            
        except Exception as e:
            return {'error': f"Segmentation failed: {e}"}
    
    def feature_extraction(self, image: np.ndarray) -> Dict:
        """Extract comprehensive image features"""
        try:
            features = {}
            
            # 1. Deep learning features
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                deep_features = self.models['feature_extractor'](img_tensor)
                features['deep_features'] = deep_features.cpu().numpy().flatten().tolist()
            
            # 2. Traditional computer vision features
            traditional_features = self._extract_traditional_features(image)
            features.update(traditional_features)
            
            # 3. Color analysis
            color_features = self._extract_color_features(image)
            features.update(color_features)
            
            # 4. Texture analysis
            texture_features = self._extract_texture_features(image)
            features.update(texture_features)
            
            # 5. Shape analysis
            shape_features = self._extract_shape_features(image)
            features.update(shape_features)
            
            return features
            
        except Exception as e:
            return {'error': f"Feature extraction failed: {e}"}
    
    def quality_assessment(self, image: np.ndarray) -> Dict:
        """Comprehensive image quality assessment"""
        try:
            # Technical quality metrics
            technical_quality = self._assess_technical_quality(image)
            
            # Content quality using deep learning
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                quality_scores = self.models['quality_assessor'](img_tensor)
                quality_scores = torch.softmax(quality_scores, dim=1).cpu().numpy()[0]
            
            # Combined assessment
            overall_quality = self._combine_quality_assessments(technical_quality, quality_scores)
            
            return {
                'technical_quality': technical_quality,
                'content_quality': {
                    'poor': float(quality_scores[0]),
                    'fair': float(quality_scores[1]), 
                    'good': float(quality_scores[2]),
                    'excellent': float(quality_scores[3])
                },
                'overall_quality': overall_quality,
                'recommendations': self._generate_quality_recommendations(technical_quality, quality_scores)
            }
            
        except Exception as e:
            return {'error': f"Quality assessment failed: {e}"}
    
    def comprehensive_analysis(self, image_path: str) -> Dict:
        """Complete image analysis pipeline"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            print("🔍 Running comprehensive analysis...")
            
            # 1. Object Detection
            detection_results = self.object_detection(image_rgb)
            
            # 2. Segmentation
            segmentation_results = self.image_segmentation(image_rgb)
            
            # 3. Feature Extraction
            features = self.feature_extraction(image_rgb)
            
            # 4. Quality Assessment
            quality_results = self.quality_assessment(image_rgb)
            
            # 5. Advanced Analytics
            analytics = self._perform_advanced_analytics(image_rgb, detection_results, segmentation_results)
            
            # Compile comprehensive report
            analysis_report = {
                'image_info': {
                    'path': image_path,
                    'shape': image.shape,
                    'size_mb': Path(image_path).stat().st_size / (1024*1024),
                    'timestamp': datetime.now().isoformat()
                },
                'detection': detection_results,
                'segmentation': segmentation_results,
                'features': features,
                'quality': quality_results,
                'analytics': analytics,
                'summary': self._generate_analysis_summary(detection_results, segmentation_results, quality_results)
            }
            
            return analysis_report
            
        except Exception as e:
            return {'error': f"Comprehensive analysis failed: {e}"}
    
    def _post_process_detections(self, boxes: List, classes: List, confidences: List, img_shape: Tuple) -> List[Dict]:
        """Advanced post-processing of detections"""
        detections = []
        
        for box, cls, conf in zip(boxes, classes, confidences):
            # Calculate additional metrics
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # Relative size
            img_area = img_shape[0] * img_shape[1]
            relative_size = area / img_area
            
            detection = {
                'box': box,
                'class': cls,
                'confidence': conf,
                'area': area,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'relative_size': relative_size,
                'center': [(box[0] + box[2])/2, (box[1] + box[3])/2]
            }
            
            detections.append(detection)
        
        # Apply NMS-like filtering
        filtered_detections = self._apply_advanced_nms(detections)
        
        return filtered_detections
    
    def _apply_advanced_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Advanced Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [det for det in detections 
                         if self._calculate_iou(current['box'], det['box']) < iou_threshold]
        
        return keep
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1]) 
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _extract_traditional_features(self, image: np.ndarray) -> Dict:
        """Extract traditional computer vision features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Edge features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        features['edge_mean'] = np.mean(edges)
        
        # 2. Corner features
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        features['corner_count'] = np.sum(corners > 0.01 * corners.max())
        
        # 3. Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['contour_count'] = len(contours)
        
        if contours:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            features['max_contour_area'] = max(areas)
            features['mean_contour_area'] = np.mean(areas)
        
        # 4. Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_magnitude_mean'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        return features
    
    def _extract_color_features(self, image: np.ndarray) -> Dict:
        """Extract comprehensive color features"""
        features = {}
        
        # RGB statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i]
            features[f'{channel}_mean'] = float(np.mean(channel_data))
            features[f'{channel}_std'] = float(np.std(channel_data))
            features[f'{channel}_skew'] = float(self._calculate_skewness(channel_data))
        
        # HSV features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(['H', 'S', 'V']):
            channel_data = hsv[:, :, i]
            features[f'{channel}_mean'] = float(np.mean(channel_data))
            features[f'{channel}_std'] = float(np.std(channel_data))
        
        # Color diversity
        features['color_diversity'] = self._calculate_color_diversity(image)
        
        # Dominant colors
        dominant_colors = self._get_dominant_colors(image, k=3)
        features['dominant_colors'] = dominant_colors
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict:
        """Extract texture features using various methods"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. LBP (Local Binary Patterns) - simplified version
        features.update(self._calculate_lbp_features(gray))
        
        # 2. GLCM features (simplified)
        features.update(self._calculate_glcm_features(gray))
        
        # 3. Gabor filter responses
        features.update(self._calculate_gabor_features(gray))
        
        return features
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict:
        """Extract shape-based features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Shape metrics
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                features['shape_compactness'] = 4 * np.pi * area / (perimeter**2)
            
            # Bounding box features
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['shape_aspect_ratio'] = w / h if h > 0 else 0
            features['shape_extent'] = area / (w * h) if w * h > 0 else 0
            
            # Convex hull features
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['shape_solidity'] = area / hull_area if hull_area > 0 else 0
        
        return features
    
    def _assess_technical_quality(self, image: np.ndarray) -> Dict:
        """Assess technical image quality"""
        quality = {}
        
        # 1. Blur detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality['blur_score'] = float(blur_score)
        quality['is_blurry'] = blur_score < 100
        
        # 2. Brightness analysis
        brightness = np.mean(gray)
        quality['brightness'] = float(brightness)
        quality['is_too_dark'] = brightness < 50
        quality['is_too_bright'] = brightness > 200
        
        # 3. Contrast analysis
        contrast = np.std(gray)
        quality['contrast'] = float(contrast)
        quality['is_low_contrast'] = contrast < 30
        
        # 4. Noise estimation
        noise_level = self._estimate_noise(gray)
        quality['noise_level'] = float(noise_level)
        quality['is_noisy'] = noise_level > 20
        
        # 5. Sharpness
        sharpness = self._calculate_sharpness(gray)
        quality['sharpness'] = float(sharpness)
        
        return quality
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_color_diversity(self, image: np.ndarray) -> float:
        """Calculate color diversity using entropy"""
        # Reduce color space for diversity calculation
        reduced = image // 32  # Reduce to 8 levels per channel
        unique_colors = np.unique(reduced.reshape(-1, 3), axis=0)
        return len(unique_colors) / (8**3)  # Normalized diversity
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List:
        """Get dominant colors using K-means clustering"""
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return colors.tolist()
    
    def _calculate_lbp_features(self, gray: np.ndarray) -> Dict:
        """Calculate simplified Local Binary Pattern features"""
        # Simplified LBP calculation
        lbp = np.zeros_like(gray)
        
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                binary = 0
                
                # 8-neighborhood
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary += 2**k
                
                lbp[i, j] = binary
        
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        
        return {
            'lbp_uniformity': float(np.max(hist)),
            'lbp_entropy': float(-np.sum(hist * np.log2(hist + 1e-7)))
        }
    
    def _calculate_glcm_features(self, gray: np.ndarray) -> Dict:
        """Calculate simplified GLCM features"""
        # Simplified GLCM calculation
        # This is a basic implementation - for production use scikit-image
        
        # Quantize to reduce computation
        quantized = (gray // 32).astype(np.uint8)
        
        # Calculate co-occurrence matrix for horizontal direction
        glcm = np.zeros((8, 8))
        for i in range(quantized.shape[0]):
            for j in range(quantized.shape[1]-1):
                glcm[quantized[i, j], quantized[i, j+1]] += 1
        
        # Normalize
        glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
        
        # Calculate features
        contrast = 0
        homogeneity = 0
        energy = 0
        
        for i in range(8):
            for j in range(8):
                contrast += glcm[i, j] * (i - j)**2
                homogeneity += glcm[i, j] / (1 + abs(i - j))
                energy += glcm[i, j]**2
        
        return {
            'glcm_contrast': float(contrast),
            'glcm_homogeneity': float(homogeneity),
            'glcm_energy': float(energy)
        }
    
    def _calculate_gabor_features(self, gray: np.ndarray) -> Dict:
        """Calculate Gabor filter features"""
        features = {}
        
        # Multiple orientations
        orientations = [0, 45, 90, 135]
        
        for angle in orientations:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            
            features[f'gabor_mean_{angle}'] = float(np.mean(filtered))
            features[f'gabor_std_{angle}'] = float(np.std(filtered))
        
        return features
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level in image"""
        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate image sharpness"""
        return cv2.Laplacian(gray, cv2.CV_64F).var()

class AdvancedFeatureExtractor(nn.Module):
    """Advanced CNN-based feature extractor"""
    
    def __init__(self, output_dim=512):
        super().__init__()
        
        # Use pre-trained ResNet as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom feature layers
        self.feature_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_layers(features)
        return features

class ImageQualityAssessor(nn.Module):
    """Neural network for image quality assessment"""
    
    def __init__(self):
        super().__init__()
        
        # Use MobileNet for efficiency
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Poor, Fair, Good, Excellent
        )
    
    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Initialize advanced CV features
    cv_features = AdvancedCVFeatures()
    
    print("🚀 Advanced Computer Vision Features initialized!")
    print("📸 Use cv_features.comprehensive_analysis() for complete analysis")
    print("🎯 Use cv_features.object_detection() for object detection") 
    print("🎨 Use cv_features.image_segmentation() for segmentation")
    print("🔍 Use cv_features.feature_extraction() for feature extraction")
    print("⭐ Use cv_features.quality_assessment() for quality analysis")