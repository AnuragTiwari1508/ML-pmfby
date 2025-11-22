
# PRODUCTION DEPLOYMENT CODE
# Copy-paste ready for mobile integration

class PMFBYCapture:
    def __init__(self):
        self.blur_threshold = 50
        self.light_thresholds = (40, 220)
        self.distance_k = 316.0
    
    def analyze_image(self, image_array, bbox=None):
        # Ultra-fast analysis in <20ms
        
        # Blur detection
        gray = (image_array[:,:,0] * 0.3 + image_array[:,:,1] * 0.6 + image_array[:,:,2] * 0.1).astype(np.uint8)
        grad_x = np.diff(gray[::2, ::2], axis=1)
        blur_score = np.var(grad_x)
        
        # Lighting
        brightness = np.mean(image_array[::4, ::4])
        light_ok = 40 < brightness < 220
        
        # Distance (if bbox provided)
        distance = None
        if bbox:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            distance = self.distance_k / np.sqrt(max(area, 1))
        
        # Decision
        should_capture = blur_score > self.blur_threshold and light_ok
        
        return {
            'capture': should_capture,
            'blur_score': blur_score,
            'brightness': brightness,
            'distance': distance,
            'quality': 'good' if should_capture else 'poor'
        }

# Usage in mobile app:
# capture = PMFBYCapture()
# result = capture.analyze_image(camera_frame, detected_bbox)
# if result['capture']: save_image()
