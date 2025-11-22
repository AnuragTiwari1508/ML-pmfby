"""
Create Sample Test Image
"""

import numpy as np
try:
    import cv2
    
    # Create a realistic test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add gradient background (simulated field)
    for i in range(480):
        img[i, :] = [30 + i//5, 80 + i//5, 20 + i//4]
    
    # Add some "crop-like" rectangles
    cv2.rectangle(img, (150, 150), (250, 300), (40, 150, 40), -1)
    cv2.rectangle(img, (350, 200), (450, 320), (35, 140, 35), -1)
    
    # Add noise for realism
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Save
    cv2.imwrite('test.jpg', img)
    print("✅ Created test.jpg")
    
except ImportError:
    # Fallback without OpenCV
    from PIL import Image
    img = Image.new('RGB', (640, 480), color=(50, 120, 40))
    img.save('test.jpg')
    print("✅ Created test.jpg (basic version)")
