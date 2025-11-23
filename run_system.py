
#!/usr/bin/env python3
"""
PMFBY ML System - Complete Runnable ML Pipeline
Real YOLO training for agricultural object detection
"""

import sys
import os
from pathlib import Path
import time

def main():
    print("🌾 PMFBY ML System - Agricultural Object Detection")
    print("🤖 Real YOLO Training Pipeline")
    print("=" * 60)
    
    try:
        # Import the simple system (works without OpenCV display issues)
        from simple_ml_system import SimplePMFBYSystem
        
        print("\n🚀 Initializing Simple ML System...")
        system = SimplePMFBYSystem()
        
        print("\n🎯 What would you like to do?")
        print("1. 🔥 Complete Pipeline (Create Dataset + Train YOLO + Test)")
        print("2. 📊 Create Dataset Only")
        print("3. 🎯 Train YOLO Model")
        print("4. 🔮 Run Inference")
        print("5. ⚡ Quick Demo (Small dataset + Fast training)")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        start_time = time.time()
        
        if choice == "1":
            print("\n🔥 Running Complete ML Pipeline...")
            print("This will create dataset, train YOLO model, and test inference")
            print("⏱️  This may take 10-30 minutes depending on your system...")
            
            confirm = input("Continue? (y/n): ").lower()
            if confirm == 'y':
                results = system.full_pipeline()
                print(f"\n🎉 Complete Pipeline Finished!")
                print(f"📊 Results: {results}")
            
        elif choice == "2":
            print("\n📊 Creating Agricultural Dataset...")
            num_images = int(input("Number of images (50-500, default 100): ") or "100")
            
            count = system.create_synthetic_dataset(num_images=num_images)
            print(f"✅ Dataset created with {count} images!")
            
        elif choice == "3":
            print("\n🎯 Training YOLO Model...")
            epochs = int(input("Number of epochs (10-100, default 20): ") or "20")
            batch_size = int(input("Batch size (4-16, default 8): ") or "8")
            
            print(f"⏱️  Training for {epochs} epochs with batch size {batch_size}")
            print("This may take 10-60 minutes...")
            
            results = system.train_yolo_model(epochs=epochs, batch_size=batch_size)
            print("✅ Model training completed!")
            
        elif choice == "4":
            print("\n🔮 Running Inference...")
            
            # Check if model exists
            model_path = system.models_path / 'pmfby_yolo' / 'weights' / 'best.pt'
            if not model_path.exists():
                print("❌ No trained model found!")
                print("Please train a model first (option 1 or 3)")
                return
            
            # Create demo images and run inference
            demo_dir = system.create_demo_inference_images()
            demo_images = list(demo_dir.glob('*.jpg'))
            
            print(f"📸 Testing on {len(demo_images)} demo images...")
            
            for img_path in demo_images:
                result = system.run_inference(str(img_path))
                if result and result['detections']:
                    print(f"\n🖼️  {img_path.name}: {result['total_objects']} objects detected")
                    for det in result['detections'][:3]:  # Show first 3
                        print(f"   - {det['class']}: {det['confidence']:.3f}")
                else:
                    print(f"\n🖼️  {img_path.name}: No objects detected")
        
        elif choice == "5":
            print("\n⚡ Quick Demo Mode...")
            print("Creating small dataset and fast training for demonstration")
            print("⏱️  This should complete in 3-10 minutes")
            
            # Small dataset
            print("\n📊 Creating 30 sample images...")
            system.create_synthetic_dataset(num_images=30)
            
            # Quick training
            print("\n🎯 Quick training (10 epochs)...")
            system.train_yolo_model(epochs=10, batch_size=4)
            
            # Test inference
            print("\n🔮 Testing inference...")
            demo_dir = system.create_demo_inference_images()
            demo_images = list(demo_dir.glob('*.jpg'))[:2]  # Test 2 images
            
            for img_path in demo_images:
                result = system.run_inference(str(img_path))
                if result:
                    print(f"📸 {img_path.name}: {result['total_objects']} detections")
            
            print("\n✅ Quick demo completed!")
            
        else:
            print("❌ Invalid choice! Please select 1-5")
            return
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {elapsed_time/60:.1f} minutes")
        
        # Show final paths
        print(f"\n📂 Project Structure:")
        print(f"   Dataset: {system.dataset_path}")
        print(f"   Models: {system.models_path}")
        print(f"   Results: {system.results_path}")
        
    except ImportError as e:
        print(f"\n❌ Missing Dependencies: {e}")
        print("\n📦 Please install required packages:")
        print("pip install torch torchvision ultralytics opencv-python numpy pandas matplotlib pyyaml")
        print("\nOr run: pip install -r requirements.txt")
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nFor debugging, here's the full error:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()