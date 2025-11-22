"""
Dataset Augmentation Tool
Generate 15k+ images from smaller dataset using augmentation
"""

import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import json
import argparse
import shutil
from typing import List, Dict


class DatasetAugmenter:
    """
    Augment crop image dataset to reach target size (15k+).
    Applies various transformations while preserving bounding boxes.
    """
    
    def __init__(self):
        """Initialize augmentation pipeline."""
        self.transform = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=0,
                p=0.5
            ),
            
            # Color & lighting augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Noise & blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            
            # Weather effects
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=10,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.9,
                rain_type='drizzle',
                p=0.1
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=1,
                num_flare_circles_upper=2,
                src_radius=100,
                p=0.1
            ),
            
            # Quality degradation
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
            
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def augment_image(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Dict:
        """
        Augment single image with bounding boxes.
        
        Args:
            image: Input image (numpy array)
            bboxes: List of bounding boxes [[xmin, ymin, xmax, ymax], ...]
            class_labels: List of class IDs for each bbox
        
        Returns:
            dict: {
                'image': augmented image,
                'bboxes': transformed bboxes,
                'class_labels': class labels
            }
        """
        # Apply augmentation
        augmented = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        return augmented
    
    def augment_dataset(
        self,
        input_dir: str,
        output_dir: str,
        annotations_file: str,
        target_count: int = 15000,
        preserve_originals: bool = True
    ):
        """
        Augment entire dataset to reach target count.
        
        Args:
            input_dir: Input images directory
            output_dir: Output directory for augmented images
            annotations_file: CSV or JSON annotations file
            target_count: Target number of images
            preserve_originals: Copy original images to output
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        annotations = self._load_annotations(annotations_file)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Input images: {len(annotations)}")
        print(f"  Target images: {target_count}")
        print(f"  Augmentations needed per image: {target_count // len(annotations)}")
        
        # Calculate augmentations per image
        aug_per_image = max(1, target_count // len(annotations))
        
        # Process each image
        augmented_annotations = []
        
        print(f"\n🔄 Augmenting dataset...")
        for img_info in tqdm(annotations):
            img_path = input_path / img_info['filename']
            
            if not img_path.exists():
                print(f"⚠️ Image not found: {img_path}")
                continue
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Preserve original if requested
            if preserve_originals:
                output_img_path = output_path / img_info['filename']
                cv2.imwrite(str(output_img_path), image)
                augmented_annotations.append(img_info.copy())
            
            # Generate augmentations
            for aug_idx in range(aug_per_image - 1):
                try:
                    # Prepare bboxes
                    bboxes = img_info.get('bboxes', [])
                    class_labels = img_info.get('class_labels', [])
                    
                    if not bboxes:
                        # No bounding boxes, just augment image
                        augmented = self.transform(image=image)
                    else:
                        # Augment with bounding boxes
                        augmented = self.augment_image(image, bboxes, class_labels)
                    
                    # Save augmented image
                    base_name = img_path.stem
                    aug_filename = f"{base_name}_aug{aug_idx:04d}{img_path.suffix}"
                    aug_path = output_path / aug_filename
                    cv2.imwrite(str(aug_path), augmented['image'])
                    
                    # Save annotation
                    aug_info = img_info.copy()
                    aug_info['filename'] = aug_filename
                    if bboxes:
                        aug_info['bboxes'] = augmented['bboxes']
                    augmented_annotations.append(aug_info)
                
                except Exception as e:
                    print(f"\n⚠️ Error augmenting {img_path.name}: {e}")
                    continue
        
        # Save augmented annotations
        self._save_annotations(augmented_annotations, output_path / 'annotations.json')
        
        print(f"\n✅ Augmentation complete!")
        print(f"  Output images: {len(augmented_annotations)}")
        print(f"  Output directory: {output_path}")
    
    def _load_annotations(self, annotations_file: str) -> List[Dict]:
        """Load annotations from CSV or JSON file."""
        ann_path = Path(annotations_file)
        
        if ann_path.suffix == '.json':
            with open(ann_path, 'r') as f:
                return json.load(f)
        
        elif ann_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(ann_path)
            
            # Group by filename
            annotations = []
            for filename, group in df.groupby('filename'):
                bboxes = []
                class_labels = []
                
                for _, row in group.iterrows():
                    if 'xmin' in row and 'ymin' in row:
                        bbox = [
                            float(row['xmin']),
                            float(row['ymin']),
                            float(row['xmax']),
                            float(row['ymax'])
                        ]
                        bboxes.append(bbox)
                        
                        if 'class' in row:
                            class_labels.append(int(row['class']))
                
                annotations.append({
                    'filename': filename,
                    'bboxes': bboxes,
                    'class_labels': class_labels,
                    'metadata': group.iloc[0].to_dict()
                })
            
            return annotations
        
        else:
            raise ValueError(f"Unsupported annotation format: {ann_path.suffix}")
    
    def _save_annotations(self, annotations: List[Dict], output_file: str):
        """Save annotations to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"\n💾 Annotations saved: {output_file}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Dataset Augmentation for PMFBY')
    parser.add_argument('--input', type=str, required=True, help='Input images directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--annotations', type=str, required=True, help='Annotations file (CSV/JSON)')
    parser.add_argument('--target', type=int, default=15000, help='Target number of images')
    parser.add_argument('--no-preserve', action='store_true', help='Do not preserve originals')
    
    args = parser.parse_args()
    
    # Create augmenter
    augmenter = DatasetAugmenter()
    
    # Augment dataset
    augmenter.augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        annotations_file=args.annotations,
        target_count=args.target,
        preserve_originals=not args.no_preserve
    )


if __name__ == '__main__':
    main()
