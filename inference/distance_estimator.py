"""
Distance Estimator using Bounding Box Area
Estimates distance to object based on calibrated bbox area
No external APIs - pure geometric calculation
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, Optional


class DistanceEstimator:
    """
    Estimate distance to object using bounding box area.
    Requires one-time calibration per device/camera.
    
    Formula: distance = k / sqrt(bbox_area)
    where k is calibration constant.
    """
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Args:
            calibration_file: Path to calibration JSON file
        """
        self.calibration_data = {}
        
        if calibration_file and Path(calibration_file).exists():
            self.load_calibration(calibration_file)
        else:
            # Default calibration (approximate for smartphone at 12MP)
            self.calibration_data = {
                'default': {
                    'k': 50000,  # Calibration constant
                    'min_distance_m': 0.5,
                    'max_distance_m': 3.0,
                    'reference_object_size_m': 0.2,  # 20cm object
                    'device': 'default'
                }
            }
    
    def estimate(self, bbox: list, device: str = 'default') -> Dict:
        """
        Estimate distance based on bounding box.
        
        Args:
            bbox: [xmin, ymin, xmax, ymax]
            device: Device identifier for calibration lookup
        
        Returns:
            dict: {
                'distance_m': float,
                'bbox_area': float,
                'is_in_range': bool,
                'guidance': str
            }
        """
        # Calculate bbox area
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        bbox_area = width * height
        
        # Get calibration for device
        calib = self.calibration_data.get(device, self.calibration_data['default'])
        k = calib['k']
        min_dist = calib['min_distance_m']
        max_dist = calib['max_distance_m']
        
        # Estimate distance
        if bbox_area > 0:
            distance_m = k / np.sqrt(bbox_area)
        else:
            distance_m = max_dist
        
        # Check if in acceptable range
        is_in_range = min_dist <= distance_m <= max_dist
        
        # Generate guidance
        if distance_m < min_dist:
            guidance = f"Move back {min_dist - distance_m:.1f}m"
        elif distance_m > max_dist:
            guidance = f"Move closer {distance_m - max_dist:.1f}m"
        else:
            guidance = "Distance OK ✓"
        
        return {
            'distance_m': float(distance_m),
            'bbox_area': float(bbox_area),
            'is_in_range': is_in_range,
            'guidance': guidance,
            'min_distance_m': min_dist,
            'max_distance_m': max_dist
        }
    
    def calibrate(self, known_distance_m: float, bbox: list, device: str = 'default') -> float:
        """
        Calibrate distance estimator with known measurement.
        
        Args:
            known_distance_m: Actual distance in meters
            bbox: Bounding box at that distance [xmin, ymin, xmax, ymax]
            device: Device identifier
        
        Returns:
            float: Calibration constant k
        """
        xmin, ymin, xmax, ymax = bbox
        bbox_area = (xmax - xmin) * (ymax - ymin)
        
        # Calculate k: distance = k / sqrt(area)
        # So k = distance * sqrt(area)
        k = known_distance_m * np.sqrt(bbox_area)
        
        # Update calibration
        if device not in self.calibration_data:
            self.calibration_data[device] = {
                'min_distance_m': 0.5,
                'max_distance_m': 3.0,
                'reference_object_size_m': 0.2,
                'device': device
            }
        
        self.calibration_data[device]['k'] = float(k)
        
        return k
    
    def save_calibration(self, filepath: str):
        """Save calibration data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        print(f"✅ Calibration saved to: {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibration data from JSON file."""
        with open(filepath, 'r') as f:
            self.calibration_data = json.load(f)
        print(f"✅ Calibration loaded from: {filepath}")
    
    def calibrate_multi_point(self, measurements: list, device: str = 'default') -> float:
        """
        Calibrate using multiple distance measurements.
        
        Args:
            measurements: List of (distance_m, bbox) tuples
            device: Device identifier
        
        Returns:
            float: Average calibration constant
        """
        k_values = []
        
        for distance_m, bbox in measurements:
            k = self.calibrate(distance_m, bbox, device)
            k_values.append(k)
        
        # Use average k
        avg_k = np.mean(k_values)
        self.calibration_data[device]['k'] = float(avg_k)
        
        print(f"📊 Calibration statistics:")
        print(f"  Mean k: {avg_k:.2f}")
        print(f"  Std k: {np.std(k_values):.2f}")
        print(f"  Range: {np.min(k_values):.2f} - {np.max(k_values):.2f}")
        
        return avg_k


def main():
    """Command-line interface for distance estimation."""
    parser = argparse.ArgumentParser(description='Distance Estimator for PMFBY')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Estimate command
    estimate_parser = subparsers.add_parser('estimate', help='Estimate distance')
    estimate_parser.add_argument('--bbox', type=float, nargs=4, required=True,
                                help='Bounding box: xmin ymin xmax ymax')
    estimate_parser.add_argument('--device', type=str, default='default',
                                help='Device identifier')
    estimate_parser.add_argument('--calib', type=str,
                                help='Calibration file')
    
    # Calibrate command
    calib_parser = subparsers.add_parser('calibrate', help='Calibrate distance estimator')
    calib_parser.add_argument('--distance', type=float, required=True,
                             help='Known distance in meters')
    calib_parser.add_argument('--bbox', type=float, nargs=4, required=True,
                             help='Bounding box: xmin ymin xmax ymax')
    calib_parser.add_argument('--device', type=str, default='default',
                             help='Device identifier')
    calib_parser.add_argument('--save', type=str, default='calibration.json',
                             help='Save calibration to file')
    
    args = parser.parse_args()
    
    if args.command == 'estimate':
        # Load calibration if provided
        estimator = DistanceEstimator(args.calib)
        
        # Estimate distance
        result = estimator.estimate(args.bbox, args.device)
        
        print(f"\n📏 Distance Estimation:")
        print(f"  Distance: {result['distance_m']:.2f}m")
        print(f"  Bbox Area: {result['bbox_area']:.0f} pixels²")
        print(f"  In Range: {result['is_in_range']}")
        print(f"  Guidance: {result['guidance']}")
    
    elif args.command == 'calibrate':
        estimator = DistanceEstimator()
        
        # Calibrate
        k = estimator.calibrate(args.distance, args.bbox, args.device)
        
        print(f"\n🎯 Calibration Complete:")
        print(f"  Device: {args.device}")
        print(f"  Calibration constant k: {k:.2f}")
        print(f"  Distance: {args.distance}m")
        print(f"  Bbox area: {(args.bbox[2]-args.bbox[0])*(args.bbox[3]-args.bbox[1]):.0f} pixels²")
        
        # Save calibration
        estimator.save_calibration(args.save)
        
        # Test estimation
        test_result = estimator.estimate(args.bbox, args.device)
        print(f"\n✅ Verification:")
        print(f"  Estimated distance: {test_result['distance_m']:.2f}m")
        print(f"  Error: {abs(test_result['distance_m'] - args.distance):.2f}m")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
