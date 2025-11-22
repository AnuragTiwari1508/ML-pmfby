"""
Geotag Validator
Extract and validate GPS coordinates from image EXIF data
No external APIs - pure Python implementation
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json
import argparse
from typing import Dict, Optional, Tuple
from datetime import datetime


class GeotagValidator:
    """
    Extract and validate GPS coordinates from image EXIF data.
    Check if location is within expected boundaries.
    """
    
    def __init__(self, expected_bounds: Optional[Dict] = None):
        """
        Args:
            expected_bounds: Dict with 'min_lat', 'max_lat', 'min_lon', 'max_lon'
                           If None, validation will only check if GPS exists
        """
        self.expected_bounds = expected_bounds
    
    def extract_gps(self, image_path: str) -> Optional[Dict]:
        """
        Extract GPS coordinates from image EXIF data.
        
        Args:
            image_path: Path to image file
        
        Returns:
            dict or None: {
                'latitude': float,
                'longitude': float,
                'altitude': float (optional),
                'timestamp': str (optional),
                'raw_gps': dict
            }
        """
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            
            if not exif_data:
                return None
            
            # Extract GPS info
            gps_info = {}
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'GPSInfo':
                    for gps_tag in value:
                        sub_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        gps_info[sub_tag_name] = value[gps_tag]
            
            if not gps_info:
                return None
            
            # Parse GPS coordinates
            latitude = self._parse_gps_coordinate(
                gps_info.get('GPSLatitude'),
                gps_info.get('GPSLatitudeRef', 'N')
            )
            
            longitude = self._parse_gps_coordinate(
                gps_info.get('GPSLongitude'),
                gps_info.get('GPSLongitudeRef', 'E')
            )
            
            if latitude is None or longitude is None:
                return None
            
            result = {
                'latitude': latitude,
                'longitude': longitude,
                'raw_gps': gps_info
            }
            
            # Optional: altitude
            if 'GPSAltitude' in gps_info:
                altitude = gps_info['GPSAltitude']
                if isinstance(altitude, tuple):
                    result['altitude'] = float(altitude[0]) / float(altitude[1])
                else:
                    result['altitude'] = float(altitude)
            
            # Optional: GPS timestamp
            if 'GPSDateStamp' in gps_info and 'GPSTimeStamp' in gps_info:
                date_str = gps_info['GPSDateStamp']
                time_tuple = gps_info['GPSTimeStamp']
                time_str = f"{int(time_tuple[0])}:{int(time_tuple[1])}:{int(time_tuple[2])}"
                result['timestamp'] = f"{date_str} {time_str}"
            
            return result
        
        except Exception as e:
            print(f"⚠️ Error extracting GPS: {e}")
            return None
    
    def _parse_gps_coordinate(self, coord_tuple, ref) -> Optional[float]:
        """
        Convert GPS coordinate from EXIF format to decimal degrees.
        
        Args:
            coord_tuple: Tuple of (degrees, minutes, seconds)
            ref: Reference ('N', 'S', 'E', 'W')
        
        Returns:
            float: Decimal degrees
        """
        if not coord_tuple:
            return None
        
        try:
            # Extract degrees, minutes, seconds
            degrees = float(coord_tuple[0])
            minutes = float(coord_tuple[1])
            seconds = float(coord_tuple[2])
            
            # Convert to decimal degrees
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            # Apply reference (negative for S and W)
            if ref in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        
        except Exception as e:
            print(f"⚠️ Error parsing coordinate: {e}")
            return None
    
    def validate(self, gps_data: Dict) -> Dict:
        """
        Validate GPS coordinates against expected bounds.
        
        Args:
            gps_data: GPS data dict from extract_gps()
        
        Returns:
            dict: {
                'is_valid': bool,
                'has_gps': bool,
                'in_bounds': bool,
                'distance_from_center': float (km, if bounds provided),
                'issues': list of issue strings
            }
        """
        result = {
            'is_valid': False,
            'has_gps': gps_data is not None,
            'in_bounds': False,
            'issues': []
        }
        
        if not gps_data:
            result['issues'].append("No GPS data found in image")
            return result
        
        lat = gps_data['latitude']
        lon = gps_data['longitude']
        
        # Check basic validity
        if not (-90 <= lat <= 90):
            result['issues'].append(f"Invalid latitude: {lat}")
        if not (-180 <= lon <= 180):
            result['issues'].append(f"Invalid longitude: {lon}")
        
        if result['issues']:
            return result
        
        # Check bounds if provided
        if self.expected_bounds:
            in_lat_bounds = (self.expected_bounds['min_lat'] <= lat <= 
                           self.expected_bounds['max_lat'])
            in_lon_bounds = (self.expected_bounds['min_lon'] <= lon <= 
                           self.expected_bounds['max_lon'])
            
            result['in_bounds'] = in_lat_bounds and in_lon_bounds
            
            if not result['in_bounds']:
                result['issues'].append("Location outside expected area")
            
            # Calculate distance from center
            center_lat = (self.expected_bounds['min_lat'] + 
                         self.expected_bounds['max_lat']) / 2
            center_lon = (self.expected_bounds['min_lon'] + 
                         self.expected_bounds['max_lon']) / 2
            
            result['distance_from_center_km'] = self._haversine_distance(
                lat, lon, center_lat, center_lon
            )
        else:
            # No bounds check, just verify GPS exists
            result['in_bounds'] = True
        
        result['is_valid'] = result['has_gps'] and not result['issues']
        
        return result
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula.
        
        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate
        
        Returns:
            float: Distance in kilometers
        """
        R = 6371  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def get_location_string(self, gps_data: Dict) -> str:
        """
        Get human-readable location string.
        
        Args:
            gps_data: GPS data dict
        
        Returns:
            str: Formatted location string
        """
        if not gps_data:
            return "No GPS data"
        
        lat = gps_data['latitude']
        lon = gps_data['longitude']
        
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        
        location_str = f"{abs(lat):.6f}°{lat_dir}, {abs(lon):.6f}°{lon_dir}"
        
        if 'altitude' in gps_data:
            location_str += f", {gps_data['altitude']:.1f}m altitude"
        
        return location_str


def main():
    """Command-line interface for geotag validation."""
    parser = argparse.ArgumentParser(description='Geotag Validator for PMFBY')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--bounds', type=str, help='Expected bounds JSON file')
    parser.add_argument('--min-lat', type=float, help='Minimum latitude')
    parser.add_argument('--max-lat', type=float, help='Maximum latitude')
    parser.add_argument('--min-lon', type=float, help='Minimum longitude')
    parser.add_argument('--max-lon', type=float, help='Maximum longitude')
    
    args = parser.parse_args()
    
    # Load or create bounds
    bounds = None
    if args.bounds and Path(args.bounds).exists():
        with open(args.bounds, 'r') as f:
            bounds = json.load(f)
    elif all([args.min_lat, args.max_lat, args.min_lon, args.max_lon]):
        bounds = {
            'min_lat': args.min_lat,
            'max_lat': args.max_lat,
            'min_lon': args.min_lon,
            'max_lon': args.max_lon
        }
    
    # Initialize validator
    validator = GeotagValidator(bounds)
    
    # Extract GPS
    print(f"\n📍 Extracting GPS from: {args.image}")
    gps_data = validator.extract_gps(args.image)
    
    if not gps_data:
        print("❌ No GPS data found in image")
        return
    
    # Display GPS info
    location_str = validator.get_location_string(gps_data)
    print(f"\n✅ GPS Found:")
    print(f"  Location: {location_str}")
    
    if 'timestamp' in gps_data:
        print(f"  Timestamp: {gps_data['timestamp']}")
    
    # Validate
    validation = validator.validate(gps_data)
    
    print(f"\n🔍 Validation:")
    print(f"  Has GPS: {validation['has_gps']}")
    print(f"  Is Valid: {validation['is_valid']}")
    
    if bounds:
        print(f"  In Bounds: {validation['in_bounds']}")
        if 'distance_from_center_km' in validation:
            print(f"  Distance from center: {validation['distance_from_center_km']:.2f} km")
    
    if validation['issues']:
        print(f"\n⚠️ Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")


if __name__ == '__main__':
    main()
