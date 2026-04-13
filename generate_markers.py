"""
Generate ArUco markers that you can print and attach to objects
This creates printable marker images
"""

import cv2
import os

def generate_aruco_markers(output_dir='markers', num_markers=10, marker_size=200):
    """
    Generate ArUco markers and save as images
    
    Args:
        output_dir: Directory to save marker images
        num_markers: Number of markers to generate
        marker_size: Size of marker in pixels
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use DICT_6X6_250 (can detect up to 250 different markers)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    print(f"Generating {num_markers} ArUco markers...")
    
    for marker_id in range(num_markers):
        # Generate marker image
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        
        # Save marker
        filename = os.path.join(output_dir, f'marker_{marker_id:03d}.png')
        cv2.imwrite(filename, marker_image)
        print(f"  Generated: {filename}")
    
    print(f"\nMarkers saved to '{output_dir}' folder")
    print(f"Print these markers and attach them to the objects/people you want to track")
    print(f"Markers ID 0-{num_markers-1} are ready to use")


if __name__ == '__main__':
    generate_aruco_markers(output_dir='markers', num_markers=10, marker_size=200)
