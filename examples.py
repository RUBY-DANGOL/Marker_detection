"""
Example usage patterns for the Person Tracker with Markers
"""

from person_tracker import PersonTrackerWithMarkers
import cv2

# Example 1: Basic webcam tracking (simplest)
def example_webcam_tracking():
    """Track persons from webcam"""
    print("Starting webcam tracking...")
    tracker = PersonTrackerWithMarkers(model_name='yolov8n.pt')
    tracker.process_video(video_source=0)


# Example 2: Track from video file
def example_video_file_tracking(video_path):
    """Track persons from a video file"""
    print(f"Tracking from: {video_path}")
    tracker = PersonTrackerWithMarkers(model_name='yolov8n.pt')
    tracker.process_video(video_source=video_path)


# Example 3: Track and save output
def example_save_output(input_source, output_file):
    """Track and save results to video file"""
    print(f"Tracking and saving to: {output_file}")
    tracker = PersonTrackerWithMarkers(model_name='yolov8n.pt')
    tracker.process_video(video_source=input_source, output_path=output_file)


# Example 4: Use larger model for better accuracy (slower)
def example_large_model():
    """Use larger YOLO model for better accuracy"""
    print("Starting tracker with large model (slower but more accurate)...")
    tracker = PersonTrackerWithMarkers(model_name='yolov8m.pt')  # medium model
    tracker.process_video(video_source=0)


# Example 5: Custom settings
def example_custom_settings():
    """Create tracker with custom settings"""
    tracker = PersonTrackerWithMarkers(model_name='yolov8n.pt')
    
    # Customize tracking parameters
    tracker.max_distance = 80  # Increase tolerance for matching
    tracker.max_frames_to_skip = 50  # Wait longer before removing lost tracks
    
    print("Tracking with custom settings...")
    tracker.process_video(video_source=0)


if __name__ == '__main__':
    import sys
    
    print("Person Tracker - Example Usage")
    print("=" * 50)
    print("\nAvailable examples:")
    print("1. example_webcam_tracking() - Basic webcam tracking")
    print("2. example_video_file_tracking('path/to/video.mp4') - Track from file")
    print("3. example_save_output(0, 'output.mp4') - Track and save")
    print("4. example_large_model() - Use larger model")
    print("5. example_custom_settings() - Custom tracking parameters")
    
    print("\nRunning basic webcam example...")
    print("Press 'q' in the video window to stop")
    print("-" * 50)
    
    # Run basic example
    example_webcam_tracking()
