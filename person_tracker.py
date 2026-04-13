"""
Person Tracker with YOLOv8 Detection and ArUco Marker Detection
Combines pre-trained YOLO for person detection with ArUco marker tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from scipy.spatial.distance import cdist
import os

class PersonTrackerWithMarkers:
    def __init__(self, model_name='yolov8n.pt', marker_dict=cv2.aruco.DICT_6X6_250):
        """
        Initialize the person tracker with YOLO and ArUco marker detection
        
        Args:
            model_name: YOLOv8 model to use (nano, small, medium, large)
            marker_dict: ArUco marker dictionary to use
        """
        # Load pre-trained YOLO model for person detection
        self.yolo = YOLO(model_name)
        
        # Setup ArUco marker detection
        self.marker_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
        self.marker_detector = cv2.aruco.ArucoDetector(self.marker_dict)
        
        # Tracking state
        self.track_id_counter = 0
        self.tracked_objects = {}  # {track_id: {'pos': (x,y), 'bbox': (), 'marker_id': None, 'frames': int}}
        self.max_distance = 50  # MaxIoU distance for matching
        self.max_frames_to_skip = 30  # Frames to wait before removing lost track
        
    def detect_persons(self, frame):
        """
        Detect persons using YOLO
        Returns: list of bounding boxes [(x1, y1, x2, y2), ...]
        """
        results = self.yolo(frame, verbose=False, conf=0.5)
        persons = []
        
        for result in results:
            for box in result.boxes:
                # Only track person class (class 0 in COCO dataset)
                if int(box.cls[0]) == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    persons.append((int(x1), int(y1), int(x2), int(y2)))
        
        return persons
    
    def detect_markers(self, frame):
        """
        Detect ArUco markers in the frame
        Returns: dict of {marker_id: [(center_x, center_y), corners]}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, rejected = self.marker_detector.detectMarkers(gray)
        
        markers = {}
        if marker_ids is not None:
            for i, marker_id in enumerate(marker_ids.flatten()):
                corners = marker_corners[i][0]
                center_x = int(np.mean(corners[:, 0]))
                center_y = int(np.mean(corners[:, 1]))
                markers[int(marker_id)] = {'center': (center_x, center_y), 'corners': corners}
        
        return markers
    
    def get_bbox_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def match_detections(self, persons, markers):
        """
        Match detected persons with markers
        Returns: list of (person_bbox, marker_id or None)
        """
        matched = []
        marker_centers = {mid: m['center'] for mid, m in markers.items()}
        
        for person_bbox in persons:
            person_center = self.get_bbox_center(person_bbox)
            
            # Find closest marker to this person
            closest_marker_id = None
            closest_distance = float('inf')
            
            for marker_id, marker_center in marker_centers.items():
                distance = np.sqrt((person_center[0] - marker_center[0])**2 + 
                                 (person_center[1] - marker_center[1])**2)
                if distance < closest_distance and distance < 100:  # 100 pixel threshold
                    closest_distance = distance
                    closest_marker_id = marker_id
            
            matched.append({
                'bbox': person_bbox,
                'marker_id': closest_marker_id,
                'center': person_center
            })
        
        return matched
    
    def update_tracks(self, matched_detections):
        """
        Update tracking information with new detections
        Returns: updated tracked_objects
        """
        # Calculate distances between tracked objects and new detections
        if self.tracked_objects and matched_detections:
            tracked_centers = [obj['pos'] for obj in self.tracked_objects.values()]
            detection_centers = [det['center'] for det in matched_detections]
            
            # Calculate distance matrix
            distances = cdist(tracked_centers, detection_centers, metric='euclidean')
            
            # Hungarian algorithm-like matching (greedy for simplicity)
            used_detections = set()
            for track_id, tracked_obj in list(self.tracked_objects.items()):
                min_distance = float('inf')
                best_det_idx = -1
                
                for det_idx, detection in enumerate(matched_detections):
                    if det_idx in used_detections:
                        continue
                    
                    distance = distances[list(self.tracked_objects.keys()).index(track_id), det_idx]
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_det_idx = det_idx
                
                # Update or remove track
                if best_det_idx >= 0:
                    detection = matched_detections[best_det_idx]
                    self.tracked_objects[track_id]['pos'] = detection['center']
                    self.tracked_objects[track_id]['bbox'] = detection['bbox']
                    self.tracked_objects[track_id]['marker_id'] = detection['marker_id']
                    self.tracked_objects[track_id]['frames'] = 0
                    used_detections.add(best_det_idx)
                else:
                    self.tracked_objects[track_id]['frames'] += 1
        
        # Add new detections as new tracks
        for det_idx, detection in enumerate(matched_detections):
            if det_idx not in used_detections if 'used_detections' in locals() else True:
                self.track_id_counter += 1
                self.tracked_objects[self.track_id_counter] = {
                    'pos': detection['center'],
                    'bbox': detection['bbox'],
                    'marker_id': detection['marker_id'],
                    'frames': 0
                }
        
        # Remove lost tracks
        lost_tracks = [tid for tid, obj in self.tracked_objects.items() 
                      if obj['frames'] > self.max_frames_to_skip]
        for track_id in lost_tracks:
            del self.tracked_objects[track_id]
    
    def draw_trackers(self, frame, markers):
        """
        Draw bounding boxes, IDs, and marker info on frame
        """
        output = frame.copy()
        
        # Draw tracked persons
        for track_id, obj in self.tracked_objects.items():
            x1, y1, x2, y2 = obj['bbox']
            color = (0, 255, 0) if obj['marker_id'] is None else (0, 255, 255)
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID:{track_id}"
            if obj['marker_id'] is not None:
                label += f" M:{obj['marker_id']}"
            
            cv2.putText(output, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cv2.circle(output, obj['pos'], 5, color, -1)
        
        # Draw markers
        if markers:
            for marker_id, marker_info in markers.items():
                corners = marker_info['corners'].astype(int)
                cv2.polylines(output, [corners], True, (255, 0, 0), 2)
                center = marker_info['center']
                cv2.circle(output, center, 5, (255, 0, 0), -1)
                cv2.putText(output, f"Marker:{marker_id}", 
                           (center[0]-20, center[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return output
    
    def process_video(self, video_source=0, output_path=None):
        """
        Process video stream (camera or file) and track persons with markers
        
        Args:
            video_source: 0 for webcam, or path to video file
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {width}x{height} @ {fps}fps")
        print("Press 'q' to stop")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect persons and markers
            persons = self.detect_persons(frame)
            markers = self.detect_markers(frame)
            
            # Match and update tracks
            matched = self.match_detections(persons, markers)
            self.update_tracks(matched)
            
            # Draw results
            output_frame = self.draw_trackers(frame, markers)
            
            # Add info text
            cv2.putText(output_frame, f"Frame: {frame_count} | Tracks: {len(self.tracked_objects)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Person Tracker with Markers', output_frame)
            
            # Write to output video
            if out:
                out.write(output_frame)
            
            # Press 'q' to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete. Processed {frame_count} frames")


def main():
    """
    Main function to run the person tracker
    """
    # Initialize tracker
    tracker = PersonTrackerWithMarkers(model_name='yolov8n.pt')
    
    # For webcam: tracker.process_video(video_source=0)
    # For video file: tracker.process_video(video_source='path/to/video.mp4')
    # To save output: tracker.process_video(video_source=0, output_path='output.mp4')
    
    # Process webcam by default
    tracker.process_video(video_source=0)


if __name__ == '__main__':
    main()
