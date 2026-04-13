#followbutdon'tcrash
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist
import RPi.GPIO as GPIO
import time

# Motor control pins
IN1 = 17    # Motor A direction
IN2 = 27    # Motor A direction
ENA = 18    # Motor A speed (PWM)
IN3 = 22    # Motor B direction
IN4 = 23    # Motor B direction
ENB = 19    # Motor B speed (PWM)

class PersonFollowerRobot:
    def __init__(self, model_name='yolo11n_ncnn_model', marker_dict=cv2.aruco.DICT_6X6_250):
        # YOLO setup
        self.yolo = YOLO(model_name, task='detect')
        self.marker_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
        self.marker_detector = cv2.aruco.ArucoDetector(self.marker_dict)

        # Tracking
        self.track_id_counter = 0
        self.tracked_objects = {}
        self.max_distance = 50
        self.max_frames_to_skip = 30

        # Distance target
        self.target_distance = 5000

        # Frame skip counter
        self.frame_count = 0
        self.last_persons = []

        # NEW: motion-state memory for target
        self.prev_target_id = None
        self.prev_target_center = None
        self.prev_target_area = None
        self.stationary_frames = 0

        # Tune these
        self.motion_threshold_px = 12       # target center movement in pixels
        self.motion_threshold_area = 900    # bbox area change
        self.stationary_frame_limit = 8     # stop after this many still frames

        # GPIO setup
        self.setup_motors()

    def setup_motors(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)
        GPIO.setup([ENA, ENB], GPIO.OUT)
        self.pwmA = GPIO.PWM(ENA, 1000)
        self.pwmB = GPIO.PWM(ENB, 1000)
        self.pwmA.start(0)
        self.pwmB.start(0)

    def move_robot(self, forward_speed, turn_speed):
        left_speed = np.clip(forward_speed - turn_speed, -100, 100)
        right_speed = np.clip(forward_speed + turn_speed, -100, 100)

        left_pwm = abs(int(left_speed))
        right_pwm = abs(int(right_speed))

        # Motor minimum threshold
        if left_pwm > 0 and left_pwm < 20:
            left_pwm = 20
        if right_pwm > 0 and right_pwm < 20:
            right_pwm = 20

        # Motor A (left)
        self.pwmA.ChangeDutyCycle(left_pwm)
        if left_speed >= 0:
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
        else:
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH)

        # Motor B (right)
        self.pwmB.ChangeDutyCycle(right_pwm)
        if right_speed >= 0:
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)
        else:
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.HIGH)

        print(f"Forward: {forward_speed:3d}, Turn: {turn_speed:3d} | L:{left_pwm:3d}, R:{right_pwm:3d}", end='\r')

    def stop_robot(self):
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)

    def reset_motion_state(self):
        self.prev_target_id = None
        self.prev_target_center = None
        self.prev_target_area = None
        self.stationary_frames = 0

    def is_target_moving(self, target_id, target_center, target_area):
        # New target or first frame
        if self.prev_target_id != target_id or self.prev_target_center is None:
            self.prev_target_id = target_id
            self.prev_target_center = target_center
            self.prev_target_area = target_area
            self.stationary_frames = 0
            return True

        dx = target_center[0] - self.prev_target_center[0]
        dy = target_center[1] - self.prev_target_center[1]
        center_shift = np.sqrt(dx * dx + dy * dy)
        area_shift = abs(target_area - self.prev_target_area)

        moved = (
            center_shift > self.motion_threshold_px or
            area_shift > self.motion_threshold_area
        )

        if moved:
            self.stationary_frames = 0
        else:
            self.stationary_frames += 1

        self.prev_target_id = target_id
        self.prev_target_center = target_center
        self.prev_target_area = target_area

        return self.stationary_frames < self.stationary_frame_limit

    def detect_persons(self, frame):
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            results = self.yolo(frame, verbose=False, conf=0.5, imgsz=320, classes=[0])
            persons = []
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        persons.append((int(x1), int(y1), int(x2), int(y2)))
            self.last_persons = persons

        return self.last_persons

    def detect_markers(self, frame):
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
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def calculate_robot_movement(self, frame, person_center, person_area):
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2

        error_x = person_center[0] - frame_center_x

        # Smaller dead zone so robot keeps target nearer the center
        dead_zone_x = frame_width * 0.05   # was 0.08

        # Stronger turn control
        if abs(error_x) < dead_zone_x:
            turn = 0
        else:
            turn = int(np.clip(error_x / (frame_width / 2) * 65, -65, 65))

        # Distance control
        distance_error = self.target_distance - person_area

        if abs(distance_error) < 2000:
            forward = 0
        else:
            forward = int(np.clip(distance_error / self.target_distance * 40, -40, 40))

        # Optional: if target is far from center, prioritize turning over forward motion
        if abs(error_x) > frame_width * 0.20:
            forward = 0

        return forward, turn

    def match_detections(self, persons, markers):
        matched = []
        marker_centers = {mid: m['center'] for mid, m in markers.items()}

        for person_bbox in persons:
            person_center = self.get_bbox_center(person_bbox)
            closest_marker_id = None
            closest_distance = float('inf')

            for marker_id, marker_center in marker_centers.items():
                distance = np.sqrt((person_center[0] - marker_center[0])**2 +
                                   (person_center[1] - marker_center[1])**2)
                if distance < closest_distance and distance < 100:
                    closest_distance = distance
                    closest_marker_id = marker_id

            matched.append({
                'bbox': person_bbox,
                'marker_id': closest_marker_id,
                'center': person_center
            })

        return matched

    def update_tracks(self, matched_detections):
        used_detections = set()

        if self.tracked_objects and matched_detections:
            tracked_centers = [obj['pos'] for obj in self.tracked_objects.values()]
            detection_centers = [det['center'] for det in matched_detections]

            distances = cdist(tracked_centers, detection_centers, metric='euclidean')

            for i, (track_id, tracked_obj) in enumerate(list(self.tracked_objects.items())):
                min_distance = float('inf')
                best_det_idx = -1

                for det_idx, detection in enumerate(matched_detections):
                    if det_idx in used_detections:
                        continue
                    distance = distances[i, det_idx]
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_det_idx = det_idx

                if best_det_idx >= 0:
                    detection = matched_detections[best_det_idx]
                    self.tracked_objects[track_id]['pos'] = detection['center']
                    self.tracked_objects[track_id]['bbox'] = detection['bbox']
                    self.tracked_objects[track_id]['marker_id'] = detection['marker_id']
                    self.tracked_objects[track_id]['frames'] = 0
                    used_detections.add(best_det_idx)
                else:
                    self.tracked_objects[track_id]['frames'] += 1

        for det_idx, detection in enumerate(matched_detections):
            if det_idx not in used_detections:
                self.track_id_counter += 1
                self.tracked_objects[self.track_id_counter] = {
                    'pos': detection['center'],
                    'bbox': detection['bbox'],
                    'marker_id': detection['marker_id'],
                    'frames': 0
                }

        lost_tracks = [tid for tid, obj in self.tracked_objects.items()
                       if obj['frames'] > self.max_frames_to_skip]
        for track_id in lost_tracks:
            del self.tracked_objects[track_id]

    def draw_trackers(self, frame, markers, target_id=None):
        output = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        center_x, center_y = frame_width // 2, frame_height // 2
        cv2.line(output, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 255), 2)
        cv2.line(output, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 255), 2)

        for track_id, obj in self.tracked_objects.items():
            x1, y1, x2, y2 = obj['bbox']
            color = (0, 0, 255) if track_id == target_id else (0, 255, 0)
            thickness = 3 if track_id == target_id else 2

            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            label = f"ID:{track_id}"
            if obj['marker_id'] is not None:
                label += f" M:{obj['marker_id']}"

            cv2.putText(output, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(output, obj['pos'], 5, color, -1)

        if markers:
            for marker_id, marker_info in markers.items():
                corners = marker_info['corners'].astype(int)
                cv2.polylines(output, [corners], True, (255, 0, 0), 2)
                cv2.circle(output, marker_info['center'], 5, (255, 0, 0), -1)

        return output

    def get_target_person(self):
        if not self.tracked_objects:
            return None

        # First priority: person with marker
        marked_persons = {}
        for track_id, obj in self.tracked_objects.items():
            if obj['marker_id'] is not None:
                area = self.get_bbox_area(obj['bbox'])
                marked_persons[track_id] = area

        if marked_persons:
            return max(marked_persons, key=marked_persons.get)

        # Fallback: largest visible person
        all_persons = {}
        for track_id, obj in self.tracked_objects.items():
            area = self.get_bbox_area(obj['bbox'])
            all_persons[track_id] = area

        if all_persons:
            return max(all_persons, key=all_persons.get)

        return None

    def process_video(self, video_source=0):
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            self.cleanup()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("Person Following Robot - STARTING")
        print("Press 'q' to stop\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                persons = self.detect_persons(frame)
                markers = self.detect_markers(frame)
                matched = self.match_detections(persons, markers)
                self.update_tracks(matched)

                target_id = self.get_target_person()
                output_frame = self.draw_trackers(frame, markers, target_id)

                if target_id and target_id in self.tracked_objects:
                    target_obj = self.tracked_objects[target_id]
                    target_bbox = target_obj['bbox']
                    target_center = target_obj['pos']
                    person_area = self.get_bbox_area(target_bbox)

                    target_moving = self.is_target_moving(target_id, target_center, person_area)

                    if target_moving:
                        forward, turn = self.calculate_robot_movement(frame, target_center, person_area)
                        self.move_robot(-forward, turn)

                        cv2.putText(output_frame, f"Area: {person_area} | Fwd: {forward} | Turn: {turn}",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        has_marker = " [MARKER]" if target_obj['marker_id'] is not None else ""
                        cv2.putText(output_frame, f"FOLLOWING ID {target_id}{has_marker}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        self.stop_robot()
                        cv2.putText(output_frame, f"TARGET STILL - STOPPED ({self.stationary_frames})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(output_frame, f"Area: {person_area}",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                else:
                    self.reset_motion_state()
                    self.stop_robot()
                    cv2.putText(output_frame, "Searching for person...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Robot Tracker', output_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopping robot...")

        finally:
            self.cleanup()
            cap.release()
            cv2.destroyAllWindows()

    def cleanup(self):
        self.stop_robot()
        self.pwmA.stop()
        self.pwmB.stop()
        GPIO.cleanup()


def main():
    tracker = PersonFollowerRobot(model_name='yolo11n_ncnn_model')
    tracker.process_video(video_source=0)


if __name__ == '__main__':
    main()

