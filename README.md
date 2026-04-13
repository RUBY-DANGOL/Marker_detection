# Person Tracker with YOLO and ArUco Markers

A real-time person tracking system that combines YOLOv8 for person detection and OpenCV ArUco markers for enhanced tracking accuracy.

## Features

✅ **Pre-trained Models** - Uses YOLOv8 (already trained on person detection)
✅ **Marker Detection** - ArUco marker detection for visual identification
✅ **Real-time Tracking** - Tracks multiple persons across frames
✅ **Webcam & Video Support** - Works with webcam or video files
✅ **Video Output** - Save tracked video to file

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Generate markers (optional but recommended):**
```bash
python generate_markers.py
```
This creates printable ArUco markers in the `markers/` folder.

## Usage

### Basic Webcam Tracking

```bash
python person_tracker.py
```

This starts tracking persons from your webcam in real-time.

### From Video File

Edit `person_tracker.py` and change the last line in `main()`:
```python
tracker.process_video(video_source='path/to/your/video.mp4')
```

### Save Output Video

```python
tracker.process_video(video_source=0, output_path='tracked_output.mp4')
```

## How It Works

1. **Person Detection (YOLO)**
   - Detects all persons in each frame
   - Uses pre-trained YOLOv8n model
   - Returns bounding boxes for each person

2. **Marker Detection (ArUco)**
   - Detects ArUco markers in the frame
   - Associates markers with nearby persons
   - Provides visual identification

3. **Tracking**
   - Matches detections across frames
   - Assigns unique track IDs to each person
   - Maintains track history using centroid matching
   - Gracefully handles occlusion (up to 30 frames)

## Visual Output

**Green boxes** - Person detected (no marker)
**Cyan boxes** - Person with associated marker

Each box shows:
- Track ID: Unique identifier for the person
- Marker ID: If a marker is associated with this person
- Center point: Small circle showing centroid

## Customization

### Using Different YOLO Models

```python
# Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
tracker = PersonTrackerWithMarkers(model_name='yolov8s.pt')
```

- `yolov8n` - Nano (fastest, lowest accuracy)
- `yolov8s` - Small
- `yolov8m` - Medium
- `yolov8l` - Large
- `yolov8x` - Extra Large (slowest, highest accuracy)

### Adjust Tracking Parameters

In `person_tracker.py`:
```python
self.max_distance = 50  # Distance threshold for matching
self.max_frames_to_skip = 30  # Frames to wait before removing lost track
```

### Marker Detection Threshold

In `detect_persons()`:
```python
results = self.yolo(frame, verbose=False, conf=0.5)  # conf = confidence threshold (0-1)
```

## Requirements

- Python 3.8+
- OpenCV 4.8+
- YOLOv8 (Ultralytics)
- NumPy, SciPy

## First Run

The first time you run the tracker, YOLOv8 will download the pre-trained weights (~40MB for nano model). This is normal and only happens once.

## Troubleshooting

**Q: Low FPS performance**
- A: Use smaller YOLO model (yolov8n)
- A: Lower video resolution
- A: Increase `max_distance` for faster matching

**Q: Markers not detected**
- A: Ensure good lighting
- A: Marker should be roughly planar and in-frame
- A: Adjust camera zoom/position

**Q: Too many ID switches**
- A: Decrease `max_distance` for stricter matching
- A: Use markers for better person identification
- A: Use larger YOLO model for better detection

## Project Structure

```
targetmatch/
├── person_tracker.py       # Main tracker class
├── generate_markers.py     # Marker generator utility
├── requirements.txt        # Dependencies
├── README.md              # This file
└── markers/               # Generated ArUco markers (created by generate_markers.py)
```

## For raspberry pi 4
Image detection model can be heavy for raspberry pi so using ncnn model of yolo11 will make the performance of the model better.
In raspberry pi- 
- YOLOv8n -> ~2-3 FPS
- YOLO11n -> ~2.8 FPS
- YOLO11n (NCNN export)  -> ~8-12 FPS
#### To convert yolo11n.pt into yolo11n_ncnn_model
```bash
pip install ultralytics
yolo export model=yolo11n.pt format=ncnn imgsz=320
```
### If you want the yolo detection and marker detection model and follow people with marker in raspberry pi 
```bash
python generate_markers.py
python  follow.py
```
## License

MIT License

## Contact

For questions or improvements, feel free to modify and extend!
