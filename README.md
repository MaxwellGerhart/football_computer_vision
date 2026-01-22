# Football Analysis Project

## Introduction
This project uses computer vision to detect and track players, referees, and footballs in broadcast football/soccer videos using YOLO object detection. It assigns players to teams based on jersey colors using K-means clustering, tracks ball possession, estimates camera movement using optical flow, and applies perspective transformation to measure real-world player movements in meters. The project also includes a minimap overlay showing player positions from a top-down view.

## Features
- **Player & Ball Detection**: Uses YOLO to detect players, goalkeepers, referees, and the ball
- **Team Assignment**: Automatically assigns players to teams based on jersey color using K-means clustering
- **Ball Possession Tracking**: Tracks which team has possession of the ball
- **Camera Movement Estimation**: Uses optical flow to compensate for camera movement
- **Perspective Transformation**: Converts pixel positions to real-world coordinates (meters)
- **Speed & Distance Calculation**: Calculates player speed and distance covered
- **Minimap Overlay**: Displays a top-down view of player positions on a pitch diagram

## Modules Used
- **YOLO (Ultralytics)**: AI object detection model for detecting players, ball, and referees
- **K-means Clustering**: Pixel segmentation to detect jersey colors and assign teams
- **Optical Flow**: Measure camera movement between frames
- **Perspective Transformation**: Convert screen coordinates to real-world field positions
- **OpenCV**: Video processing, drawing, and computer vision utilities

## Project Structure
```
├── main.py                      # Main entry point
├── trackers/                    # Object tracking with YOLO + ByteTrack
├── team_assigner/               # Team assignment using jersey colors
├── player_ball_assigner/        # Ball possession detection
├── camera_movement_estimator/   # Optical flow for camera movement
├── view_transformer/            # Perspective transformation
├── speed_and_distance_estimator/# Player speed and distance calculations
├── minimap/                     # Minimap overlay visualization
├── field_calibrator/            # Field line detection for calibration
├── utils/                       # Video I/O and helper functions
├── models/                      # YOLO model weights (best.pt)
├── input_videos/                # Input video files
├── output_videos/               # Processed output videos
└── stubs/                       # Cached tracking data
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/football_computer_vision.git
cd football_computer_vision
```

2. Create and activate a virtual environment:
```bash
python -m venv football_cv
# Windows
football_cv\Scripts\activate
# Linux/Mac
source football_cv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the trained YOLO model and place it in the `models/` directory:
   - [Trained YOLO v5 Model](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Usage

### Basic Usage
```bash
python main.py <video_path>
```

### Command Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `video_path` | Path to the input video file | Required |
| `--output` | Path for the output video file | `output_videos/output_video.mp4` |
| `--no-speed` | Disable speed and distance calculations | False |

### Examples
```bash
# Basic analysis
python main.py input_videos/match.mp4

# Custom output path
python main.py input_videos/match.mp4 --output my_analysis.mp4

# Disable speed/distance overlay
python main.py input_videos/match.mp4 --no-speed
```

## Output
The processed video includes:
- Bounding boxes around detected players, referees, and the ball
- Player ID numbers
- Team colors (ellipse under each player)
- Ball possession indicator
- Camera movement display
- Team ball control percentages
- Minimap with player positions
- Speed and distance statistics (optional)

## Requirements
- Python 3.x
- ultralytics
- supervision
- opencv-python
- numpy
- matplotlib
- pandas
- scikit-learn

## Sample Video
- [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Known Limitations
- The perspective transformation is calibrated for specific camera angles; different broadcast views may require recalibration
- Goalkeeper team assignment can be inaccurate due to different jersey colors
- Ball detection can be inconsistent when the ball is partially occluded

## License
This project is for educational purposes.