# Football Tracking System

## Overview
This project is a **Football Tracking System** that leverages **YOLO-based object detection** and **computer vision techniques** to track and analyze player and ball movements on the field. It includes functionalities such as player tracking, ball tracking, speed estimation, team assignment, and camera movement estimation.

## Features
- **Player and Ball Detection:** Uses YOLO for detecting players and the ball in video frames.
- **Camera Movement Estimation:** Computes camera movement across frames to improve tracking accuracy.
- **Player-Ball Assignment:** Assigns the ball to the nearest player based on distance.
- **Speed and Distance Estimation:** Calculates player speed and total distance covered.
- **Team Assignment:** Clusters players into teams using color analysis.
- **View Transformation:** Warps the field view to a standard coordinate system.
- **Tracking & Interpolation:** Keeps track of object movements and interpolates missing data.
- **Video Processing Utilities:** Reads and writes videos while overlaying visual annotations.

## Installation
### Prerequisites
Ensure you have Python installed before proceeding.

### Clone Repository
```sh
git clone https://github.com/Rhuthvik-D/Football-Tracking.git
cd football-tracking
```

### Setting Up Virtual Environment
It is recommended to use a virtual environment for dependency management:

```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

Once the virtual environment is activated, install the required dependencies:

```sh
pip install -r requirements.txt
```

To deactivate the virtual environment:
```sh
deactivate
```

## Usage
### 1. Uploading a Video
Before running the system, ensure that you have placed a video file in the `input_videos/` folder. The video should be in `.mp4` format. Rename the video file or update the filename in `main.py` if necessary.

### 2. Running the Tracker
Once the video is in place, execute the tracking system using:

```sh
python main.py
```

### 3. Modules & Functionality
| Module | Description |
|--------|-------------|
| `main.py` | The main script that orchestrates the tracking pipeline. |
| `tracker.py` | Runs YOLO-based detection and tracking for players and ball. |
| `camera_movement_estimator.py` | Estimates camera movement across frames. |
| `player_ball_assigner.py` | Assigns ball possession to the nearest player. |
| `speed_and_distance_estimator.py` | Computes players' speed and total distance covered. |
| `team_assigner.py` | Assigns players to teams using color-based clustering. |
| `bbox_utils.py` | Utility functions for bounding box calculations. |
| `video_utils.py` | Functions to read and save videos. |
| `view_transformer.py` | Warps field view into a standardized coordinate system. |

## Methodologies
### 1. **Object Detection**
   - Uses the **YOLO model** to detect players, referees, and the ball in each frame.
   - Assigns bounding boxes to detected objects.

### 2. **Object Tracking**
   - Utilizes **ByteTrack** to maintain consistent object IDs across frames.
   - Stores player and ball positions in structured tracking data.

### 3. **Camera Movement Estimation**
   - Uses optical flow techniques to detect frame-by-frame camera movement.
   - Adjusts object positions accordingly to counteract camera shift.

### 4. **Player and Ball Assignment**
   - Calculates the Euclidean distance between players and the ball.
   - Assigns ball possession to the nearest player within a predefined distance threshold.

### 5. **Speed and Distance Calculation**
   - Uses frame-to-frame displacement of transformed positions to estimate speed.
   - Computes total distance covered by each player using accumulated frame distances.

### 6. **Team Assignment**
   - Uses **K-Means clustering** to differentiate teams based on jersey colors.
   - Assigns players to teams dynamically during gameplay.

### 7. **View Transformation**
   - Transforms real-world positions from the camera perspective to a fixed coordinate system.
   - Allows better analysis of player movements relative to the field layout.

### 8. **Video Processing and Annotation**
   - Reads and writes videos with **OpenCV**.
   - Overlays tracking information, such as speed, ball possession, and player/team identification, on the output video.

## Output
The system generates:
- Processed **video output** with tracking overlays saved in `output_videos/`.
- JSON **tracking data** containing movement metrics.
- Team possession stats and speed estimations.

## Contributing
Feel free to fork this repository and submit a pull request with enhancements or bug fixes!

## License
This project is licensed under the MIT License.
