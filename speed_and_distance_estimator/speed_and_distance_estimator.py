import sys
import cv2
sys.path.append('../')
from utils import measure_dist, get_foot_position

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.fps = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_disatnce = {}

        for obj, obj_tracks in tracks.items():
            if obj == 'ball' or obj == 'referees':
                continue
            number_of_frames = len(obj_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in obj_tracks[frame_num].items():
                    if track_id not in obj_tracks[last_frame]:
                        continue

                    start_position = obj_tracks[frame_num][track_id]['position_transformed']
                    end_position = obj_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_dist(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.fps
                    speed_mps = distance_covered / time_elapsed
                    speed_kmph = speed_mps * 3.6

                    if obj not in total_disatnce:
                        total_disatnce[obj] = {}

                    if track_id not in total_disatnce[obj]:
                        total_disatnce[obj][track_id] = 0
                    
                    total_disatnce[obj][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[obj][frame_num_batch]:
                            continue
                        tracks[obj][frame_num_batch][track_id]['speed'] = speed_kmph
                        tracks[obj][frame_num_batch][track_id]['distance'] = total_disatnce[obj][track_id]
    
    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj == 'ball' or obj == 'referees':
                    continue
                for _, track_info in obj_tracks[frame_num].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/hr", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)
        return output_frames



                            


