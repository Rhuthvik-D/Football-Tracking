from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if obj == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[obj][frame_num][track_id]['position'] = position #add position to the track
                    
    #interpolate missing values in the frames
    def interpolate_ball_pos(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions] #get track track_id and respective bounding box from the ball positions. if the track_id is null, then bbox is null
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1) #conf = 0.1 enough to detect person and ball
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        #if already exists the output file, read from it
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],  # has a collection of frames. In each frame, it has a dictionary of track_id and bbox
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v:k for k,v in class_names.items()}

            #change to supervision detection format

            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert Goal keeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = class_names_inv["player"]

            #Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == class_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == class_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}     
        

        #save the output to a file
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)


        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center = (x_center, y2),
            axes = (int(width), int(width*0.35)),#major axis and minor axis of an ellipse
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4                              
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2   #move half of width from center of rectangle to left
        x2_rect = x_center + rectangle_width//2   #move half of width from center of rectangle to right
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)    #fills rect with color
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10    #text padding for bigger numbers(more than 2 digits)

            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2)


        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) #triangle on ball with filled color
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) #triangle on ball with black border

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        #draw semi transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), -1)
        alpha = 0.4 #transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame) #overlay the rectangle on the frame

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        #get number of times each team has ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]         
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control:{team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control:{team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            #draw circle beneath user
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            #draw players
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255), _)

            #draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            #draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            
            output_video_frames.append(frame)
        return output_video_frames


