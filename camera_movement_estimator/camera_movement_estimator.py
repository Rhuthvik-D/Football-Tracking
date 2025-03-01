import pickle
import cv2
import numpy as np
import sys
import os
sys.path.append('../')
from utils import measure_dist, measure_xy_distance


class CameraMovementEstimator():
    def __init__(self, frame):
        self.min_distance = 5

        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2, #pyramid level to downscale the image upto tweice for capturing features
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) #criteria to stop the algorithm after 10 iterations or 0.03 accuracy
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)   #chooses top and bottom of the frame
        mask_features[:,0:20] = 1 #first 20 rows of pixels in the frame
        mask_features[:,900:1050] = 1 #last 20 rows of pixels in the frame

        self.features = dict(
            maxCorners = 100, #no. of corners utilized for good features
            qualityLevel = 0.3, #quality level of the corners, higher quality means better features but less number of features
            minDistance = 3, #minimum distance between two corners 
            blockSize = 7, #block size
            mask = mask_features, #mask to apply on the frame
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        #read stub file
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        #calculate camera movement fro each frame
        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)#call all prev frames as old
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features) ##"**" is used to unpack the dictionary

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params) #to see where the new features are in the new frame

            #measure the distance between the old and new frames to know if the camera moved
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel() #converts the array into a single row
                old_features_point = old.ravel() 

                distance = measure_dist(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.min_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y] #if the camera moved, then update the camera movement
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features) #update the features        

            old_gray = frame_gray.copy()
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num] #get the camera movement for the frame
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1]) #adjust the position of the object based on the camera movement
                    tracks[obj][frame_num][track_id]['position_adjusted'] = position_adjusted

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1) #draw a rectangle on the frame
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  #overlay the rectangle on the frame

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

            output_frames.append(frame)
        
        return output_frames

