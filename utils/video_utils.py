import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = [] #empty list of frames
    while True: 
        ret, frame = cap.read() #read frame by frame. ret is a boolean value that returns true if the frame is read correctly
        if not ret: #if ret is false, break the loop
            break
        frames.append(frame)
    return frames


def save_video(output_video_frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #output format
    out = cv2.VideoWriter(output_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) #output video. 24 is the number of frames per second. shape[1] is "x" and shape[0] is "y"
    for frame in output_video_frames:
        out.write(frame)
    out.release()