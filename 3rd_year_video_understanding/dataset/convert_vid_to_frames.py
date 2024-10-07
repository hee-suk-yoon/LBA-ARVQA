import cv2
import os
import json
import argparse

from tqdm import tqdm
from multiprocessing import Pool
from moviepy.editor import VideoFileClip


video_folder = "/data2/esyoon_hdd/DiDeMo/video/YFCC100M_videos"
output_folder = "/data2/esyoon_hdd/DiDeMo/frames"


video_list = os.listdir(video_folder)

def get_args():
    parser = argparse.ArgumentParser(description="Convert video to frames")
    parser.add_argument("--video_folder", type=str, default="./sample/videos", help="Path to the folder containing videos")
    parser.add_argument("--output_folder", type=str, default="/data2/esyoon_hdd/DiDeMo/frames", help="Path to the folder to save frames")
    return parser.parse_args()

# =============================================================================================================
# for multiprocessing
def process_video(video_name):
    video_path = os.path.join(video_folder, video_name)
    try:
        # vidcap = cv2.VideoCapture(video_path)
        clip = VideoFileClip(video_path)
         # Get the total number of frames 
        total_frames = int(clip.fps * clip.duration)

        num_output_frames = 50

        # Calculate the interval between frames to be saved
        frame_interval = max(1, total_frames // num_output_frames)

        count = 0
        saved_count = 0

        output_folder_video = os.path.join(output_folder, video_name.split('.')[0])
        # output_folder_video = os.path.join(output_folder, video_name)

        if not os.path.exists(output_folder_video):
            os.makedirs(output_folder_video)

        for i, frame in enumerate(clip.iter_frames()):
            if i % frame_interval == 0 and i // frame_interval < num_output_frames:
                frame_filename = os.path.join(output_folder_video, f"{i // frame_interval:06d}.jpg")
                frame_image = frame[:, :, ::-1]  # Convert RGB to BGR for OpenCV compatibility
                cv2.imwrite(frame_filename, frame_image)
        clip.close()

    except:
        print(f'Error processing video: {video_name}')
        errored_videos.append(video_name)
        error_messages.append(f'Error processing video: {video_name}')


    print(f'Finished processing video: {video_name}')

# =============================================================================================================
if __name__ == "__main__":    
    # video_list = os.listdir(video_folder)

    # Use a pool of workers to process multiple videos in parallel
    with Pool(processes=os.cpu_count()) as pool:  # Use all available CPU cores
    # with Pool(processes=1) as pool:  # Use all available CPU cores
        pool.map(process_video, video_list)

    print("Finished processing all videos.")
# =============================================================================================================