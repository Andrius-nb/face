import cv2
import torch
import numpy as np
import os
from pathlib import Path

def load_yolo_model():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def process_video(video_path, output_path, model):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Initialize counts
    previous_vehicle_count = 0
    previous_pedestrian_count = 0
    total_vehicle_count = 0
    total_pedestrian_count = 0

    # Define the classes to detect
    vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']  # All vehicles
    pedestrian_classes = ['person']

    # Get video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = 1280
    new_height = int(original_height * (new_width / original_width))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (new_width, new_height))

        # Perform detection
        results = model(frame)
        detections = results.pandas().xyxy[0]
        
        # Filter to keep only vehicles and pedestrians
        detections = detections[detections['confidence'] >= 0.3]  # Bỏ các đối tượng có confidence thấp
        detections = detections[detections['name'].isin(vehicle_classes + pedestrian_classes)]
        
        # Count objects
        current_vehicle_count = len(detections[detections['name'].isin(vehicle_classes)])
        current_pedestrian_count = len(detections[detections['name'].isin(pedestrian_classes)])

        # Update total counts
        if current_vehicle_count > previous_vehicle_count:
            total_vehicle_count += (current_vehicle_count - previous_vehicle_count)
        if current_pedestrian_count > previous_pedestrian_count:
            total_pedestrian_count += (current_pedestrian_count - previous_pedestrian_count)

        # Draw detection boxes
        for idx, detection in detections.iterrows():
            x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
            label = detection['name']
            conf = detection['confidence']
            
            # Green for vehicles, red for people
            color = (0, 255, 0) if label in vehicle_classes else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display counts
        info_text = [
            f"Current Vehicles: {current_vehicle_count}",
            f"Total Vehicles: {total_vehicle_count}",
            f"Current People: {current_pedestrian_count}",
            f"Total People: {total_pedestrian_count}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the frame to the output video
        out.write(frame)

        # Update previous counts
        previous_vehicle_count = current_vehicle_count
        previous_pedestrian_count = current_pedestrian_count

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")

def process_multiple_videos(video_paths, output_dir):
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load YOLOv5 model
    model = load_yolo_model()

    # Process each video
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist.")
            continue
        
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, f"result_{video_name}")
        print(f"Processing {video_path}...")
        process_video(video_path, output_path, model)

if __name__ == "__main__":
    # List of video files to process
    video_paths = [
        "test33333.mp4"
    ]

    # Directory to save processed videos
    output_dir = "result_videos"

    # Process all videos
    process_multiple_videos(video_paths, output_dir)
