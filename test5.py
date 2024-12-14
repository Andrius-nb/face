import cv2
import torch
import numpy as np
import os
from pathlib import Path
import argparse
from bytetrack import BYTETracker
from bytetrack.utils import tlwh2xyxy, xyxy2tlwh

def load_yolo_model():
    # Load YOLOv5 model
    model = torch.hub.load('./yolov5', 'yolov5l', source='local', pretrained=True)
    return model

def adjust_parameters(frame, detections):
    # Evaluate frame brightness and object density
    num_objects = len(detections)
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Adjust confidence
    confidence = 0.3  # Default
    if brightness < 50:  # Dark frame
        confidence = 0.2
    elif num_objects > 20:  # Crowded frame
        confidence = 0.4

    return confidence

def process_video(video_path, output_path, model, default_confidence):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Initialize counts for vehicles and pedestrians
    previous_vehicle_count = 0
    previous_pedestrian_count = 0
    total_vehicle_count = 0
    total_pedestrian_count = 0

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = 1280
    new_height = int(original_height * (new_width / original_width))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

    vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']
    pedestrian_classes = ['person']

    # Initialize ByteTracker
    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (new_width, new_height))

        results = model(frame)
        detections = results.pandas().xyxy[0]

        # Adjust parameters based on frame and detections
        confidence = adjust_parameters(frame, detections)

        detections = detections[detections['confidence'] >= confidence]
        detections = detections[detections['name'].isin(vehicle_classes + pedestrian_classes)]

        # Convert detections to format required by ByteTracker
        det_boxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values
        det_scores = detections['confidence'].values
        det_classes = detections['class'].values

        # Run ByteTracker
        online_targets = tracker.update(det_boxes, det_scores, det_classes)

        # Count vehicles and pedestrians
        current_vehicle_count = 0
        current_pedestrian_count = 0

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            class_id = t.class_id
            
            if tlwh[2] * tlwh[3] > 0:  # width * height > 0
                x1, y1, x2, y2 = tlwh2xyxy(tlwh)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Determine class and color
                if class_id in vehicle_classes:
                    current_vehicle_count += 1
                    color = (0, 255, 0)
                    label = f"{vehicle_classes[class_id]} {tid}"
                elif class_id in pedestrian_classes:
                    current_pedestrian_count += 1
                    color = (0, 0, 255)
                    label = f"Person {tid}"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update total counts
        if current_vehicle_count > previous_vehicle_count:
            total_vehicle_count += (current_vehicle_count - previous_vehicle_count)
        if current_pedestrian_count > previous_pedestrian_count:
            total_pedestrian_count += (current_pedestrian_count - previous_pedestrian_count)

        # Display counts for vehicles and pedestrians
        info_text = [
            f"Current Vehicles: {current_vehicle_count}",
            f"Total Vehicles: {total_vehicle_count}",
            f"Current People: {current_pedestrian_count}",
            f"Total People: {total_pedestrian_count}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write frame
        out.write(frame)

        # Update previous counts
        previous_vehicle_count = current_vehicle_count
        previous_pedestrian_count = current_pedestrian_count

    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")

def process_multiple_videos(video_paths, output_dir, confidence):
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
        process_video(video_path, output_path, model, confidence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Object Detection and Tracking")
    parser.add_argument('--confidence', type=float, default=0.3, help='Default confidence threshold for YOLOv5')
    args = parser.parse_args()

    # List of video files to process
    video_paths = ["video/test33.mp4"]

    # Directory to save processed videos
    output_dir = "result_videos"

    # Process all videos
    process_multiple_videos(video_paths, output_dir, args.confidence)