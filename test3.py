import cv2
import torch
import numpy as np
import os
from pathlib import Path
from strongsort import StrongSORT
import argparse

def load_yolo_model():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    return model

def initialize_tracker(max_dist, device):
    # Initialize StrongSort tracker with a stronger Re-ID model
    tracker = StrongSORT(
        model_path="osnet_x0_5",  # Re-ID model
        device=device,  # Use device from argument (CPU/GPU)
        max_dist=max_dist,
        max_iou_distance=0.7
    )
    return tracker

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

    # Adjust max_dist
    max_dist = 0.2  # Default
    if num_objects > 15:
        max_dist = 0.3
    elif num_objects < 5:
        max_dist = 0.1

    return confidence, max_dist

def process_video(video_path, output_path, model, tracker, default_confidence):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = 1280
    new_height = int(original_height * (new_width / original_width))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (new_width, new_height))

        results = model(frame)
        detections = results.pandas().xyxy[0]

        # Adjust parameters based on frame and detections
        confidence, max_dist = adjust_parameters(frame, detections)
        tracker.max_dist = max_dist

        detections = detections[detections['confidence'] >= confidence]
        vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']
        pedestrian_classes = ['person']
        detections = detections[detections['name'].isin(vehicle_classes + pedestrian_classes)]

        # Prepare detections for tracker
        bbox_xywh = []
        confidences = []
        classes = []

        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
            conf = detection['confidence']
            label = detection['name']

            bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            confidences.append(conf)
            classes.append(vehicle_classes.index(label) if label in vehicle_classes else len(vehicle_classes))

        bbox_xywh = np.array(bbox_xywh)
        confidences = np.array(confidences)

        # Update tracker
        outputs = tracker.update(bbox_xywh, confidences, classes)

        for output in outputs:
            x1, y1, x2, y2, track_id, cls = output
            label = vehicle_classes[cls] if cls < len(vehicle_classes) else "person"
            color = (0, 255, 0) if label in vehicle_classes else (0, 0, 255)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{label} ID: {track_id}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write frame
        out.write(frame)

    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")

def process_multiple_videos(video_paths, output_dir, confidence, max_dist):
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load YOLOv5 model
    model = load_yolo_model()

    # Initialize DeepSORT tracker
    tracker = initialize_tracker(max_dist)

    # Process each video
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist.")
            continue

        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, f"result_{video_name}")
        print(f"Processing {video_path}...")
        process_video(video_path, output_path, model, tracker, confidence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Object Detection and Tracking")
    parser.add_argument('--confidence', type=float, default=0.3, help='Default confidence threshold for YOLOv5')
    parser.add_argument('--max_dist', type=float, default=0.2, help='Default max distance for StrongSORT tracker')
    args = parser.parse_args()

    # List of video files to process
    video_paths = ["test3.mp4"]

    # Directory to save processed videos
    output_dir = "result_videos"

    # Process all videos
    process_multiple_videos(video_paths, output_dir, args.confidence, args.max_dist)



#Cần hêm option để lựa chọn device (CPU/GPU)