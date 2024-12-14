import cv2
import torch
import numpy as np
from pathlib import Path

def load_yolo_model():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def process_video(video_path):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Initialize counts
    previous_vehicle_count = 0
    previous_pedestrian_count = 0
    total_vehicle_count = 0
    total_pedestrian_count = 0

    # Load YOLOv5 model
    model = load_yolo_model()
    
    # Define the classes to detect
    vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']  # All vehicles
    pedestrian_classes = ['person']

    # Get video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = 1280
    new_height = int(original_height * (new_width / original_width))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (new_width, new_height))

        # Perform detection
        results = model(frame)
        detections = results.pandas().xyxy[0]
        
        # Filter to keep only vehicles and pedestrians
        detections = detections[detections['confidence'] >= 0.5]  # Bỏ các đối tượng có confidence thấp
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

        cv2.imshow('Detection', frame)

        # Update previous counts
        previous_vehicle_count = current_vehicle_count
        previous_pedestrian_count = current_pedestrian_count

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "test33.mp4"
    process_video(video_path)
