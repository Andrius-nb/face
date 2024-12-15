import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

def load_yolo_model():
    model = torch.hub.load('./yolov5', 'yolov5s', source='local', pretrained=True)
    return model

def process_video(video_path, output_path, model, default_confidence):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Không thể mở video {video_path}")
        return

    deep_sort = DeepSort(max_age=30)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = 1280
    new_height = int(original_height * (new_width / original_width))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

    vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']
    pedestrian_classes = ['person']
    
    class_to_index = {name: idx for idx, name in model.names.items()}
    
    vehicle_indices = [class_to_index[c] for c in vehicle_classes if c in class_to_index]
    pedestrian_indices = [class_to_index[c] for c in pedestrian_classes if c in class_to_index]
    interested_indices = vehicle_indices + pedestrian_indices

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Khởi tạo bộ đếm
    current_vehicle_count = 0
    total_vehicle_count = 0
    current_pedestrian_count = 0
    total_pedestrian_count = 0
    tracked_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (new_width, new_height))
        results = model(frame_resized)

        detections = results.xyxy[0].cpu().numpy()
        
        valid_detections = detections[
            (detections[:, 4] >= default_confidence) & 
            np.isin(detections[:, 5], interested_indices)
        ]
        
        # Reset current counts for each frame
        current_vehicle_count = 0
        current_pedestrian_count = 0

        if len(valid_detections) > 0:
            track_boxes = valid_detections[:, :4]
            track_scores = valid_detections[:, 4]
            track_class_ids = valid_detections[:, 5]
            
            detections_for_deepsort = [
                ([x1, y1, x2 - x1, y2 - y1], score, class_id) 
                for (x1, y1, x2, y2), score, class_id in zip(track_boxes, track_scores, track_class_ids)
            ]

            tracks = deep_sort.update_tracks(detections_for_deepsort, frame=frame_resized)

            for track in tracks:
                if track.is_confirmed():
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    class_id = track.get_det_class()
                    label = 'Vehicle' if class_id in vehicle_indices else 'Person'
                    color = (0, 255, 0) if label == 'Vehicle' else (0, 0, 255)
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, f"{label} ID:{track.track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Đếm đối tượng
                    if label == 'Vehicle':
                        current_vehicle_count += 1
                        if track.track_id not in tracked_ids:
                            tracked_ids.add(track.track_id)
                            total_vehicle_count += 1
                    else:
                        current_pedestrian_count += 1
                        if track.track_id not in tracked_ids:
                            tracked_ids.add(track.track_id)
                            total_pedestrian_count += 1

        # Hiển thị số lượng đối tượng trên video
        cv2.putText(frame_resized, f"Current Vehicles: {current_vehicle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_resized, f"Total Vehicles: {total_vehicle_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_resized, f"Current People: {current_pedestrian_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_resized, f"Total People: {total_pedestrian_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame_resized)

        # Hiển thị tiến độ xử lý
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = (current_frame / frame_count) * 100
        print(f"\rProcessing video... {current_frame}/{frame_count} frames ({progress:.2f}%)", end="")

    cap.release()
    out.release()
    print(f"\nVideo đầu ra đã được lưu tại {output_path}")
    
def process_multiple_videos(video_paths, output_dir, confidence):
    start_time = time.time()
    print(f"Thời gian bắt đầu: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model = load_yolo_model()

    total_videos = len(video_paths)

    for idx, video_path in enumerate(video_paths, 1):
        if not Path(video_path).exists():
            print(f"Error: Video file {video_path} does not exist.")
            continue

        video_name = Path(video_path).name
        output_path = Path(output_dir) / f"result_{video_name}"
        print(f"Processing {video_path} ({idx}/{total_videos})...")
        process_video(video_path, str(output_path), model, confidence)

    end_time = time.time()
    print(f"Thời gian kết thúc: {time.strftime('%H:%M:%S', time.localtime(end_time))}")

    completion_time = end_time - start_time
    hours, rem = divmod(completion_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Thời gian hoàn thành: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Object Detection")
    parser.add_argument('--confidence', type=float, default=0.6, help='Default confidence threshold for YOLOv5')
    args = parser.parse_args()

    video_paths = [
        "video/test31.mp4",
    ]

    output_dir = "result_videos"

    process_multiple_videos(video_paths, output_dir, args.confidence)
