import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

detection_count = 0
previous_face_count = 0

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ webcam. Thoát...")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        current_face_count = 0
        if results.detections:
            for detection in results.detections:
                # Vẽ khung hình
                mp_drawing.draw_detection(frame, detection)
                current_face_count += 1
            #Tính số mặt hiển thị và số mặt đã nhận diện
        if current_face_count > previous_face_count:
            detection_count += (current_face_count - previous_face_count)
        previous_face_count = current_face_count

        # cv2.putText(frame, f"Faces: {current_face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f"Total: {detection_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imshow('Mediapipe Face Detection', frame)

        print(f"Faces: {current_face_count}, Count: {detection_count}")

        # Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()