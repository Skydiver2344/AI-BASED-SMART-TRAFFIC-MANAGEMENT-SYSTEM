import cv2
from ultralytics import YOLO


def detect_vehicles(frame, model):
    """Detect objects in the given video frame using YOLO."""
    results = model(frame)
    return results


def count_vehicles(boxes, classes, vehicle_classes, roi):
    """Count vehicles whose bounding-box centers lie inside the ROI."""
    vehicle_count = 0
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    for box, cls in zip(boxes, classes):
        if int(cls) not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Count only vehicles inside the ROI.
        if (roi_x1 <= center_x <= roi_x2 and
                roi_y1 <= center_y <= roi_y2):
            vehicle_count += 1

    return vehicle_count


def draw_detections(frame, results, model, vehicle_classes, roi):
    """Draw bounding boxes, center points, and labels for vehicles inside the ROI."""
    classes = results[0].boxes.cls
    boxes = results[0].boxes.xyxy
    annotated_frame = frame.copy()

    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    for box, cls in zip(boxes, classes):
        if int(cls) not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Skip vehicles outside the ROI.
        if not (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
            continue

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

        label = model.names[int(cls)]
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(y1 - text_height - 8, 0)

        cv2.rectangle(annotated_frame, (x1, label_top), (x1 + text_width + 15, y1), (0, 255, 0), -1)
        cv2.putText(annotated_frame, label.upper(), (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (255, 255, 255), 2)

    return annotated_frame


# Load the YOLO model.
model = YOLO("C:/Users/divya/OneDrive/Desktop/MINOR PROJECT 5th sem/Code/yolov8n.pt")

# COCO class IDs for cars, motorcycles, buses, and trucks.
vehicle_classes = [2, 3, 5, 7]

# Open the input video.
video = cv2.VideoCapture("C:/Users/divya/OneDrive/Desktop/MINOR PROJECT 5th sem/Code/video.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video width:", width)
print("Video height:", height)

# Define the ROI for the left carriageway.
roi_x1 = 0
roi_y1 = 0
roi_x2 = 900
roi_y2 = 1080
roi = (roi_x1, roi_y1, roi_x2, roi_y2)

# Process video frames until the video ends or the user exits.
while True:
    success, frame = video.read()
    if not success:
        break

    results = detect_vehicles(frame, model)
    vehicle_count = count_vehicles(results[0].boxes.xyxy, results[0].boxes.cls, vehicle_classes, roi)
    annotated_frame = draw_detections(frame, results, model, vehicle_classes, roi)

    # Display the current vehicle count.
    cv2.rectangle(annotated_frame, (10, 10), (250, 60), (0, 255, 0), -1)
    cv2.putText(annotated_frame, f"VEHICLES DETECTED: {vehicle_count}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Video", annotated_frame)

    # Exit when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources.
video.release()
cv2.destroyAllWindows()