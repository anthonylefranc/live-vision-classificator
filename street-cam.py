# Import necessary libraries
from ultralytics import YOLO  # Import YOLO model
import cv2                    # Import OpenCV library
import time                   # Import time module for tracking time
from sort import Sort          # Import SORT tracking algorithm
import numpy as np             # Import NumPy for array manipulation

# Initialize SORT tracker
tracker = Sort(max_age=45, min_hits=3, iou_threshold=0.35)

# Start webcam
cap = cv2.VideoCapture(1)  # Open default camera (index 1 for secondary webcam)
cap.set(3, 1280)            # Set frame width to 640 pixels
cap.set(4, 720)            # Set frame height to 480 pixels

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")  # Load YOLOv8 model with pre-trained weights

# Define object classes for detection (subset: person, car, truck)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize counters for unique detections
unique_people = set()
unique_cars = set()
unique_trucks = set()

# Track start time for elapsed time
start_time = time.time()

# Function to draw bounding boxes and text on the frame
def draw_boxes(frame, boxes, class_name, color=(0, 255, 0)):
    for box in boxes:
        x1, y1, x2, y2, obj_id = map(int, box)  # Extract coordinates and ID from SORT
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f'{class_name} {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Infinite loop to continuously capture frames from the camera
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Perform object detection using the YOLO model on the captured frame
    results = model(img, stream=True)
    detections = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extract the coordinates and the confidence score
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get class label and confidence
            cls = int(box.cls[0])
            conf = box.conf[0].item()

            # Check if the class index is within the bounds of classNames list
            if cls < len(classNames):
                class_name = classNames[cls]
            else:
                # Skip this detection if the class index is out of range
                continue

            # Only track persons, cars, and trucks
            if class_name in ["person", "car", "truck"] and conf > 0.5:
                detections.append([x1, y1, x2, y2, conf])  # Append detection for SORT tracker

    # Convert detections into numpy array
    np_detections = np.array(detections)

    # Update SORT tracker with the new detections
    tracks = tracker.update(np_detections)

    # Separate the tracks by class type
    person_tracks = []
    car_tracks = []
    truck_tracks = []

    for track in tracks:
        x1, y1, x2, y2, obj_id = track

        # Assign unique ID and add to the respective sets
        if class_name == "person":
            unique_people.add(obj_id)
            person_tracks.append(track)
        elif class_name == "car":
            unique_cars.add(obj_id)
            car_tracks.append(track)
        elif class_name == "truck":
            unique_trucks.add(obj_id)
            truck_tracks.append(track)

    # Draw the bounding boxes and overlay the tracked objects
    draw_boxes(img, person_tracks, "person", color=(0, 255, 0))
    draw_boxes(img, car_tracks, "car", color=(0, 0, 255))
    draw_boxes(img, truck_tracks, "truck", color=(255, 0, 0))

    # Calculate the elapsed time
    elapsed_time = int(time.time() - start_time)

    # Display counts and elapsed time as overlay text
    overlay_text = f'People: {len(unique_people)}, Cars: {len(unique_cars)}, Trucks: {len(unique_trucks)}, Time: {elapsed_time}s'
    cv2.putText(img, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display the frame with detection and tracking
    cv2.imshow('YOLO + SORT Tracking', img)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
