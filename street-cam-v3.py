# Import necessary libraries
from ultralytics import YOLO  # Import YOLO model
import cv2                    # Import OpenCV library
import time                   # Import time module for tracking time

# Start webcam with HD resolution (1920x1080)
cap = cv2.VideoCapture(1)  # Open default camera (index 1 for secondary webcam)
cap.set(3, 1920)           # Set frame width to 1920 pixels (Full HD width)
cap.set(4, 1080)           # Set frame height to 1080 pixels (Full HD height)

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")  # Load YOLOv8 model with pre-trained weights

# Define object classes for detection (subset: person, car, truck, dog, bicycle, motorbike)
classNames = ["person", "bicycle", "car", "motorbike", "truck", "dog"]

# Color map for bounding boxes and text
color_map = {
    "person": (0, 255, 0),      # Green
    "bicycle": (255, 255, 0),   # Cyan
    "car": (0, 0, 255),         # Red
    "truck": (255, 0, 0),       # Blue
    "dog": (0, 165, 255)        # Orange
}

# Function to draw bounding boxes and text on the frame, including a stroke for better visibility
def draw_boxes(frame, boxes, class_name, color):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Get coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = class_name
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3)  # Black stroke for visibility
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 1)  # Colored text

# Function to overlay the detection count in the top-middle position
def overlay_count(frame, counts):
    overlay_text = f'Persons: {counts["person"]}, Cars: {counts["car"]}, Trucks: {counts["truck"]}, Bicycles: {counts["bicycle"]}, Dogs: {counts["dog"]}'
    text_size = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2  # Center the text
    cv2.putText(frame, overlay_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 3)  # Black stroke for visibility
    cv2.putText(frame, overlay_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)  # White text

# Function to overlay the quit message at the bottom-middle position
def overlay_quit_message(frame):
    quit_message = 'Type "q" to quit the program'
    text_size = cv2.getTextSize(quit_message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2  # Center the text
    text_y = frame.shape[0] - 30  # Position above the bottom
    cv2.putText(frame, quit_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 3)  # Black stroke for visibility
    cv2.putText(frame, quit_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)  # White text

# Infinite loop to continuously capture frames from the camera
while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        break

    # Perform object detection using the YOLO model on the captured frame
    results = model(img)
    
    # Initialize object counts
    counts = {"person": 0, "car": 0, "truck": 0, "bicycle": 0, "dog": 0}

    # Track the bounding boxes for each class
    person_boxes = []
    car_boxes = []
    truck_boxes = []
    bicycle_boxes = []
    dog_boxes = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extract the coordinates from YOLO's detection (bounding boxes)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class label and confidence
            cls = int(box.cls[0])
            conf = box.conf[0].item()

            # Check if the class index is within the bounds of classNames list
            if cls < len(classNames):
                class_name = classNames[cls]
            else:
                continue  # Skip this detection if the class index is out of range

            # Update counts and add bounding box to respective list
            if class_name == "person":
                counts["person"] += 1
                person_boxes.append([x1, y1, x2, y2])
            elif class_name == "car":
                counts["car"] += 1
                car_boxes.append([x1, y1, x2, y2])
            elif class_name == "truck":
                counts["truck"] += 1
                truck_boxes.append([x1, y1, x2, y2])
            elif class_name == "bicycle":
                counts["bicycle"] += 1
                bicycle_boxes.append([x1, y1, x2, y2])
            elif class_name == "dog":
                counts["dog"] += 1
                dog_boxes.append([x1, y1, x2, y2])

    # Draw bounding boxes for each detected class
    draw_boxes(img, person_boxes, "person", color_map["person"])
    draw_boxes(img, car_boxes, "car", color_map["car"])
    draw_boxes(img, truck_boxes, "truck", color_map["truck"])
    draw_boxes(img, bicycle_boxes, "bicycle", color_map["bicycle"])
    draw_boxes(img, dog_boxes, "dog", color_map["dog"])

    # Overlay the object counts at the top-middle position
    overlay_count(img, counts)

    # Overlay the quit message at the bottom-middle position
    overlay_quit_message(img)

    # Display the frame with detection and tracking
    cv2.imshow('YOLO Object Detection', img)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
