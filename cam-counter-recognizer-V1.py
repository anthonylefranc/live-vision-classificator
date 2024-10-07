# Import libraries
from ultralytics import YOLO  # Import YOLO model from Ultralytics
import cv2                    # Import OpenCV library
import math                   # Import math module for mathematical operations
import time                   # Import time module for tracking time

# Start webcam
cap = cv2.VideoCapture(1)    # Open default camera (index 1 for secondary webcam)
cap.set(3, 640)               # Set frame width to 640 pixels
cap.set(4, 480)               # Set frame height to 480 pixels

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")  # Load YOLOv8 model with pre-trained weights

# Define object classes for detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Initialize counters for people, cars, and trucks
people_count = 0
car_count = 0
truck_count = 0

# Track start time for elapsed time
start_time = time.time()

# Infinite loop to continuously capture frames from the camera
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Perform object detection using the YOLO model on the captured frame
    results = model(img, stream=True)

    # Reset counts for each frame
    current_people = 0
    current_cars = 0
    current_trucks = 0

    # Iterate through the results of object detection
    for r in results:
        boxes = r.boxes  # Extract bounding boxes for detected objects

        # Iterate through each bounding box
        for box in boxes:
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integer values

            # Draw the bounding box on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Calculate and print the confidence score of the detection
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Determine the class name of the detected object
            cls = int(box.cls[0])
            class_name = classNames[cls]
            print("Class name -->", class_name)

            # Draw text indicating the class name on the frame
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

            # Update counters based on detected class
            if class_name == "person":
                current_people += 1
            elif class_name == "car":
                current_cars += 1
            elif class_name == "truck":
                current_trucks += 1

    # Update the total counts
    people_count += current_people
    car_count += current_cars
    truck_count += current_trucks

    # Calculate the elapsed time
    elapsed_time = int(time.time() - start_time)  # Elapsed time in seconds

    # Overlay the counters and elapsed time on the frame
    overlay_text = f"People: {people_count}, Cars: {car_count}, Trucks: {truck_count}, Time: {elapsed_time}s"
    cv2.putText(img, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with detected objects and counters in a window named "WEBCAM STREAM"
    cv2.imshow('WEBCAM STREAM', img)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
