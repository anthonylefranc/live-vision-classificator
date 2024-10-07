# Import libraries
from ultralytics import YOLO  # Import YOLO model from Ultralytics
import cv2                   # Import OpenCV library
import math                  # Import math module for mathematical operations

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

# Infinite loop to continuously capture frames from the camera
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Perform object detection using the YOLO model on the captured frame
    results = model(img, stream=True)

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
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)

            # Determine and print the class name of the detected object
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Draw text indicating the class name on the frame
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Display the frame with detected objects in a window named "Webcam"
    cv2.imshow('WEBCAM STREAM', img)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()