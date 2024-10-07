# Import necessary libraries
from ultralytics import YOLO  # YOLO for person detection
import cv2                    # OpenCV for video feed
import math                   # Math operations for rounding
import face_recognition        # Face recognition for identifying Anthony

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)               # Set frame width
cap.set(4, 480)               # Set frame height

# Load YOLO model for person detection
model = YOLO("yolo-Weights/yolov8n.pt")

# Define object classes
classNames = ["person", ...]

# Load Anthony's face encoding (add your own images here)
anthony_image = face_recognition.load_image_file("anthony.jpg")
anthony_face_encoding = face_recognition.face_encodings(anthony_image)[0]

# List to store all known face encodings and their names
known_face_encodings = [anthony_face_encoding]
known_face_names = ["Anthony"]

# Variables to keep track of recognized persons and counts
recognized_persons = set()
total_persons_count = 0

while True:
    # Capture frame from webcam
    success, img = cap.read()

    # Perform object detection (to find people)
    results = model(img, stream=True)

    # Process each detection result
    for r in results:
        boxes = r.boxes

        # Process each bounding box
        for box in boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            
            # Only consider if the detected object is a person
            cls = int(box.cls[0])
            if classNames[cls] == "person":
                # Draw the bounding box for person
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # Perform face recognition on the detected person
                rgb_frame = img[:, :, ::-1]  # Convert frame from BGR to RGB

                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face matches Anthony
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                    
                    # Count unique persons
                    if name not in recognized_persons:
                        recognized_persons.add(name)
                        total_persons_count += 1

                    # Display the name and bounding box on the webcam feed
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the total number of persons detected on the camera feed
    cv2.putText(img, f'Total Persons: {total_persons_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the camera feed with bounding boxes and recognized names
    cv2.imshow('Cam', img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
