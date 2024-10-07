import cv2
import numpy as np

# Load the pre-trained OpenCV deep learning face detector (Caffe model)
face_detector = cv2.dnn.readNetFromCaffe(
    r"C:\Users\antho\ml_projects_2024\real_time_vision_classificator\models\deploy.prototxt", 
    r"C:\Users\antho\ml_projects_2024\real_time_vision_classificator\models\res10_300x300_ssd_iter_140000.caffemodel"
)

# Prepare Local Binary Patterns Histogram (LBPH) recognizer for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Anthony's face from images folder and assign a label
anthony_img = cv2.imread(r"C:\Users\antho\ml_projects_2024\real_time_vision_classificator\images\anthony.jpg", cv2.IMREAD_GRAYSCALE)

# Check if image was loaded correctly
if anthony_img is None:
    print("Error: Anthony's image not found.")
    exit()

recognizer.train([anthony_img], np.array([0]))  # Label "0" for Anthony
labels = {0: "Anthony"}  # Map label 0 to Anthony's name

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

while True:
    # Capture frame from webcam
    success, img = cap.read()
    
    # Convert the frame to grayscale for face recognition
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prepare the frame for face detection (using the deep learning face detector)
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Only consider confident detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Extract the region of interest for face recognition
            face = gray_frame[y1:y2, x1:x2]

            # Perform face recognition
            try:
                label, confidence = recognizer.predict(face)
            except:
                continue  # Skip if face recognition fails

            # Display the name if recognized
            name = labels.get(label, "Unknown")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show the video feed with face detection/recognition
    cv2.imshow('Video', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
