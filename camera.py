# Import the OpenCV library
import cv2

# Open the default camera (index 0). For an external webcam, you can switch to index (1)
cap = cv2.VideoCapture(0)  # Corrected placement of 'cap'

# Set the frame width to 640 pixels
cap.set(3, 640)

# Set the frame height to 480 pixels
cap.set(4, 480)

# Infinite loop to continuously capture frames from the camera
while True:
    # Read a frame from the camera
    ret, img = cap.read()

    # Check if frame was read correctly
    if not ret:
        print("Failed to grab frame")
        break

    # Display the captured frame in a window named "Cam"
    cv2.imshow('Cam', img)

    # Wait for a key press for 1 millisecond
    # If the pressed key is 'q', exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
