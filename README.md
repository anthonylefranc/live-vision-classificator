# Live Vision Classificator

This project implements live object recognition and classification using YOLO through a webcam feed. The system captures live video, recognizes objects, and displays information such as bounding boxes and object counts in real-time. It also explores various attempts at object tracking and face recognition.

## Project Structure

### 1. **camera.py**
   - Activates the webcam and displays the live feed.
   - Basic script to test camera functionality.

### 2. **object-cam.py**
   - Implements object recognition using the YOLO model.
   - Recognizes objects like people, cars, bicycles, etc., and displays bounding boxes around them.

### 3. **cam-counter-recognizer-V1.py** & **cam-counter-recognizer-V2.py**
   - Initial attempts to implement unique face recognition.
   - These versions tried to differentiate between multiple faces in the feed but were eventually abandoned.

### 4. **Street-cam.py**
   - Extends `object-cam.py` to recognize objects in a street environment.
   - Focuses on detecting street-specific objects such as cars, people, and bicycles.

### 5. **Street-cam-V2.py**
   - Adds SORT and DeepSORT tracking to keep objects in memory and track them as they move.
   - However, this approach proved ineffective as the object tags flickered, leading to inconsistent tracking.

### 6. **Street-cam-V3.py**
   - Latest version of the object recognition system.
   - Uses only YOLO for object recognition, removing the ineffective SORT and DeepSORT trackers.
   - Includes a live counter for tracking the number of detected objects (e.g., people, cars, trucks).
   - Displays categories in real-time for live street monitoring via webcam.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/anthonylefranc/live-vision-classificator.git
