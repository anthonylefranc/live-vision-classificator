# YOLO + SORT Object Detection and Tracking System

## Overview

This program is a real-time object detection and tracking system that utilizes the **YOLOv8** model in combination with the **SORT** tracking algorithm. The goal of the application is to detect and track multiple object classes, such as **people, cars, trucks, dogs, bicycles, and motorbikes**, using a webcam feed.

The program draws bounding boxes around detected objects, assigns unique IDs to each tracked object, and provides a live count for each category. This can be useful in a variety of settings, such as traffic monitoring, surveillance, or analysis of crowded environments.

## Features

- **Real-Time Object Detection**: Utilizes the YOLOv8 model for detecting multiple objects in a webcam feed.
- **Object Tracking with SORT**: Assigns unique IDs to objects, even if they leave and re-enter the frame.
- **Class Counting and Overlay**: Tracks and displays the count of detected objects by category, with on-screen overlay information for better visualization.
- **User Instructions Overlay**: Displays a message to guide the user on how to stop the program.

## How It Works

1. **Model and Camera Setup**:
   - The program starts by initializing a **YOLOv8** model with pre-trained weights and a **SORT** tracker.
   - The webcam (set to Full HD resolution) is used as the source for capturing real-time video frames.

2. **Object Detection**:
   - Each frame from the webcam is processed using the YOLOv8 model, which detects various object classes such as people, cars, trucks, dogs, bicycles, and motorbikes.
   - Only objects with a detection confidence score above a certain threshold are considered to reduce false positives.

3. **Object Tracking**:
   - Detected objects are passed to the **SORT** algorithm, which maintains unique IDs for each object.
   - The SORT tracker helps keep track of objects even when they move between frames, providing smooth tracking.

4. **Visualization**:
   - Bounding boxes with unique IDs are drawn around each detected object on the webcam feed.
   - The program also overlays text showing the total number of detected objects by category, along with the elapsed time since the program started.
   - An instruction message is added at the bottom of the frame to inform users how to stop the program (pressing the 'q' key).

## Usage

### Prerequisites
- **Python 3.6+**
- **YOLOv8**: The ultralytics package, which contains the YOLO model.
- **OpenCV**: Used for capturing video from the webcam and drawing bounding boxes.
- **NumPy**: For handling numerical data, especially for formatting detections.
- **SORT**: For tracking detected objects over time.

### Installation
1. **Clone the Repository**:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
