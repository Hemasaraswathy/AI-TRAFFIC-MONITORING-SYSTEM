# 🚦 AI Traffic Monitoring System

An AI-powered real-time traffic monitoring system developed using **Python, OpenCV, and YOLOv8**. The system analyzes traffic video footage, detects vehicles, tracks them, and counts different types of vehicles automatically.

This project demonstrates how **Computer Vision and Deep Learning** can be applied to intelligent traffic monitoring and vehicle analysis.

## 📌 Project Overview

Traditional traffic monitoring requires manual observation and counting, which can be time-consuming and inaccurate.

The **AI Traffic Monitoring System** automates this process by using the **YOLOv8 object detection model** to identify vehicles from traffic videos. The system processes each frame, detects vehicles, tracks their movement, and maintains vehicle counts.

## ✨ Features

* 🚗 Automatic vehicle detection
* 🏍️ Motorcycle detection
* 🚌 Bus detection
* 🚚 Truck detection
* 🔢 Automatic vehicle counting
* 🎯 Real-time object tracking
* 🎥 Traffic video processing
* 🖥️ OpenCV-based video display
* 📊 Vehicle count information displayed on the video
* 🤖 YOLOv8-based object detection
* 📱 Supports different video orientations
* ⚡ Frame-by-frame video analysis

## 🧠 Vehicle Classes

The system currently detects the following vehicle categories:

| Class ID | Vehicle        |
| -------- | -------------- |
| 2        | 🚗 Car         |
| 3        | 🏍️ Motorcycle |
| 5        | 🚌 Bus         |
| 7        | 🚚 Truck       |

These classes are based on the **COCO dataset classes used by YOLOv8**.

## 🛠️ Technologies Used

* **Python** – Main programming language
* **YOLOv8** – Object detection model
* **Ultralytics** – YOLO implementation
* **OpenCV** – Video processing and visualization
* **NumPy** – Numerical and image-array operations
* **Git & GitHub** – Version control and project hosting

## 📂 Project Structure

```text
AI-TRAFFIC-MONITORING-SYSTEM/
│
├── main.py
├── yolov8n.pt
│
├── videos/
│   └── traffic.mp4
│
└── README.md
```

## 📄 Files Used

### `main.py`

The main Python program of the project.

It is responsible for:

* Loading the YOLOv8 model
* Reading the traffic video
* Detecting vehicles
* Identifying vehicle classes
* Tracking detected vehicles
* Counting vehicles
* Displaying detection results
* Showing vehicle counts on the video

### `yolov8n.pt`

The pretrained **YOLOv8 Nano** model used for vehicle detection.

The model provides fast object detection while keeping computational requirements relatively low.

### `videos/traffic.mp4`

The input traffic video used for testing the AI vehicle detection and counting system.

The system reads this video frame by frame and performs vehicle detection and tracking.

### `README.md`

This documentation file explains the project, technologies, files, features, installation process, and usage instructions.

## ⚙️ System Workflow

```text
Traffic Video
      ↓
OpenCV Video Capture
      ↓
YOLOv8 Object Detection
      ↓
Vehicle Identification
      ↓
Object Tracking
      ↓
Vehicle Counting
      ↓
Results Displayed on Video
```

## 🔄 How the System Works

1. The program loads the pretrained **YOLOv8 model**.
2. The traffic video is opened using **OpenCV**.
3. Each frame of the video is processed.
4. YOLOv8 identifies objects present in the frame.
5. Vehicle classes such as cars, motorcycles, buses, and trucks are selected.
6. Detected vehicles are tracked across frames.
7. Unique vehicles are counted to reduce duplicate counting.
8. The vehicle counts are displayed along with the processed video.
9. The final result provides an automated overview of traffic flow.

## 💻 Requirements

Make sure Python is installed on your system.

Install the required libraries using:

```bash
pip install ultralytics opencv-python numpy
```

## ▶️ How to Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/Hemasaraswathy/AI-TRAFFIC-MONITORING-SYSTEM.git
```

### Step 2 — Open the project

```bash
cd AI-TRAFFIC-MONITORING-SYSTEM
```

### Step 3 — Install dependencies

```bash
pip install ultralytics opencv-python numpy
```

### Step 4 — Make sure the required files are present

```text
main.py
yolov8n.pt
videos/traffic.mp4
```

### Step 5 — Run the project

```bash
python main.py
```

The traffic video will open and the detected vehicles and their counts will be displayed.

## 📊 Example Output

The system displays the processed traffic video with bounding boxes around detected vehicles and vehicle-count information.

Example:

```text
Car:          XX
Motorcycle:   XX
Bus:          XX
Truck:        XX
Total:        XX
```

The actual values depend on the input traffic video.

## 🎯 Applications

This system can be used as a foundation for:

* Smart traffic monitoring
* Intelligent transportation systems
* Vehicle density analysis
* Traffic flow monitoring
* Road surveillance
* Automated vehicle counting
* Smart city applications
* Traffic management systems

## 🚀 Future Enhancements

The project can be further improved by adding:

* 📈 Traffic density graphs
* 🚦 Traffic signal optimization
* 🛣️ Multiple lane monitoring
* 📹 Live CCTV camera support
* 🚨 Accident detection
* ⚠️ Wrong-way vehicle detection
* 📱 Web dashboard
* ☁️ Cloud-based monitoring
* 📊 Database integration
* 📍 Location-based traffic analysis
* 🧠 Advanced vehicle tracking
* 📑 Automatic traffic reports

## 🔐 Note

The current project is designed for educational and demonstration purposes. Detection and counting accuracy can vary depending on video quality, camera angle, lighting conditions, vehicle overlap, and traffic density.

## 👩‍💻 Author

**Hema Saraswathy**

GitHub:
https://github.com/Hemasaraswathy

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

### 🏷️ Keywords

`Artificial Intelligence` `Computer Vision` `YOLOv8` `Python` `OpenCV` `Vehicle Detection` `Vehicle Counting` `Object Tracking` `Traffic Monitoring` `Deep Learning` `Smart Transportation` `Smart City`
