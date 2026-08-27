
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open traffic video
video_path = "videos/traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Store vehicle IDs that have already been counted
counted_ids = set()

# Vehicle totals
vehicle_count = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    current_vehicles = 0

    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Ignore non-vehicle objects
            if class_id not in vehicle_classes:
                continue

            if confidence < 0.40:
                continue

            # Get tracking ID
            if box.id is None:
                continue

            track_id = int(box.id[0])

            vehicle_name = vehicle_classes[class_id]

            current_vehicles += 1

            # Count vehicle only once
            unique_id = (track_id, vehicle_name)

            if unique_id not in counted_ids:

                counted_ids.add(unique_id)

                vehicle_count[vehicle_name] += 1

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display vehicle ID
            label = f"{vehicle_name} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Total vehicles counted
    total_count = sum(vehicle_count.values())

    # Traffic level
    if current_vehicles <= 5:
        traffic_level = "LOW"
    elif current_vehicles <= 15:
        traffic_level = "MEDIUM"
    else:
        traffic_level = "HIGH"

    # Display information
    cv2.putText(
        frame,
        f"Cars Passed: {vehicle_count['Car']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Bikes Passed: {vehicle_count['Motorcycle']}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses Passed: {vehicle_count['Bus']}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks Passed: {vehicle_count['Truck']}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Vehicles: {total_count}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Current Traffic: {current_vehicles}",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic Level: {traffic_level}",
        (20, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Resize vertical video to fit screen
    height, width = frame.shape[:2]

    max_width = 900
    max_height = 700

    scale = min(
        max_width / width,
        max_height / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_frame = cv2.resize(
        frame,
        (new_width, new_height)
    )

    # Display
    cv2.imshow(
        "AI Traffic Monitoring System",
        resized_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()