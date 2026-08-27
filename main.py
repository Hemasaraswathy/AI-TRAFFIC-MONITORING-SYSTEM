import cv2
import time
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# AI TRAFFIC MONITORING SYSTEM
# FINAL STABLE VERSION
# YOLO11m + BoT-SORT + Automatic Orientation + Vehicle Counting
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "videos" / "traffic.mp4"
MODEL_PATH = BASE_DIR / "yolo11m.pt"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "traffic_counted.mp4"


# ============================================================
# 2. YOLO SETTINGS
# ============================================================

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50

TRACKER = "botsort.yaml"

# YOLO image size
IMAGE_SIZE = 640

# Minimum frames before a track can be counted
MIN_TRACK_FRAMES = 4

# Maximum movement between consecutive frames
# Prevents sudden tracker jumps from creating false counts.
MAX_MOVEMENT_RATIO = 0.30


# ============================================================
# 3. VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())


# ============================================================
# 4. COUNTING LINE POSITION
# ============================================================

# Horizontal video:
# vertical line at 50%
#
# Vertical video:
# horizontal line at 65%
#
# The vertical position is intentionally lower because
# the uploaded traffic video has vehicles moving through
# the lower road area.

HORIZONTAL_LINE_POSITION = 0.50
VERTICAL_LINE_POSITION = 0.65

# Safety zone
LINE_MARGIN_PERCENT = 0.025


# ============================================================
# 5. DISPLAY
# ============================================================

WINDOW_NAME = "AI Traffic Monitoring System"

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 850


# ============================================================
# 6. TEXT DRAWING
# ============================================================

def draw_text(
    frame,
    text,
    position,
    font_scale=0.6,
    color=(255, 255, 255),
    thickness=2
):

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x = max(5, x)
    y = max(text_height + 8, y)

    cv2.rectangle(
        frame,
        (
            x - 5,
            y - text_height - 7
        ),
        (
            x + text_width + 5,
            y + baseline + 5
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 7. RESIZE DISPLAY
# ============================================================

def resize_for_display(frame):

    h, w = frame.shape[:2]

    scale = min(
        MAX_DISPLAY_WIDTH / w,
        MAX_DISPLAY_HEIGHT / h,
        1.0
    )

    new_width = max(
        1,
        int(w * scale)
    )

    new_height = max(
        1,
        int(h * scale)
    )

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# 8. SIDE CALCULATION
# ============================================================

def get_side(position, line, margin):

    if position < line - margin:
        return -1

    if position > line + margin:
        return 1

    return 0


# ============================================================
# 9. TIME FORMAT
# ============================================================

def format_time(seconds):

    seconds = max(0, seconds)

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    seconds = int(
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# 10. START
# ============================================================

print()
print("=" * 70)
print("              AI TRAFFIC MONITORING SYSTEM")
print("=" * 70)


# ============================================================
# 11. CHECK MODEL
# ============================================================

print("\nChecking YOLO model...")

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"""
YOLO model not found.

Expected:
{MODEL_PATH}

Make sure yolo11m.pt is inside the project folder.
"""
    )

print("YOLO model found.")


# ============================================================
# 12. CHECK VIDEO
# ============================================================

print("\nChecking traffic video...")

if not VIDEO_PATH.exists():

    raise FileNotFoundError(
        f"""
traffic.mp4 was not found.

Expected:
{VIDEO_PATH}

Your folder must contain:

AI TRAFFIC MONITORING SYSTEM
│
├── main.py
├── main_backup.py
├── yolo11m.pt
│
├── videos
│   └── traffic.mp4
│
└── output
"""
    )

print("Traffic video found.")


# ============================================================
# 13. LOAD MODEL
# ============================================================

print("\nLoading YOLO11m...")

model = YOLO(
    str(MODEL_PATH)
)

print("YOLO11m loaded successfully.")


# ============================================================
# 14. OPEN VIDEO
# ============================================================

print("\nOpening video...")

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    raise RuntimeError(
        "Video exists but OpenCV could not open it."
    )


# ============================================================
# 15. VIDEO INFORMATION
# ============================================================

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)


# ============================================================
# 16. AUTOMATIC ORIENTATION
# ============================================================

if height > width:

    orientation = "VERTICAL"

else:

    orientation = "HORIZONTAL"


print()
print("=" * 55)
print("VIDEO INFORMATION")
print("=" * 55)

print(f"Width       : {width}")
print(f"Height      : {height}")
print(f"FPS         : {fps:.2f}")
print(f"Frames      : {total_frames}")
print(f"Duration    : {format_time(duration)}")
print(f"Orientation : {orientation}")


# ============================================================
# 17. COUNTING LINE
# ============================================================

if orientation == "VERTICAL":

    line_position = int(
        height * VERTICAL_LINE_POSITION
    )

    line_margin = max(
        10,
        int(
            height * LINE_MARGIN_PERCENT
        )
    )

    print()
    print("COUNTING MODE : HORIZONTAL LINE")
    print(f"Line Y        : {line_position}")

else:

    line_position = int(
        width * HORIZONTAL_LINE_POSITION
    )

    line_margin = max(
        10,
        int(
            width * LINE_MARGIN_PERCENT
        )
    )

    print()
    print("COUNTING MODE : VERTICAL LINE")
    print(f"Line X        : {line_position}")


print(
    f"Safety margin : {line_margin}"
)


# ============================================================
# 18. OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# 19. COUNTERS
# ============================================================

vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}


total_count = 0


# Horizontal directions
left_to_right = 0
right_to_left = 0


# Vertical directions
top_to_bottom = 0
bottom_to_top = 0


# ============================================================
# 20. TRACKING DATA
# ============================================================

previous_positions = {}

last_valid_sides = {}

track_classes = {}

track_frames = {}

counted_ids = set()

last_seen = {}

track_history = {}


# ============================================================
# 21. EVENT
# ============================================================

last_event = ""
event_timer = 0


# ============================================================
# 22. PERFORMANCE
# ============================================================

frame_number = 0

processing_start = time.time()

display_fps = 0.0


# ============================================================
# 23. RESET FUNCTION
# ============================================================

def reset_counts():

    global vehicle_counts
    global total_count
    global left_to_right
    global right_to_left
    global top_to_bottom
    global bottom_to_top

    vehicle_counts = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }

    total_count = 0

    left_to_right = 0
    right_to_left = 0

    top_to_bottom = 0
    bottom_to_top = 0

    counted_ids.clear()


# ============================================================
# 24. MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        frame_number += 1


        # ====================================================
        # YOLO TRACKING
        # ====================================================

        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            classes=VEHICLE_CLASS_IDS,
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False
        )


        result = results[0]


        # ====================================================
        # COUNTING LINE
        # ====================================================

        if orientation == "VERTICAL":

            cv2.line(
                frame,
                (0, line_position),
                (width, line_position),
                (0, 255, 255),
                4
            )

            cv2.line(
                frame,
                (
                    0,
                    line_position - line_margin
                ),
                (
                    width,
                    line_position - line_margin
                ),
                (0, 120, 120),
                1
            )

            cv2.line(
                frame,
                (
                    0,
                    line_position + line_margin
                ),
                (
                    width,
                    line_position + line_margin
                ),
                (0, 120, 120),
                1
            )

            draw_text(
                frame,
                "COUNTING LINE",
                (
                    15,
                    max(
                        35,
                        line_position - 10
                    )
                ),
                font_scale=0.60,
                color=(0, 255, 255),
                thickness=2
            )

        else:

            cv2.line(
                frame,
                (line_position, 0),
                (line_position, height),
                (0, 255, 255),
                4
            )

            cv2.line(
                frame,
                (
                    line_position - line_margin,
                    0
                ),
                (
                    line_position - line_margin,
                    height
                ),
                (0, 120, 120),
                1
            )

            cv2.line(
                frame,
                (
                    line_position + line_margin,
                    0
                ),
                (
                    line_position + line_margin,
                    height
                ),
                (0, 120, 120),
                1
            )

            draw_text(
                frame,
                "COUNTING LINE",
                (
                    min(
                        line_position + 10,
                        max(10, width - 180)
                    ),
                    40
                ),
                font_scale=0.60,
                color=(0, 255, 255),
                thickness=2
            )


        # ====================================================
        # VEHICLES
        # ====================================================

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            class_ids = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )


            # =================================================
            # EACH TRACK
            # =================================================

            for (
                box,
                track_id,
                class_id,
                confidence
            ) in zip(
                boxes,
                track_ids,
                class_ids,
                confidences
            ):


                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                x1 = max(
                    0,
                    min(x1, width - 1)
                )

                x2 = max(
                    0,
                    min(x2, width - 1)
                )

                y1 = max(
                    0,
                    min(y1, height - 1)
                )

                y2 = max(
                    0,
                    min(y2, height - 1)
                )


                # ------------------------------------------------
                # CENTER
                # ------------------------------------------------

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # ------------------------------------------------
                # VEHICLE NAME
                # ------------------------------------------------

                vehicle_name = VEHICLE_CLASSES.get(
                    class_id,
                    "Vehicle"
                )


                track_classes[
                    track_id
                ] = vehicle_name


                # ------------------------------------------------
                # TRACK AGE
                # ------------------------------------------------

                track_frames[
                    track_id
                ] = (
                    track_frames.get(
                        track_id,
                        0
                    ) + 1
                )


                last_seen[
                    track_id
                ] = frame_number


                # ------------------------------------------------
                # POSITION FOR COUNTING
                # ------------------------------------------------

                if orientation == "VERTICAL":

                    current_position = center_y

                else:

                    current_position = center_x


                # ------------------------------------------------
                # HISTORY
                # ------------------------------------------------

                if track_id not in track_history:

                    track_history[
                        track_id
                    ] = []


                track_history[
                    track_id
                ].append(
                    (
                        center_x,
                        center_y
                    )
                )


                if len(
                    track_history[track_id]
                ) > 20:

                    track_history[
                        track_id
                    ] = track_history[
                        track_id
                    ][-20:]


                # ------------------------------------------------
                # DRAW BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # ------------------------------------------------
                # CENTER POINT
                # ------------------------------------------------

                cv2.circle(
                    frame,
                    (
                        center_x,
                        center_y
                    ),
                    5,
                    (0, 0, 255),
                    -1
                )


                # ------------------------------------------------
                # LABEL
                # ------------------------------------------------

                label = (
                    f"{vehicle_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )


                draw_text(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),
                    font_scale=0.48,
                    color=(0, 255, 0),
                    thickness=2
                )


                # =================================================
                # COUNTING
                # =================================================

                if (
                    track_frames[track_id]
                    >= MIN_TRACK_FRAMES
                ):


                    # ---------------------------------------------
                    # FIRST POSITION
                    # ---------------------------------------------

                    if track_id not in previous_positions:

                        previous_positions[
                            track_id
                        ] = current_position

                        initial_side = get_side(
                            current_position,
                            line_position,
                            line_margin
                        )

                        if initial_side != 0:

                            last_valid_sides[
                                track_id
                            ] = initial_side

                        continue


                    # ---------------------------------------------
                    # PREVIOUS POSITION
                    # ---------------------------------------------

                    previous_position = (
                        previous_positions[
                            track_id
                        ]
                    )


                    # ---------------------------------------------
                    # MOVEMENT
                    # ---------------------------------------------

                    movement = abs(
                        current_position
                        -
                        previous_position
                    )


                    if orientation == "VERTICAL":

                        dimension = height

                    else:

                        dimension = width


                    movement_ratio = (
                        movement
                        /
                        max(
                            dimension,
                            1
                        )
                    )


                    # Ignore impossible jumps
                    if (
                        movement_ratio
                        <= MAX_MOVEMENT_RATIO
                    ):


                        # -----------------------------------------
                        # CURRENT SIDE
                        # -----------------------------------------

                        current_side = get_side(
                            current_position,
                            line_position,
                            line_margin
                        )


                        # -----------------------------------------
                        # PREVIOUS VALID SIDE
                        # -----------------------------------------

                        previous_side = (
                            last_valid_sides.get(
                                track_id,
                                None
                            )
                        )


                        # -----------------------------------------
                        # ONLY VALID SIDES
                        # -----------------------------------------

                        if current_side != 0:


                            if previous_side is None:

                                last_valid_sides[
                                    track_id
                                ] = current_side


                            # =====================================
                            # HORIZONTAL VIDEO
                            # =====================================

                            elif (
                                orientation == "HORIZONTAL"
                                and
                                track_id not in counted_ids
                            ):


                                # LEFT -> RIGHT
                                if (
                                    previous_side == -1
                                    and
                                    current_side == 1
                                ):

                                    counted_ids.add(
                                        track_id
                                    )

                                    vehicle_counts[
                                        vehicle_name
                                    ] += 1

                                    left_to_right += 1

                                    last_event = (
                                        f"{vehicle_name} "
                                        f"ID:{track_id} "
                                        "LEFT -> RIGHT"
                                    )

                                    event_timer = 50


                                # RIGHT -> LEFT
                                elif (
                                    previous_side == 1
                                    and
                                    current_side == -1
                                ):

                                    counted_ids.add(
                                        track_id
                                    )

                                    vehicle_counts[
                                        vehicle_name
                                    ] += 1

                                    right_to_left += 1

                                    last_event = (
                                        f"{vehicle_name} "
                                        f"ID:{track_id} "
                                        "RIGHT -> LEFT"
                                    )

                                    event_timer = 50


                            # =====================================
                            # VERTICAL VIDEO
                            # =====================================

                            elif (
                                orientation == "VERTICAL"
                                and
                                track_id not in counted_ids
                            ):


                                # TOP -> BOTTOM
                                if (
                                    previous_side == -1
                                    and
                                    current_side == 1
                                ):

                                    counted_ids.add(
                                        track_id
                                    )

                                    vehicle_counts[
                                        vehicle_name
                                    ] += 1

                                    top_to_bottom += 1

                                    last_event = (
                                        f"{vehicle_name} "
                                        f"ID:{track_id} "
                                        "TOP -> BOTTOM"
                                    )

                                    event_timer = 50


                                # BOTTOM -> TOP
                                elif (
                                    previous_side == 1
                                    and
                                    current_side == -1
                                ):

                                    counted_ids.add(
                                        track_id
                                    )

                                    vehicle_counts[
                                        vehicle_name
                                    ] += 1

                                    bottom_to_top += 1

                                    last_event = (
                                        f"{vehicle_name} "
                                        f"ID:{track_id} "
                                        "BOTTOM -> TOP"
                                    )

                                    event_timer = 50


                            # Save valid side
                            last_valid_sides[
                                track_id
                            ] = current_side


                    # Save current position
                    previous_positions[
                        track_id
                    ] = current_position


        # ====================================================
        # TOTAL
        # ====================================================

        total_count = (
            vehicle_counts["Car"]
            +
            vehicle_counts["Motorcycle"]
            +
            vehicle_counts["Bus"]
            +
            vehicle_counts["Truck"]
        )


        # ====================================================
        # ACTIVE TRACKS
        # ====================================================

        active_tracks = sum(
            1
            for track_id, last_frame in last_seen.items()
            if frame_number - last_frame <= 10
        )


        # ====================================================
        # DASHBOARD
        # ====================================================

        dashboard_width = min(
            400,
            width - 20
        )

        dashboard_height = min(
            350,
            height - 20
        )


        overlay = frame.copy()


        cv2.rectangle(
            overlay,
            (10, 10),
            (
                dashboard_width,
                dashboard_height
            ),
            (0, 0, 0),
            -1
        )


        frame = cv2.addWeighted(
            overlay,
            0.72,
            frame,
            0.28,
            0
        )


        # ====================================================
        # TITLE
        # ====================================================

        draw_text(
            frame,
            "AI TRAFFIC MONITOR",
            (25, 40),
            font_scale=0.72,
            color=(255, 255, 255),
            thickness=2
        )


        # ====================================================
        # TOTAL
        # ====================================================

        draw_text(
            frame,
            f"TOTAL VEHICLES : {total_count}",
            (25, 78),
            font_scale=0.63,
            color=(0, 255, 255),
            thickness=2
        )


        # ====================================================
        # DIRECTION
        # ====================================================

        if orientation == "VERTICAL":

            draw_text(
                frame,
                f"TOP -> BOTTOM : {top_to_bottom}",
                (25, 112),
                font_scale=0.55,
                color=(0, 255, 0),
                thickness=2
            )

            draw_text(
                frame,
                f"BOTTOM -> TOP : {bottom_to_top}",
                (25, 142),
                font_scale=0.55,
                color=(0, 165, 255),
                thickness=2
            )

        else:

            draw_text(
                frame,
                f"LEFT -> RIGHT : {left_to_right}",
                (25, 112),
                font_scale=0.55,
                color=(0, 255, 0),
                thickness=2
            )

            draw_text(
                frame,
                f"RIGHT -> LEFT : {right_to_left}",
                (25, 142),
                font_scale=0.55,
                color=(0, 165, 255),
                thickness=2
            )


        # ====================================================
        # VEHICLE COUNTS
        # ====================================================

        draw_text(
            frame,
            f"Cars        : {vehicle_counts['Car']}",
            (25, 178),
            font_scale=0.55,
            color=(255, 255, 255),
            thickness=2
        )

        draw_text(
            frame,
            f"Motorcycles : {vehicle_counts['Motorcycle']}",
            (25, 207),
            font_scale=0.55,
            color=(255, 255, 255),
            thickness=2
        )

        draw_text(
            frame,
            f"Buses       : {vehicle_counts['Bus']}",
            (25, 236),
            font_scale=0.55,
            color=(255, 255, 255),
            thickness=2
        )

        draw_text(
            frame,
            f"Trucks      : {vehicle_counts['Truck']}",
            (25, 265),
            font_scale=0.55,
            color=(255, 255, 255),
            thickness=2
        )


        # ====================================================
        # ACTIVE TRACKS
        # ====================================================

        draw_text(
            frame,
            f"ACTIVE TRACKS : {active_tracks}",
            (25, 295),
            font_scale=0.48,
            color=(180, 180, 180),
            thickness=1
        )


        # ====================================================
        # FPS
        # ====================================================

        elapsed = (
            time.time()
            -
            processing_start
        )


        if elapsed > 0:

            display_fps = (
                frame_number
                /
                elapsed
            )


        draw_text(
            frame,
            f"FPS: {display_fps:.1f}",
            (
                max(
                    10,
                    width - 105
                ),
                35
            ),
            font_scale=0.50,
            color=(255, 255, 255),
            thickness=1
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if total_frames > 0:

            progress = (
                frame_number
                /
                total_frames
            )

        else:

            progress = 0


        progress = max(
            0,
            min(
                progress,
                1
            )
        )


        progress_width = min(
            400,
            width - 40
        )


        progress_x = 20

        progress_y = max(
            10,
            height - 25
        )


        cv2.rectangle(
            frame,
            (
                progress_x,
                progress_y
            ),
            (
                progress_x + progress_width,
                progress_y + 8
            ),
            (60, 60, 60),
            -1
        )


        cv2.rectangle(
            frame,
            (
                progress_x,
                progress_y
            ),
            (
                progress_x
                +
                int(
                    progress_width
                    *
                    progress
                ),
                progress_y + 8
            ),
            (0, 255, 255),
            -1
        )


        # ====================================================
        # EVENT
        # ====================================================

        if event_timer > 0:

            draw_text(
                frame,
                last_event,
                (
                    15,
                    max(
                        30,
                        height - 55
                    )
                ),
                font_scale=0.55,
                color=(0, 255, 255),
                thickness=2
            )

            event_timer -= 1


        # ====================================================
        # WRITE OUTPUT
        # ====================================================

        out.write(frame)


        # ====================================================
        # DISPLAY
        # ====================================================

        display_frame = resize_for_display(
            frame
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # Q = QUIT
        if key == ord("q"):

            print(
                "\nStopped by user."
            )

            break


        # R = RESET
        elif key == ord("r"):

            reset_counts()

            print(
                "\nAll counts reset."
            )


finally:

    cap.release()

    out.release()

    cv2.destroyAllWindows()


# ============================================================
# FINAL RESULT
# ============================================================

total_count = sum(
    vehicle_counts.values()
)


print()
print("=" * 70)
print("                 TRAFFIC COUNT COMPLETE")
print("=" * 70)

print()

print(
    f"Total Vehicles : {total_count}"
)

print(
    f"Cars           : {vehicle_counts['Car']}"
)

print(
    f"Motorcycles    : {vehicle_counts['Motorcycle']}"
)

print(
    f"Buses          : {vehicle_counts['Bus']}"
)

print(
    f"Trucks         : {vehicle_counts['Truck']}"
)

print()

if orientation == "VERTICAL":

    print(
        f"Top -> Bottom  : {top_to_bottom}"
    )

    print(
        f"Bottom -> Top  : {bottom_to_top}"
    )

else:

    print(
        f"Left -> Right  : {left_to_right}"
    )

    print(
        f"Right -> Left  : {right_to_left}"
    )

print()

print(
    "Output video:"
)

print(
    OUTPUT_PATH
)

print()

print("=" * 70)
print("Controls:")
print("Q = Quit")
print("R = Reset counts")
print("=" * 70)