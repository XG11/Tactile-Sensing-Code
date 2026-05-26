import serial
import csv
import time
import os
from datetime import datetime

import cv2
import py3DCal as p3d


SERIAL_PORT = "/dev/tty.usbmodem135529601"
BAUD = 2000000
DURATION = 10

GELSIGHT_FPS = 30

SESSION_NAME = datetime.now().strftime("session_%Y%m%d_%H%M%S")
os.makedirs(SESSION_NAME, exist_ok=True)

sensor_file = os.path.join(SESSION_NAME, "sensors.csv")
video_file = os.path.join(SESSION_NAME, "gelsight.mp4")


# ---------- GelSight ----------
gsmini = p3d.GelsightMini()
gsmini.connect()
print("GelSight connected")

# Capture one frame to get size
first_frame = gsmini.capture_image()
height, width = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    video_file,
    fourcc,
    GELSIGHT_FPS,
    (width, height)
)

if not video_writer.isOpened():
    raise RuntimeError("Could not open MP4 video writer")


# ---------- Serial ----------
ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.01)
time.sleep(2)
ser.reset_input_buffer()


sensor_header = [
    "pc_time_s",
    "teensy_time_us",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "load_raw",
]

sensor_count = 0
video_count = 0
bad_count = 0

last_status = time.time()
last_gelsight_capture = 0
gelsight_interval = 1.0 / GELSIGHT_FPS

print("Recording to folder:", SESSION_NAME)


with open(sensor_file, "w", newline="") as sensor_f:
    sensor_writer = csv.writer(sensor_f)
    sensor_writer.writerow(sensor_header)

    start = time.time()

    # Write first frame
    video_writer.write(first_frame)
    video_count += 1
    last_gelsight_capture = start

    while time.time() - start < DURATION:
        now = time.time()
        pc_time_s = now - start

        # ---------- Read Teensy serial ----------
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if line and not line.startswith("time_us") and not line.startswith("#"):
            parts = line.split(",")

            if len(parts) == 8:
                sensor_writer.writerow([pc_time_s] + parts)
                sensor_count += 1
            else:
                bad_count += 1

        # ---------- Capture GelSight frame ----------
        if now - last_gelsight_capture >= gelsight_interval:
            frame = gsmini.capture_image()

            # Make sure frame size matches video size
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            video_writer.write(frame)
            video_count += 1
            last_gelsight_capture += gelsight_interval

        # ---------- Status ----------
        if now - last_status >= 1.0:
            elapsed = now - start
            print(
                f"elapsed={elapsed:.1f}s, "
                f"sensor_samples={sensor_count}, "
                f"video_frames={video_count}, "
                f"bad_lines={bad_count}"
            )
            last_status = now


ser.close()
video_writer.release()

print("Saved:")
print(sensor_file)
print(video_file)
print("Sensor samples:", sensor_count)
print("Video frames:", video_count)