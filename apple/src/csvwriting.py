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
gelsight_folder = os.path.join(SESSION_NAME, "gelsight")
gelsight_index_file = os.path.join(SESSION_NAME, "gelsight_index.csv")

os.makedirs(gelsight_folder, exist_ok=True)


# ---------- GelSight ----------
gsmini = p3d.GelsightMini()
gsmini.connect()
print("GelSight connected")


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

gelsight_header = [
    "pc_time_s",
    "frame_id",
    "filename",
]


sensor_count = 0
gelsight_count = 0
bad_count = 0

last_status = time.time()
last_gelsight_capture = 0
gelsight_interval = 1.0 / GELSIGHT_FPS

print("Recording to folder:", SESSION_NAME)


with open(sensor_file, "w", newline="") as sensor_f, \
     open(gelsight_index_file, "w", newline="") as gel_f:

    sensor_writer = csv.writer(sensor_f)
    gel_writer = csv.writer(gel_f)

    sensor_writer.writerow(sensor_header)
    gel_writer.writerow(gelsight_header)

    start = time.time()

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

        # ---------- Capture GelSight ----------
        if now - last_gelsight_capture >= gelsight_interval:
            img = gsmini.capture_image()

            filename = f"gelsight_{gelsight_count:06d}.png"
            path = os.path.join(gelsight_folder, filename)

            cv2.imwrite(path, img)
            gel_writer.writerow([pc_time_s, gelsight_count, filename])

            gelsight_count += 1
            last_gelsight_capture = now

        # ---------- Status ----------
        if now - last_status >= 1.0:
            elapsed = now - start
            print(
                f"elapsed={elapsed:.1f}s, "
                f"sensor_samples={sensor_count}, "
                f"gelsight_frames={gelsight_count}, "
                f"bad_lines={bad_count}"
            )
            last_status = now


ser.close()

print("Saved:")
print(sensor_file)
print(gelsight_index_file)
print(gelsight_folder)
print("Sensor samples:", sensor_count)
print("GelSight frames:", gelsight_count)
print("Bad lines:", bad_count)