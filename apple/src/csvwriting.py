import serial
import csv
import time
from datetime import datetime

# Change this to your Teensy serial port
PORT = "/dev/tty.usbmodem135529601" 

BAUD = 921600

RECORD_SECONDS = 60

# ================= SERIAL =================
ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)

# ================= FILE NAME =================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"session_{timestamp}.csv"

print(f"Recording to {filename}")

# ================= RECORD =================
start_time = time.time()

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)

    # CSV header
    writer.writerow([
        "time_ms",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "load_raw"
    ])

    while time.time() - start_time < RECORD_SECONDS:

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        print(line)

        parts = line.split(",")

        # only save valid rows
        if len(parts) == 8:
            writer.writerow(parts)

# ================= CLEANUP =================
ser.close()

print("Recording complete.")