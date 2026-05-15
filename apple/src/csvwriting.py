import serial
import csv
import time

# Change this to your Teensy serial port
PORT = "/dev/tty.usbmodem135529601" 

BAUD = 921600
OUTPUT_FILE = "teensy_data_01.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    print("Recording... Press Ctrl+C to stop.")

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            print(line)

            # skip header if Teensy already prints it
            parts = line.split(",")

            if len(parts) == 8:
                writer.writerow(parts)
                f.flush()

    except KeyboardInterrupt:
        print("\nStopped recording.")

ser.close()