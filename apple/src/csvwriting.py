import serial
import csv
import time
import os
import threading
from datetime import datetime

import cv2
import sounddevice as sd
import soundfile as sf
import py3DCal as p3d


SERIAL_PORT = "/dev/tty.usbmodem135529601"
BAUD = 2000000
DURATION = 10

AUDIO_DEVICE_NAME = "Teensy"
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1

GELSIGHT_FPS = 30


SESSION_NAME = datetime.now().strftime("session_%Y%m%d_%H%M%S")
os.makedirs(SESSION_NAME, exist_ok=True)

sensor_file = os.path.join(SESSION_NAME, "sensors.csv")
video_file = os.path.join(SESSION_NAME, "gelsight.mp4")
audio_file = os.path.join(SESSION_NAME, "audio.wav")


stop_event = threading.Event()
start_time = None

sensor_count = 0
bad_count = 0


def serial_thread_func():
    global sensor_count, bad_count

    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.01)
    time.sleep(2)
    ser.reset_input_buffer()

    with open(sensor_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pc_time_s",
            "teensy_time_us",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "load_raw",
        ])

        while not stop_event.is_set():
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            if line.startswith("time_us") or line.startswith("#"):
                continue

            parts = line.split(",")

            if len(parts) == 8:
                pc_time_s = time.time() - start_time
                writer.writerow([pc_time_s] + parts)
                sensor_count += 1
            else:
                bad_count += 1

    ser.close()


# ---------- GelSight ----------
gsmini = p3d.GelsightMini()
gsmini.connect()
print("GelSight connected")

first_frame = gsmini.capture_image()
height, width = first_frame.shape[:2]

video_writer = cv2.VideoWriter(
    video_file,
    cv2.VideoWriter_fourcc(*"mp4v"),
    GELSIGHT_FPS,
    (width, height),
)

if not video_writer.isOpened():
    raise RuntimeError("Could not open MP4 video writer")


# ---------- Audio ----------
devices = sd.query_devices()
audio_device = None

for i, d in enumerate(devices):
    if AUDIO_DEVICE_NAME.lower() in d["name"].lower() and d["max_input_channels"] > 0:
        audio_device = i
        print("Using audio device:", i, d["name"])
        break

if audio_device is None:
    raise RuntimeError("Could not find Teensy audio input device")


print("Recording to folder:", SESSION_NAME)


with sf.SoundFile(
    audio_file,
    mode="w",
    samplerate=AUDIO_SAMPLE_RATE,
    channels=AUDIO_CHANNELS,
    subtype="PCM_16",
) as wav_file:

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        wav_file.write(indata)

    with sd.InputStream(
        device=audio_device,
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype="int16",
        callback=audio_callback,
    ):
        start_time = time.time()

        t = threading.Thread(target=serial_thread_func)
        t.start()

        video_count = 0
        last_status = time.time()
        next_frame_time = start_time

        while time.time() - start_time < DURATION:
            now = time.time()

            if now >= next_frame_time:
                frame = gsmini.capture_image()

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                video_writer.write(frame)
                video_count += 1
                next_frame_time += 1.0 / GELSIGHT_FPS

            if now - last_status >= 1.0:
                elapsed = now - start_time
                print(
                    f"elapsed={elapsed:.1f}s, "
                    f"sensor_samples={sensor_count}, "
                    f"sensor_rate={sensor_count / elapsed:.1f} Hz, "
                    f"video_frames={video_count}, "
                    f"video_rate={video_count / elapsed:.1f} FPS, "
                    f"bad_lines={bad_count}"
                )
                last_status = now

        stop_event.set()
        t.join()


video_writer.release()

print("Saved:")
print(sensor_file)
print(video_file)
print(audio_file)
print("Sensor samples:", sensor_count)
print("Bad lines:", bad_count)