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
DURATION = 30

AUDIO_DEVICE_NAME = "Teensy"
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1

GELSIGHT_FPS = 30


SESSION_NAME = datetime.now().strftime("session_%Y%m%d_%H%M%S")
os.makedirs(SESSION_NAME, exist_ok=True)

sensor_file = os.path.join(SESSION_NAME, "sensors.csv")
video_file = os.path.join(SESSION_NAME, "gelsight.mp4")
video_ts_file = os.path.join(SESSION_NAME, "gelsight_timestamps.csv")
audio_file = os.path.join(SESSION_NAME, "audio.wav")
audio_ts_file = os.path.join(SESSION_NAME, "audio_timestamps.csv")

stop_event = threading.Event()

start_time_ns = None

sensor_count = 0
bad_count = 0
video_count = 0
audio_sample_index = 0


def now_s():
    return (time.perf_counter_ns() - start_time_ns) / 1e9


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
                writer.writerow([now_s()] + parts)
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


with open(video_ts_file, "w", newline="") as vts_f, \
     open(audio_ts_file, "w", newline="") as ats_f, \
     sf.SoundFile(
         audio_file,
         mode="w",
         samplerate=AUDIO_SAMPLE_RATE,
         channels=AUDIO_CHANNELS,
         subtype="PCM_16",
     ) as wav_file:

    video_ts_writer = csv.writer(vts_f)
    audio_ts_writer = csv.writer(ats_f)

    video_ts_writer.writerow([
        "frame_idx",
        "pc_time_s",
    ])

    audio_ts_writer.writerow([
        "sample_start",
        "frames",
        "pc_time_s_callback",
        "input_buffer_adc_time",
        "current_time",
    ])

    def audio_callback(indata, frames, time_info, status):
        global audio_sample_index

        if status:
            print("Audio status:", status)

        callback_time_s = now_s()

        wav_file.write(indata)

        audio_ts_writer.writerow([
            audio_sample_index,
            frames,
            callback_time_s,
            time_info.inputBufferAdcTime,
            time_info.currentTime,
        ])

        audio_sample_index += frames

    with sd.InputStream(
        device=audio_device,
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype="int16",
        callback=audio_callback,
    ):
        start_time_ns = time.perf_counter_ns()

        t = threading.Thread(target=serial_thread_func)
        t.start()

        last_status = time.perf_counter()
        next_frame_time = time.perf_counter()

        while now_s() < DURATION:
            current_perf = time.perf_counter()

            if current_perf >= next_frame_time:
                frame_time_s = now_s()
                frame = gsmini.capture_image()

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                video_writer.write(frame)
                video_ts_writer.writerow([video_count, frame_time_s])
                video_count += 1

                next_frame_time += 1.0 / GELSIGHT_FPS

            if current_perf - last_status >= 1.0:
                elapsed = now_s()
                print(
                    f"elapsed={elapsed:.1f}s, "
                    f"sensor_samples={sensor_count}, "
                    f"sensor_rate={sensor_count / elapsed:.1f} Hz, "
                    f"video_frames={video_count}, "
                    f"video_rate={video_count / elapsed:.1f} FPS, "
                    f"audio_samples={audio_sample_index}, "
                    f"audio_rate={audio_sample_index / elapsed:.1f} Hz, "
                    f"bad_lines={bad_count}"
                )
                last_status = current_perf

        stop_event.set()
        t.join()


video_writer.release()

print("Saved:")
print(sensor_file)
print(video_file)
print(video_ts_file)
print(audio_file)
print(audio_ts_file)
print("Sensor samples:", sensor_count)
print("Video frames:", video_count)
print("Audio samples:", audio_sample_index)
print("Bad lines:", bad_count)