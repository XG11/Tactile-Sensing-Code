import serial
import csv
import time
import sounddevice as sd
import soundfile as sf

SERIAL_PORT = "/dev/tty.usbmodem135529601" 
BAUD = 115200
DURATION = 60

AUDIO_DEVICE_NAME = "Teensy"
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1

sensor_file = "sensors.csv"
audio_file = "audio.wav"

# Find Teensy audio device
devices = sd.query_devices()
audio_device = None

for i, d in enumerate(devices):
    if AUDIO_DEVICE_NAME.lower() in d["name"].lower() and d["max_input_channels"] > 0:
        audio_device = i
        print("Using audio device:", i, d["name"])
        break

if audio_device is None:
    raise RuntimeError("Could not find Teensy audio input device")

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)

audio_frames = []

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_frames.append(indata.copy())

print("Recording...")

with sf.SoundFile(audio_file, mode="w",
                  samplerate=AUDIO_SAMPLE_RATE,
                  channels=AUDIO_CHANNELS,
                  subtype="PCM_16") as wav_file:

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        wav_file.write(indata)

    with sd.InputStream(device=audio_device,
                        samplerate=AUDIO_SAMPLE_RATE,
                        channels=AUDIO_CHANNELS,
                        dtype="int16",
                        callback=audio_callback):

        with open(sensor_file, "w", newline="") as f:
            writer = csv.writer(f)

            start = time.time()

            while time.time() - start < DURATION:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                print(line)

                parts = line.split(",")

                if len(parts) == 8:
                    writer.writerow(parts)

ser.close()

print("Saved:")
print(sensor_file)
print(audio_file)