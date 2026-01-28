import serial, struct, numpy as np, matplotlib.pyplot as plt, time
from pathlib import Path

PORT = '/dev/cu.usbmodem160572101' 
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=2)
out_directory = "recordings"
session_name = "slide11"
Path(out_directory).mkdir(exist_ok=True)
bin_path = Path(out_directory) / f"{session_name}.bin"
meta_path = Path(out_directory) / f"{session_name}.meta"

r0, c0 = 0, 24    # top- corner
H, W = 10, 10  

def read_exactly_number(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise RuntimeError("Serial timeout")
        buf.extend(chunk)
    return buf

frame_count = 0
MAX_FRAME_COUNT = 1000
# recording
with open(bin_path, "wb") as bin_file:   #"wb" means write binary which overwirte
    #change to "ab" for appending
    while frame_count < MAX_FRAME_COUNT:
        if ser.read(1) != b'D':
            continue
        if ser.read(3) != b"ATA":
            continue

        FRAMES = struct.unpack('<i', ser.read(4))[0]
        ROWS   = struct.unpack('<i', ser.read(4))[0]
        COLS   = struct.unpack('<i', ser.read(4))[0]
        print("Packet Frames", FRAMES) #ok just remember FRAMES var is arbitrary

        nbytes = FRAMES * ROWS * COLS

        print("receiving bytes")
        payload = read_exactly_number(nbytes)

        if read_exactly_number(4) != b"DONE":
            print("footer mismatch")
            continue

        bin_file.write(payload)
        bin_file.flush()

        frame_count += FRAMES
        print(f"stored {frame_count} frames total")

        if not meta_path.exists():
            with open(meta_path, "w") as f:
                f.write(f"frames_per_packet={FRAMES}\n")
                f.write(f"rows={ROWS}\n")
                f.write(f"cols={COLS}\n")
                f.write(f"bytes_per_frame = {ROWS*COLS}\n")
    