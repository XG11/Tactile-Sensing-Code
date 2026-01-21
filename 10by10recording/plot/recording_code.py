import serial, struct, numpy as np, matplotlib.pyplot as plt, time
from pathlib import Path

PORT = '/dev/cu.usbmodem160572101' 
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=2)
out_directory = "recordings"
session_name = "run01"
Path(out_directory).mkdir(exist_ok=True)
bin_path = Path(out_directory) / f"{session_name}.bin"
meta_path = Path(out_directory) / f"{session_name}.meta"

r0, c0 = 0, 24    # top- corner
H, W = 10, 10  

def read_exactly(n):
    

# --- Setup display ---
with open(bin_path, "ab") as bin_file:
    while True:
        if ser.read(1) != b'D':
            continue
        if ser.read(3) != b"ATA":
            continue

        FRAMES = struct.unpack('i', ser.read(4))[0]
        ROWS   = struct.unpack('i', ser.read(4))[0]
        COLS   = struct.unpack('i', ser.read(4))[0]

        nbytes = FRAMES * ROWS * COLS

        print("receiving bytes")
        payload = read_exactly[]