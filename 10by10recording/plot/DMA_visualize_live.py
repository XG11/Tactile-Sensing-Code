'''
import serial, struct, numpy as np, matplotlib.pyplot as plt, time

PORT = '/dev/cu.usbmodem160572101'   # adjust for your system
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=2)

# --- Setup display ---
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((48,48)), cmap='gray', vmin=0, vmax=1)
ax.set_title("Initializing...")
plt.ion()
plt.show()
while True:
    hdr = ser.read(4)
    if hdr != b'DATA':
        continue

    FRAMES = struct.unpack('i', ser.read(4))[0]
    ROWS   = struct.unpack('i', ser.read(4))[0]
    COLS   = struct.unpack('i', ser.read(4))[0]

    nbytes = FRAMES * ROWS * COLS
    t_start = time.time()
    raw = ser.read(nbytes)
    t_end = time.time()
    footer = ser.read(4)

    if len(raw) < nbytes:
        print("Incomplete block, skipping...")
        continue

    # --- Compute stats ---
    elapsed = t_end - t_start
    speed_kBps = nbytes / elapsed / 1024
    #print(f"{FRAMES} frames ({nbytes/1024:.1f} KB) in {elapsed*1000:.1f} ms → {speed_kBps:.1f} KB/s")

    # --- Convert to 3D array ---
    data = np.frombuffer(raw, dtype=np.uint8).reshape((FRAMES, ROWS, COLS))

    # --- Pick several evenly spaced frames for display ---
    # e.g., last 3 frames spaced by ~FRAMES/3 apart
    num_to_show = 5
    step = max(1, FRAMES // (num_to_show + 1))
    frame_indices = [FRAMES - (i+1)*step for i in range(num_to_show)]
    frame_indices = [i for i in frame_indices if i >= 0]

    for idx in frame_indices:
        frame = data[idx]
        im.set_array(frame)
        ax.set_title(f"Block frame {idx}/{FRAMES}")
        plt.pause(0.0001)

        '''

import serial, struct, numpy as np, matplotlib.pyplot as plt, time

PORT = '/dev/cu.usbmodem160572101' 
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=2)

r0, c0 = 0, 24    # top- corner
H, W = 10, 10  

# --- Setup display ---
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((H,W)), aspect = 1/1.52, cmap='gray', vmin=0, vmax=1)
ax.set_title("Initializing...")
plt.ion()
plt.show()
while True:
    hdr = ser.read(4)
    if hdr != b'DATA':
        continue

    FRAMES = struct.unpack('i', ser.read(4))[0]
    ROWS   = struct.unpack('i', ser.read(4))[0]
    COLS   = struct.unpack('i', ser.read(4))[0]

    nbytes = FRAMES * ROWS * COLS
    t_start = time.time()
    raw = ser.read(nbytes)
    t_end = time.time()
    footer = ser.read(4)

    if len(raw) < nbytes:
        print("Incomplete block, skipping...")
        continue

    # --- Compute stats ---
    #elapsed = t_end - t_start
    #speed_kBps = nbytes / elapsed / 1024
    #print(f"{FRAMES} frames ({nbytes/1024:.1f} KB) in {elapsed*1000:.1f} ms → {speed_kBps:.1f} KB/s")

    # --- Convert to 3D array ---
    data = np.frombuffer(raw, dtype=np.uint8).reshape((FRAMES, ROWS, COLS))

    # --- Pick several evenly spaced frames for display ---
    # e.g., last 3 frames spaced by ~FRAMES/3 apart
    num_to_show = 5
    step = max(1, FRAMES // (num_to_show + 1))
    frame_indices = [FRAMES - (i+1)*step for i in range(num_to_show)]
    frame_indices = [i for i in frame_indices if i >= 0]

    for idx in frame_indices:
        frame = data[idx]
        cropped = frame[r0:r0+10, c0:c0+10]
        im.set_array(cropped)
        ax.set_title(f"Block frame {idx}/{FRAMES}")
        plt.pause(0.0001)