'''
import serial, struct, numpy as np, matplotlib.pyplot as plt, matplotlib.animation as animation

PORT = '/dev/cu.usbmodem150585901'
BAUD = 115200
ser = serial.Serial(PORT, BAUD)

while True:
    if ser.read(4) == b'DATA':
        break

FRAMES = struct.unpack('i', ser.read(4))[0]
ROWS   = struct.unpack('i', ser.read(4))[0]
COLS   = struct.unpack('i', ser.read(4))[0]
print(f"Frames={FRAMES}, Rows={ROWS}, Cols={COLS}")

raw = ser.read(FRAMES * ROWS * COLS)
data = np.frombuffer(raw, dtype=np.uint8).reshape((FRAMES, ROWS, COLS))
assert ser.read(4) == b'DONE'

fig, ax = plt.subplots()
im = ax.imshow(data[0], cmap='gray', vmin=0, vmax=1)

def update(i):
    im.set_array(data[i])
    ax.set_title(f"Frame {i}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=50, blit=False)
plt.show() '''


#---------------------

import serial, struct, numpy as np, matplotlib.pyplot as plt, time

PORT = '/dev/cu.usbmodem150585901'
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=2)

# --- Setup display ---
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((48,48)), cmap='gray', vmin=0, vmax=1)
ax.set_title("Initializing...")
plt.ion()
plt.show()

while True:
    # --- Wait for new data block ---
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

    if footer != b'DONE' or len(raw) < nbytes:
        print("Incomplete block, skipping...")
        continue

    # --- Compute stats ---
    elapsed = t_end - t_start
    speed_kBps = nbytes / elapsed / 1024
    print(f"{FRAMES} frames ({nbytes/1024:.1f} KB) in {elapsed*1000:.1f} ms → {speed_kBps:.1f} KB/s")

    # --- Show only one representative frame ---
    data = np.frombuffer(raw, dtype=np.uint8).reshape((FRAMES, ROWS, COLS))
    frame = data[-1]   # show last frame of each block
    im.set_array(frame)
    ax.set_title(f"Latest block, Frame {FRAMES-1}")
    plt.pause(0.001)

    frame = data[-20]   # show last frame of each block
    im.set_array(frame)
    ax.set_title(f"Latest block, Frame {FRAMES-20}")
    plt.pause(0.001)

    frame = data[-40]   # show last frame of each block
    im.set_array(frame)
    ax.set_title(f"Latest block, Frame {FRAMES-40}")
    plt.pause(0.001)
