import serial
import numpy as np
import matplotlib.pyplot as plt

PORT = '/dev/cu.usbmodem150585901'
BAUD = 115200
ROWS, COLS = 48, 48

ser = serial.Serial(PORT, BAUD)
frames = []
frame = []

while True:
    line = ser.readline().decode(errors='ignore').strip()
    if line == "done":
        break
    elif line == "flag":
        if len(frame) == ROWS:
            frames.append(np.array(frame, dtype=np.uint8))
        frame = []
    elif line:
        row = [int(x) for x in line.split(',')]
        frame.append(row)

frames = np.array(frames)
np.savez("captured_frames.npz", frames=frames)
print(f"Saved {frames.shape} to captured_frames.npz")

# Optional: visualize
plt.ion()
fig, ax = plt.subplots()
for i, f in enumerate(frames):
    ax.clear()
    ax.imshow(f, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"Frame {i+1}/{len(frames)}")
    plt.pause(0.05)
plt.ioff()
plt.show()
