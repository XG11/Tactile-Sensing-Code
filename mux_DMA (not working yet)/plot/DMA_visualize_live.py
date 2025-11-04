import serial
import numpy as np
import matplotlib.pyplot as plt

PORT = '/dev/cu.usbmodem150585901'
BAUD = 115200
ROWS, COLS = 48, 48

ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"Connected to {PORT}")

plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((ROWS, COLS)), cmap='gray_r', vmin=0, vmax=1)
#ax.set_title("48×48 tactile array")
plt.show(block=False)

frame = []
while True:
    try:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue

        if line == "flag":
            if len(frame) == ROWS:
                arr = np.array(frame, dtype=np.uint8)
                im.set_data(arr)
                plt.pause(0.001)
            frame = []

        else:
            try:
                row = [int(x) for x in line.split(',')]
                if len(row) == COLS:
                    frame.append(row)
            except ValueError:
                continue

    except KeyboardInterrupt:
        print("\nStopped by user.")
        break

ser.close()
plt.ioff()
plt.show()
