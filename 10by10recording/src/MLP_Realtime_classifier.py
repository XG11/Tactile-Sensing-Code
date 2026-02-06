import serial, struct, numpy as np, time
from collections import deque

PORT = '/dev/cu.usbmodem160572101'
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=2)

print("Streaming started...")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tactile_mlp_model_trial_random_test.npz")

model = np.load(MODEL_PATH)

mean = model["mean"]
std  = model["std"]
W1   = model["W1"]
b1   = model["b1"]
W2   = model["W2"]
b2   = model["b2"]

FEATURES = mean.shape[0]

WINDOW = 100   # sliding window length
frame_buffer = deque(maxlen=WINDOW)

def read_exactly_number(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise RuntimeError("Serial timeout")
        buf.extend(chunk)
    return buf


def predict_window(x_window):

    X_test = (x_window - mean) / std

    Z1 = X_test @ W1.T + b1
    A1 = np.maximum(0, Z1)

    pooled = np.mean(A1, axis=0)

    Z2 = W2 @ pooled + b2

    expZ = np.exp(Z2 - np.max(Z2))
    probs = expZ / np.sum(expZ)

    pred = np.argmax(probs)

    return pred, probs


# --------------------
# REALTIME LOOP
# --------------------
while True:

    # Wait for header
    if ser.read(1) != b'D':
        continue
    if ser.read(3) != b"ATA":
        continue

    FRAMES = struct.unpack('<i', ser.read(4))[0]
    ROWS   = struct.unpack('<i', ser.read(4))[0]
    COLS   = struct.unpack('<i', ser.read(4))[0]

    nbytes = FRAMES * ROWS * COLS

    payload = read_exactly_number(nbytes)

    if read_exactly_number(4) != b"DONE":
        print("Footer mismatch")
        continue

    #print("Waiting for raw bytes...")

    raw = ser.read(20)
    #print("RAW BYTES:", raw)


    # --------------------
    # Process frames
    # --------------------
    data = np.frombuffer(payload, dtype=np.uint8)

    frames_full = data.reshape(FRAMES, ROWS, COLS)

# SAME ROI AS TRAINING
    r0, c0 = 0, 24
    H, W   = 10, 10

    frames_crop = frames_full[:, r0:r0+H, c0:c0+W]

    frames = frames_crop.reshape(FRAMES, H*W)

    for f in frames:

        if f.shape[0] != FEATURES:
            continue

        frame_buffer.append(f.astype(np.float32))

        if len(frame_buffer) == WINDOW:

            x_window = np.array(frame_buffer)

            pred, probs = predict_window(x_window)

            if pred == 0:
                label = "NO MOTION"
            elif pred == 1:
                label = "POKE"
            else:
                label = "SLIDE"

            print(label, np.round(probs, 3))
