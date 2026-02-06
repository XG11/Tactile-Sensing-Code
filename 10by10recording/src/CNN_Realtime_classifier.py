

import serial, struct, numpy as np, time
from collections import deque

PORT = '/dev/cu.usbmodem160572101'
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=2)

print("Streaming started...")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tactile_cnn_model_trial_03.npz")

model = np.load(MODEL_PATH)

conv_weight = model["conv_weight"]  # (1,1,3,3)
conv_bias   = model["conv_bias"]    # (1,)
fc_weight   = model["fc_weight"]    # (3,16)
fc_bias     = model["fc_bias"]      # (3,)



FEATURES = 10 * 10

WINDOW = 50   # sliding window length
frame_buffer = deque(maxlen=WINDOW)

def conv2d_numpy(x, weight, bias):
    # x shape: (10,10)
    # weight shape: (1,1,3,3)

    kernel = weight[0,0]   # (3,3)
    bias = bias[0]

    H, W = x.shape
    F = kernel.shape[0]

    out_h = H - F + 1
    out_w = W - F + 1

    out = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            region = x[i:i+F, j:j+F]
            out[i,j] = np.sum(region * kernel) + bias

    return out


def relu(x):
    return np.maximum(0, x)


def maxpool2x2(x):

    H, W = x.shape
    out = np.zeros((H//2, W//2), dtype=np.float32)

    for i in range(0, H, 2):
        for j in range(0, W, 2):

            block = x[i:i+2, j:j+2]
            out[i//2, j//2] = np.max(block)

    return out


def softmax(z):

    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)



def predict_window_cnn(x_window):

    logits_all = []

    for frame in x_window:

        img = frame.reshape(10,10).astype(np.float32)

        # CNN forward (feature + logit extraction)
        z1 = conv2d_numpy(img, conv_weight, conv_bias)
        a1 = np.maximum(0, z1)
        p1 = maxpool2x2(a1)
        flat = p1.flatten()

        z2 = fc_weight @ flat + fc_bias   # LOGITS, shape (3,)
        logits_all.append(z2)

    logits_all = np.array(logits_all)     # (T,3)

    mean_logits = np.mean(logits_all, axis=0)
    mean_probs = softmax(mean_logits)

    pred = np.argmax(mean_probs)

    return pred, mean_probs



def read_exactly_number(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise RuntimeError("Serial timeout")
        buf.extend(chunk)
    return buf


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

            pred, probs = predict_window_cnn(x_window)

            if pred == 0:
                label = "NO MOTION"
            elif pred == 1:
                label = "POKE"
            else:
                label = "SLIDE"

            print(label, np.round(probs, 3))
