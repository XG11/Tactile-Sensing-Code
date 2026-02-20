import serial, struct, numpy as np, time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ---------------- SERIAL ----------------
PORT = '/dev/cu.usbmodem160572101'
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=2)

print("Streaming started...")

# ---------------- MODEL ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "3layerMLPparameter/3layermlp6_1000ep_torch_with_adam.pth")

T = 50
NODES = 100

# ----- Model Definition --
class TactileMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        # x: (B, T, 100)

        B, T, D = x.shape

        x = x.view(B * T, D)           # (B*T, 100)

        z1 = torch.relu(self.fc1(x))   # (B*T, 50)

        z1 = z1.view(B, T, 50)         # (B, T, 50)

        pooled = z1.mean(dim=1)        # (B, 50)

        z2 = torch.relu(self.fc2(pooled))  # (B, 64)
        z3 = torch.relu(self.fc3(z2))      # (B, 64)

        out = self.fc4(z3)                 # (B, 4)

        return out

# ----- Load model -----
device = torch.device("cpu")
model = TactileMLP().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------REALTIME

WINDOW = T
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

    x_tensor = torch.tensor(x_window, dtype=torch.float32)
    x_tensor = x_tensor.unsqueeze(0)  # (1, 100, 50)

    with torch.no_grad():
        outputs = model(x_tensor)
        probs = F.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1).item()

    return pred, probs.cpu().numpy()[0]

# ---------------- LOOP ----------------

while True:

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

    data = np.frombuffer(payload, dtype=np.uint8)
    frames_full = data.reshape(FRAMES, ROWS, COLS)

    r0, c0 = 0, 24
    Hc, Wc = 10, 10
    frames_crop = frames_full[:, r0:r0+Hc, c0:c0+Wc]
    frames = frames_crop.reshape(FRAMES, Hc*Wc)

    for f in frames:

        if f.shape[0] != NODES:
            continue

        frame_buffer.append(f.astype(np.float32))

        if len(frame_buffer) == WINDOW:

            x_window = np.array(frame_buffer)

            pred, probs = predict_window(x_window)

            if pred == 0:
                label = "NO MOTION"
            elif pred == 1:
                label = "POKE"
            elif pred == 2:
                label = "....slow slide....."
            else:
                label = "FAST SLIDE"

            print(label, np.round(probs, 3))
