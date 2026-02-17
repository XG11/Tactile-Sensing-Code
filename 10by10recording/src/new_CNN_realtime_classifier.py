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
MODEL_PATH = os.path.join(BASE_DIR, "tactile_cnn_05.pth")

T = 50
NODES = 100

# ----- Build adjacency-----
H = 10
W = 10

# ----- Model Definition --
class TemporalCNN(nn.Module):
    def __init__(self):
        super(TemporalCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=50, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # Kernel size 2, stride 2

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layer
        # Initial 10x10 -> pool1 (5x5) -> pool2 (2x2) -> pool3 (1x1)
        # So, the output will be 64 * 1 * 1 = 64 features.
        self.fc = nn.Linear(64, 3) # 3 classes: no motion, poke, slide

    def forward(self, x):
        B, T_seq, Nodes = x.shape # (batch_size, T, 100)
        H, W = 10, 10 # Assuming nodes = H * W, H and W are global variables

        # Reshape for CNN input: (B*T_seq, 1, H, W)
        x = x.view(B , T_seq, H, W)

        # First convolutional block
        x = self.pool1(F.relu(self.conv1(x))) # (B*T_seq, 16, 5, 5)

        # Second convolutional block
        x = self.pool2(F.relu(self.conv2(x))) # (B*T_seq, 32, 2, 2)

        # Third convolutional block
        x = self.pool3(F.relu(self.conv3(x))) # (B*T_seq, 64, 1, 1)

        # Flatten for the fully connected layer
        x = x.view(B, -1) # (B*T_seq, 64)

        # Fully connected layer
        logits = self.fc(x) # (B*T_seq, 3)

        # Reshape back to (B, T_seq, 3) and average over the sequence length
        #logits = logits.view(B, T_seq, 3)
        #logits = logits.mean(dim=1) # (B, 3)

        return logits

# ----- Load model -----
device = torch.device("cpu")
model = TemporalCNN().to(device)
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
            else:
                label = "SLIDE"

            print(label, np.round(probs, 3))
