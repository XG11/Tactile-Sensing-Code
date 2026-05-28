import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

# =========================
# Load WAV file
# =========================
wav_path = "/Users/xiaoruigu/PlatformIO/Projects/apple/session_20260528_084607/audio.wav"   # Change to your file

sample_rate, audio = wavfile.read(wav_path)

# Convert stereo to mono if needed
if audio.ndim > 1:
    audio = audio[:, 0]

# Normalize audio
audio = audio.astype(np.float32)
audio = audio / np.max(np.abs(audio))

# =========================
# Time-domain waveform
# =========================
duration = len(audio) / sample_rate
time = np.linspace(0, duration, len(audio))

# =========================
# Frequency spectrum (FFT)
# =========================
N = len(audio)

yf = fft(audio)
xf = fftfreq(N, 1 / sample_rate)

# Keep only positive frequencies
positive_idx = xf >= 0
xf = xf[positive_idx]
yf = np.abs(yf[positive_idx])

# =========================
# Plot
# =========================
plt.figure(figsize=(14, 8))

# ---- Waveform ----
plt.subplot(2, 1, 1)
plt.plot(time, audio)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# ---- Spectrum ----
plt.subplot(2, 1, 2)
plt.plot(xf, yf)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, sample_rate / 2)
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================
# Info
# =========================
print(f"Sample Rate : {sample_rate} Hz")
print(f"Duration    : {duration:.2f} s")
print(f"Total Samples: {N}")