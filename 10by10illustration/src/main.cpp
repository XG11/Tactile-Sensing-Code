#include <Arduino.h>
#include <vector>

// --- Multiplexer pin setup ---
const uint8_t muxEnPins[3] = {3, 4, 5};
const uint8_t muxSelPins[4] = {19, 18, 17, 16};
const uint8_t signalPin = 14;

const uint8_t READmuxEnPins[3] = {0, 1, 2};
const uint8_t READmuxSelPins[4] = {23, 22, 21, 20};
const uint8_t READsignalPin = 15;


const int ROWS = 48;
const int COLS = 48;
const int FRAMES = 20;
const int DEBOUNCE_FRAMES = 3;
std::vector<std::vector<int>> debounceCounter(ROWS, std::vector<int>(COLS, 0));
std::vector<std::vector<bool>> stableState(ROWS, std::vector<bool>(COLS, false));

std::vector<std::vector<std::vector<bool>>> matrix(
  FRAMES, std::vector<std::vector<bool>>(ROWS, std::vector<bool>(COLS, false))
);

// --- Helpers ---
void selectChannel(uint8_t channel) {
  for (uint8_t i = 0; i < 4; i++)
    digitalWriteFast(muxSelPins[i], (channel >> i) & 0x01);
}

void READselectChannel(uint8_t channel) {
  for (uint8_t i = 0; i < 4; i++)
    digitalWriteFast(READmuxSelPins[i], (channel >> i) & 0x01);
}

void enableMux(uint8_t muxIndex) {
  for (uint8_t i = 0; i < 3; i++) digitalWriteFast(muxEnPins[i], HIGH);
  digitalWriteFast(muxEnPins[muxIndex], LOW);
}

void READenableMux(uint8_t rmuxIndex) {
  for (uint8_t i = 0; i < 3; i++) digitalWriteFast(READmuxEnPins[i], HIGH);
  digitalWriteFast(READmuxEnPins[rmuxIndex], LOW);
}


void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < 3; i++) {
    pinMode(muxEnPins[i], OUTPUT);
    digitalWriteFast(muxEnPins[i], HIGH);
    pinMode(READmuxEnPins[i], OUTPUT);
    digitalWriteFast(READmuxEnPins[i], HIGH);
  }
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(muxSelPins[i], OUTPUT);
    pinMode(READmuxSelPins[i], OUTPUT);
    digitalWriteFast(muxSelPins[i], LOW);
    digitalWriteFast(READmuxSelPins[i], LOW);
  }

  pinMode(signalPin, OUTPUT);
  digitalWriteFast(signalPin, HIGH);
  pinMode(READsignalPin, INPUT);

  Serial.println("Ready...");
}

void sendMatrixBinary() {
  while (!Serial) delay(10);
  Serial.flush();

  // Header
  Serial.write("DATA", 4);
  Serial.write((uint8_t*)&FRAMES, sizeof(FRAMES));
  Serial.write((uint8_t*)&ROWS, sizeof(ROWS));
  Serial.write((uint8_t*)&COLS, sizeof(COLS));

  // --- Optimized buffered sending ---
  uint8_t buffer[256];
  int idx = 0;

  for (int f = 0; f < FRAMES; f++) {
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        buffer[idx++] = matrix[f][r][c] ? 1 : 0;
        if (idx >= sizeof(buffer)) {
          Serial.write(buffer, idx);
          idx = 0;
        }
      }
    }
  }

  // Send remaining bytes
  if (idx > 0) Serial.write(buffer, idx);

  // Footer
  Serial.write("DONE", 4);
  Serial.send_now();  // force USB flush immediately
}

void loop() {
  while (1) {
    // --- Capture FRAMES frames ---
    for (int f = 0; f < FRAMES; f++) {
      int rowIndex = 0;
      for (uint8_t mux = 0; mux < 3; mux++) {
        enableMux(mux);
        for (uint8_t ch = 0; ch < 16; ch++) {
          selectChannel(ch);
          delayMicroseconds(1);

          int colIndex = 0;
          for (uint8_t rmux = 0; rmux < 3; rmux++) {
            READenableMux(rmux);
            for (uint8_t rch = 0; rch < 16; rch++) {
              READselectChannel(rch);
              delayMicroseconds(1);
              matrix[f][rowIndex][colIndex] = digitalReadFast(READsignalPin);
              colIndex++;
            }
          }
          rowIndex++;
        }
      }
    }

    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        bool current = matrix[FRAMES - 1][r][c];
        if (current == stableState[r][c]) {
          debounceCounter[r][c] = 0;
        } else {
          debounceCounter[r][c]++;
          if (debounceCounter[r][c] >= DEBOUNCE_FRAMES) {
            stableState[r][c] = current;
            debounceCounter[r][c] = 0;
          }
        }
      }
    }

    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        matrix[FRAMES - 1][r][c] = stableState[r][c];
      }
    }

    sendMatrixBinary();

  }
}