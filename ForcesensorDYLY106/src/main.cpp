#include <Arduino.h>
#include <Adafruit_NAU7802.h>

Adafruit_NAU7802 nau;
bool calibrated = false;

void runCalibration() {
  Serial.println("Starting calibration...");

  // Flush readings
  for (uint8_t i = 0; i < 10; i++) {
    while (!nau.available()) delay(1);
    nau.read();
  }

  while (!nau.calibrate(NAU7802_CALMOD_INTERNAL)) {
    Serial.println("Failed to calibrate internal offset, retrying!");
    delay(1000);
  }
  Serial.println("Calibrated internal offset");

  while (!nau.calibrate(NAU7802_CALMOD_OFFSET)) {
    Serial.println("Failed to calibrate system offset, retrying!");
    delay(1000);
  }
  Serial.println("Calibrated system offset");

  calibrated = true;
  Serial.println("Calibration finished");
}

void setup() {
  Serial.begin(115200);
  Serial.println("NAU7802");

  if (!nau.begin()) {
    Serial.println("Failed to find NAU7802");
    while (1) delay(10);
  }

  Serial.println("Found NAU7802");

  nau.setLDO(NAU7802_3V0);
  nau.setGain(NAU7802_GAIN_128);
  nau.setRate(NAU7802_RATE_10SPS);

  Serial.println("Press 'c' in Serial Monitor to calibrate.");
}

void loop() {

  // Check for calibration command
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'c' || cmd == 'C') {
      runCalibration();
    }
  }

  // Read sensor
  while (!nau.available()) delay(1);

  int32_t val = nau.read();

  float gram = val * 0.00306936771;
  float newton = gram * 9.81 / 1000.0;

  Serial.print("Read: ");
  Serial.println(val);

  Serial.print("Gram: ");
  Serial.println(gram);

  Serial.print("Newton: ");
  Serial.println(newton);

  delay(200);
}