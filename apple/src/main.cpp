#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <Audio.h>
#include <Protocentral_ADS1220.h>
#include <Adafruit_BNO08x.h>

// ================= LOAD CELL ADS1220 =================
#define ADS1220_CS_PIN    10
#define ADS1220_DRDY_PIN  9

Protocentral_ADS1220 pc_ads1220;
int32_t loadCellRaw = 0;

// ================= BNO08x =================
#define BNO08X_RESET -1

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

float ax = 0, ay = 0, az = 0;
float gx = 0, gy = 0, gz = 0;

// ================= I2S MIC -> USB AUDIO =================
// Teensy 4.0 I2S mic pins:
// BCLK  = pin 21
// LRCLK = pin 20
// DIN   = pin 8
// 3V3   = 3.3V
// GND   = GND

AudioInputI2S        i2sMic;
AudioOutputUSB       usbAudio;

// Send mic to both left and right USB audio channels
AudioConnection patchCord1(i2sMic, 0, usbAudio, 0);
AudioConnection patchCord2(i2sMic, 0, usbAudio, 1);

// ================= SAMPLE RATE =================
const unsigned long SAMPLE_INTERVAL_MS = 50; // 20 Hz sensor CSV
unsigned long lastSampleTime = 0;

void setupBNOReports() {
  long reportIntervalUs = 50000; // 20 Hz

  if (!bno08x.enableReport(SH2_ACCELEROMETER, reportIntervalUs)) {
    Serial.println("Could not enable accelerometer");
  }

  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, reportIntervalUs)) {
    Serial.println("Could not enable gyroscope");
  }
}

void updateIMU() {
  if (bno08x.wasReset()) {
    setupBNOReports();
  }

  while (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ACCELEROMETER:
        ax = sensorValue.un.accelerometer.x;
        ay = sensorValue.un.accelerometer.y;
        az = sensorValue.un.accelerometer.z;
        break;

      case SH2_GYROSCOPE_CALIBRATED:
        gx = sensorValue.un.gyroscope.x;
        gy = sensorValue.un.gyroscope.y;
        gz = sensorValue.un.gyroscope.z;
        break;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // Audio memory for I2S mic -> USB audio
  AudioMemory(16);

  // ---------- BNO08x I2C ----------
  Wire.begin();
  delay(100);

  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    if (!bno08x.begin_I2C(0x4B, &Wire)) {
      Serial.println("Failed to find BNO08x");
      while (1) delay(10);
    }
  }

  setupBNOReports();

  // ---------- ADS1220 SPI ----------
  SPI.begin();

  pc_ads1220.begin(ADS1220_CS_PIN, ADS1220_DRDY_PIN);

  pc_ads1220.set_pga_gain(PGA_GAIN_128);
  pc_ads1220.set_data_rate(DR_20SPS);
  pc_ads1220.set_FIR_Filter(FIR_5060);
  pc_ads1220.set_conv_mode_continuous();

  Serial.println("time_us,ax,ay,az,gx,gy,gz,load_raw");
}

void loop() {
  updateIMU();

  if (digitalRead(ADS1220_DRDY_PIN) == LOW) {
    loadCellRaw = pc_ads1220.Read_WaitForData();
  }

  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;

    unsigned long t = micros();

    Serial.print(t);
    Serial.print(",");

    Serial.print(ax, 6); Serial.print(",");
    Serial.print(ay, 6); Serial.print(",");
    Serial.print(az, 6); Serial.print(",");

    Serial.print(gx, 6); Serial.print(",");
    Serial.print(gy, 6); Serial.print(",");
    Serial.print(gz, 6); Serial.print(",");

    Serial.println(loadCellRaw);
  }
}