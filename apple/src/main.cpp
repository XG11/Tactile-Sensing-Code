#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <Audio.h>
#include <Protocentral_ADS1220.h>
#include <Adafruit_BNO08x.h>

// ================= LOAD CELL ADS1220 =================
#define ADS1220_CS_PIN    6
#define ADS1220_DRDY_PIN  5

Protocentral_ADS1220 pc_ads1220;
int32_t loadCellRaw = 0;

// ================= BNO08x SPI =================
#define BNO08X_CS     10
#define BNO08X_INT    9
#define BNO08X_RESET  4

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

float ax = 0, ay = 0, az = 0;
float gx = 0, gy = 0, gz = 0;

// ================= I2S MIC -> USB AUDIO =================
// BCLK  = pin 21
// LRCLK = pin 20
// DIN   = pin 7
// 3V3   = 3.3V
// GND   = GND

AudioInputI2S        i2sMic;
AudioOutputUSB       usbAudio;

AudioConnection patchCord1(i2sMic, 0, usbAudio, 0);
AudioConnection patchCord2(i2sMic, 0, usbAudio, 1);

// ================= SAMPLE RATE =================
const unsigned long SAMPLE_INTERVAL_US = 2500; //
// 5000 -> 200Hz
unsigned long lastSampleTime = 0;

void setupBNOReports() {
  long acc_reportIntervalUs = 2500; //
  long gyro_reportIntervalUs = 2500; 

  if (!bno08x.enableReport(SH2_ACCELEROMETER, acc_reportIntervalUs)) {
    Serial.println("Could not enable accelerometer");
  }

  delay(20);

  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, gyro_reportIntervalUs)) {
    Serial.println("Could not enable gyroscope");
  }

  delay(20);
}

void updateIMU() {
  if (bno08x.wasReset()) {
    delay(50);
    setupBNOReports();
  }

  int count = 0;

  if (bno08x.getSensorEvent(&sensorValue)) {
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
  Serial.begin(2000000);
  while (!Serial) delay(10);

  AudioMemory(16);

  // ---------- SPI ----------
  pinMode(BNO08X_CS, OUTPUT);
  digitalWrite(BNO08X_CS, HIGH);

  pinMode(ADS1220_CS_PIN, OUTPUT);
  digitalWrite(ADS1220_CS_PIN, HIGH);

  SPI.begin();
  delay(200);

  // ---------- BNO08x SPI ----------
  delay(200);
  // According to datasheet, acc sample report up to 400hz and gyro up to 500hz
  if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
    Serial.println("Failed to find BNO08x over SPI");
    while (1) delay(10);
  }

  delay(100);
  setupBNOReports();

  // ---------- ADS1220 SPI ----------
  pc_ads1220.begin(ADS1220_CS_PIN, ADS1220_DRDY_PIN);

  pc_ads1220.set_pga_gain(PGA_GAIN_128);
  pc_ads1220.set_data_rate(DR_1000SPS);   // 
  pc_ads1220.set_conv_mode_continuous();

  Serial.println("time_us,ax,ay,az,gx,gy,gz,load_raw");
}

void loop() {
  updateIMU();

  if (digitalRead(ADS1220_DRDY_PIN) == LOW) {
    loadCellRaw = pc_ads1220.Read_WaitForData();
  }

  unsigned long now = micros();

  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime += SAMPLE_INTERVAL_US;

    static unsigned long prevPrintTime = 0;

    unsigned long dt = now - prevPrintTime;
    prevPrintTime = now;

    Serial.print(now);
    Serial.print(",");
    //Serial.print(dt);
    //Serial.print(",");

    Serial.print(ax, 3); Serial.print(",");
    Serial.print(ay, 3); Serial.print(",");
    Serial.print(az, 3); Serial.print(",");

    Serial.print(gx, 3); Serial.print(",");
    Serial.print(gy, 3); Serial.print(",");
    Serial.print(gz, 3); Serial.print(",");

    Serial.println(loadCellRaw);
  }
}