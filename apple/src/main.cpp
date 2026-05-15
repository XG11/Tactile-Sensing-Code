#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>

#define BNO08X_RESET -1

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

float aX, aY, aZ;
float gX, gY, gZ;

unsigned long lastPrintTime = 0;

void setReports() {
  Serial.println("Setting BNO08x reports...");

  if (!bno08x.enableReport(SH2_ACCELEROMETER, 10000)) {
    Serial.println("Could not enable accelerometer");
  }

  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000)) {
    Serial.println("Could not enable gyroscope");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("Starting BNO08x I2C test...");

  Wire.begin();
  delay(100);

  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("Could not find BNO08x at 0x4A, trying 0x4B...");

    if (!bno08x.begin_I2C(0x4B, &Wire)) {
      Serial.println("Failed to find BNO08x at 0x4A or 0x4B.");
      while (1) delay(10);
    }
  }

  Serial.println("BNO08x found!");

  setReports();

  Serial.println("Accel X\tAccel Y\tAccel Z\t|\tGyro X\tGyro Y\tGyro Z");
  Serial.println("------------------------------------------------------------");
}

void loop() {
  if (bno08x.wasReset()) {
    Serial.println("BNO08x reset detected");
    setReports();
  }

  if (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ACCELEROMETER:
        aX = sensorValue.un.accelerometer.x;
        aY = sensorValue.un.accelerometer.y;
        aZ = sensorValue.un.accelerometer.z;
        break;

      case SH2_GYROSCOPE_CALIBRATED:
        gX = sensorValue.un.gyroscope.x;
        gY = sensorValue.un.gyroscope.y;
        gZ = sensorValue.un.gyroscope.z;
        break;
    }
  }

  if (millis() - lastPrintTime >= 100) {
    lastPrintTime = millis();

    Serial.print(aX, 2); Serial.print("\t");
    Serial.print(aY, 2); Serial.print("\t");
    Serial.print(aZ, 2); Serial.print("\t|\t");

    Serial.print(gX, 2); Serial.print("\t");
    Serial.print(gY, 2); Serial.print("\t");
    Serial.println(gZ, 2);
  }
}