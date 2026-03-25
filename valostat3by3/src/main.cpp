
#include <Arduino.h>

// put function declarations here:
int analog_pin1 = 22;
int analog_pin2 = 21;
int analog_pin3 = 20;
int enable1 = 8;
int enable2 = 9;
int enable3 = 10;


void setup() {
  // put your setup code here, to run once:

  Serial.begin(115200);

  pinMode(analog_pin1, INPUT);
  pinMode(analog_pin2, INPUT);
  pinMode(analog_pin3, INPUT);
  pinMode(enable1, OUTPUT);
  pinMode(enable2, OUTPUT);
  pinMode(enable3, OUTPUT);
  digitalWriteFast(enable1, LOW);
  digitalWriteFast(enable2, LOW);
  digitalWriteFast(enable3, LOW);






}

void loop() {
  // put your main code here, to run repeatedly:

  int calibration11 = 48;
  int calibration12 = 53;
  int calibration13 = 87;
  int calibration21 = 57;
  int calibration22 = 41;
  int calibration23 = 45;
  int calibration31 = 69;
  int calibration32 = 52;
  int calibration33 = 44;

  digitalWriteFast(enable1, HIGH);
  int read11 = analogRead(analog_pin1) - calibration11;
  delay(5); 
  int read12 = analogRead(analog_pin2) - calibration12;
  delay(5);
  int read13 = analogRead(analog_pin3) - calibration13;
  delay(5);
  digitalWriteFast(enable1, LOW);

  digitalWriteFast(enable2, HIGH);
  int read21 = analogRead(analog_pin1) - calibration21;
  delay(5); 
  int read22 = analogRead(analog_pin2) - calibration22;
  delay(5);
  int read23 = analogRead(analog_pin3) - calibration23;
  delay(5);
  digitalWriteFast(enable2, LOW);

  digitalWriteFast(enable3, HIGH);
  int read31 = analogRead(analog_pin1) - calibration31;
  delay(5); 
  int read32 = analogRead(analog_pin2) - calibration32;
  delay(5);
  int read33 = analogRead(analog_pin3) - calibration33;
  delay(5);
  digitalWriteFast(enable3, LOW);

  Serial.println("RAW READING-----------------------------");
  Serial.print(read11); Serial.print("\t"); Serial.print(read12); Serial.print("\t"); Serial.println(read13);
  Serial.print(read21); Serial.print("\t"); Serial.print(read22); Serial.print("\t"); Serial.println(read23);
  Serial.print(read31); Serial.print("\t"); Serial.print(read32); Serial.print("\t"); Serial.println(read33);
}


