#include <Arduino.h>

// put function declarations here:
const int outputPin = 9; // Use any suitable digital pin
void setup() {
  // put your setup code here, to run once:
  pinMode(outputPin, OUTPUT);
  //digitalWrite(outputPin, HIGH);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(outputPin, HIGH);
  delay(500); 
  digitalWrite(outputPin, LOW);
  delay(500);
}





