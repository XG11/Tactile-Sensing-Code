

#include <Arduino.h>

// Simple 3x3 scan: activate rows 2,3,4 in sequence; read cols 5,6,7.


const uint8_t rowPins[3] = {2, 3, 4};
const uint8_t colPins[3] = {5, 6, 7};



// unit32_t counting = 0;

void setup() {
  Serial.begin(115200);

  for (int c = 0; c < 3; c++) {
    pinMode(colPins[c], INPUT_PULLDOWN);
  }


  for (int r = 0; r < 3; r++) {
    pinMode(rowPins[r], OUTPUT);
    digitalWrite(rowPins[r], LOW);
  }
}

void loop() {
  for (int r = 0; r < 3; r++) {
    // Turn this row ON
    digitalWriteFast(rowPins[r], HIGH);
    //delayNanoseconds(10);
    //delayMicroseconds(1); // tiny settle time
    delay(1);

    // Read columns in order 5,6,7 and print this row
    
    Serial.print('[');
    for (int c = 0; c < 3; c++) {
      //delay(1000);
      int pressed = digitalReadFast(colPins[c]); // LOW means pressed on active row
      Serial.print(pressed);
      if (c < 2) Serial.print(' ');
    }
    Serial.println(']'); 
    // Serial.println(counting++);
    
      

    // Turn this row OFF
    digitalWriteFast(rowPins[r], LOW);
    //delay(10);
  }

  Serial.println();   // blank line between frames
  //delayMicroseconds(100);         // slow down printing a bit
  delay(10);
}



/*
#include <Arduino.h>
// Simple one button demo, isolate all other buttons.
// This code select one row and on/off in a given frequency.


const uint8_t rowPin = 2;
const uint8_t colPin = 5;



// unit32_t counting = 0;

void setup() {
  Serial.begin(115200);


  pinMode(colPin, INPUT_PULLDOWN);


  pinMode(rowPin, OUTPUT);
  digitalWrite(rowPin, LOW);
}

void loop() {
  
  // Turn this row ON
  digitalWriteFast(rowPin, HIGH);
  delayMicroseconds(3); // tiny settle time
  int pressed = digitalReadFast(colPin == HIGH) ? 1 : 0; // LOW means pressed on active row
  Serial.print(pressed);
  digitalWriteFast(rowPin, LOW);
  Serial.println();   // blank line between frames
  delay(100);         // slow down printing a bit
}
*/

/*
#include <Arduino.h>

// Simple one button demo, isolate all other buttons.
// This code turn on one row constantly.


const uint8_t rowPin = 2;
const uint8_t colPin = 5;



// unit32_t counting = 0;

void setup() {
  Serial.begin(115200);


  pinMode(colPin, INPUT_PULLDOWN);


  pinMode(rowPin, OUTPUT);
  digitalWrite(rowPin, HIGH);
}

void loop() {
  int pressed = digitalReadFast(colPin); // LOW means pressed on active row
  Serial.print(pressed);
  Serial.println();   // blank line between frames
  delay(100);         // slow down printing a bit
}*/