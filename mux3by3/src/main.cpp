
#include <Arduino.h>

//Pin
const uint8_t muxEnPins[3] = {5, 6, 7};   // EN pins for 3 muxes
const uint8_t muxSelPins[4] = {11, 10, 9, 8}; // S0..S3 shared select lines
const uint8_t signalPin = 3;             // Output pin to send HIGH/LOW signal

const uint8_t READmuxEnPins[3] = {16, 17, 18};   // EN pins for 3 muxes
const uint8_t READmuxSelPins[4] = {19, 20, 21, 22}; // S0..S3 shared select lines
const uint8_t READsignalPin = 14;             // Iutput pin to READ HIGH/LOW signal

void selectChannel(uint8_t channel) {
  // S0 = least significant bit (bit 0)
  digitalWrite(muxSelPins[0], channel & 0x01);

  // S1 = bit 1 (shift right by 1, then mask)
  digitalWrite(muxSelPins[1], (channel >> 1) & 0x01);

  // S2 = bit 2
  digitalWrite(muxSelPins[2], (channel >> 2) & 0x01);

  // S3 = most significant bit (bit 3)
  digitalWrite(muxSelPins[3], (channel >> 3) & 0x01);
}

void READselectChannel(uint8_t channel) {
  // S0 = least significant bit (bit 0)
  digitalWrite(READmuxSelPins[0], channel & 0x01);

  // S1 = bit 1 (shift right by 1, then mask)
  digitalWrite(READmuxSelPins[1], (channel >> 1) & 0x01);

  // S2 = bit 2
  digitalWrite(READmuxSelPins[2], (channel >> 2) & 0x01);

  // S3 = most significant bit (bit 3)
  digitalWrite(READmuxSelPins[3], (channel >> 3) & 0x01);
}


void enableMux(uint8_t muxIndex) {
  // Disable all mux first
  for (uint8_t i = 0; i < 3; i++) {
    digitalWrite(muxEnPins[i], HIGH); // EN high = disabled
  }
  digitalWriteFast(muxEnPins[muxIndex], LOW); // EN low = enable selected mux
}

void READenableMux(uint8_t muxIndex) {
  // Disable all mux first
  for (uint8_t i = 0; i < 3; i++) {
    digitalWrite(READmuxEnPins[i], HIGH); // EN high = disabled
  }
  digitalWriteFast(READmuxEnPins[muxIndex], LOW); // EN low = enable selected mux
}

void setup() {
  Serial.begin(115200);

  // Setup enable pins
  for (uint8_t i = 0; i < 3; i++) {
    pinMode(muxEnPins[i], OUTPUT);
    digitalWriteFast(muxEnPins[i], HIGH);
  }
  // Setup READ enable pins
  for (uint8_t i = 0; i < 3; i++) {
    pinMode(READmuxEnPins[i], OUTPUT);
    digitalWriteFast(READmuxEnPins[i], HIGH);
  }


  // Setup select pins
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(muxSelPins[i], OUTPUT);
    digitalWriteFast(muxSelPins[i], LOW);
  }

  // Setup READ select pins
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(READmuxSelPins[i], OUTPUT);
    digitalWriteFast(READmuxSelPins[i], LOW);
  }

  // Setup signal pin
  pinMode(signalPin, OUTPUT);
  //digitalWriteFast(signalPin, LOW);
  digitalWriteFast(signalPin, HIGH);

  // Setup READ signal pin
  pinMode(READsignalPin, INPUT);

}

void loop() {
  Serial.println("[");  // start of matrix

  for (uint8_t mux = 0; mux < 3; mux++) {           // 3 write muxes -> 48 rows
    enableMux(mux);
    delayMicroseconds(5);

    for (uint8_t ch = 0; ch < 16; ch++) {          // 16 channels per mux
      selectChannel(ch);
      delayMicroseconds(5); 

      for (uint8_t rmux = 0; rmux < 3; rmux++) {   // 3 read muxes -> 48 columns
        READenableMux(rmux);
        delayMicroseconds(5);
        for (uint8_t rch = 0; rch < 16; rch++) {
          READselectChannel(rch);
          delayMicroseconds(5);
          Serial.print(digitalReadFast(READsignalPin));
          Serial.print(' '); // space between columns
        }
      }

      Serial.println(); // move to next row
    }
  }

  Serial.println("]");  // end of matrix
  Serial.println();     // blank line between frames
  delay(1);
}




/*
#include <Arduino.h>

//Pin
const uint8_t muxSelPins[4] = {11, 10, 9, 8}; // S0..S3 shared select lines
const uint8_t signalPin = 14;             // Output pin to send HIGH/LOW signal

void selectChannel(uint8_t channel) {
  // S0 = least significant bit (bit 0)
  digitalWriteFast(muxSelPins[0], channel & 0x01);

  // S1 = bit 1 (shift right by 1, then mask)
  digitalWriteFast(muxSelPins[1], (channel >> 1) & 0x01);

  // S2 = bit 2
  digitalWriteFast(muxSelPins[2], (channel >> 2) & 0x01);

  // S3 = most significant bit (bit 3)
  digitalWriteFast(muxSelPins[3], (channel >> 3) & 0x01);
}


void setup() {
  Serial.begin(115200);

  // Setup enable pins
  pinMode(7, OUTPUT);
  digitalWrite(7, HIGH);
  pinMode(6, OUTPUT);
  digitalWrite(6, HIGH);
  pinMode(5, OUTPUT);
  digitalWrite(5, LOW);

  // Setup select pins
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(muxSelPins[i], OUTPUT);
    digitalWrite(muxSelPins[i], LOW);
  }

  // Setup signal pin
  pinMode(signalPin, OUTPUT);
  //digitalWriteFast(signalPin, LOW);
  digitalWrite(signalPin, HIGH);
}

void loop() {
  for (uint8_t ch = 0; ch < 16; ch++) {
    //selectChannel(ch);
    delay(10);
  }
}

*/

/*
#include <Arduino.h>

const uint8_t signalPin = 14;             // Output pin to send HIGH/LOW signal



void setup() {
  Serial.begin(115200);

  // Setup enable pins
  pinMode(7, OUTPUT);
  digitalWriteFast(7, HIGH);
  pinMode(6, OUTPUT);
  digitalWriteFast(6, HIGH);
  pinMode(5, OUTPUT);
  digitalWriteFast(5, HIGH);

  // Setup signal pin
  pinMode(signalPin, OUTPUT);
  //digitalWriteFast(signalPin, LOW);
  digitalWriteFast(signalPin, HIGH);
}

void loop() {
  
}
*/

/*
#include <Arduino.h>

//Pin
//const uint8_t muxEnPins[3] = {5, 6, 7};   // EN pins for 3 muxes
//const uint8_t muxSelPins[4] = {11, 10, 9, 8}; // S0..S3 shared select lines
//const uint8_t signalPin = 3;             // Output pin to send HIGH/LOW signal

const uint8_t muxEnPins[3] = {16, 17, 18};   // EN pins for 3 muxes
const uint8_t muxSelPins[4] = {19, 20, 21, 22}; // S0..S3 shared select lines
const uint8_t signalPin = 14;             // Output pin to send HIGH/LOW signal


void enableMux(uint8_t muxIndex) {
  // Disable all mux first
  for (uint8_t i = 0; i < 3; i++) {
    digitalWrite(muxEnPins[i], HIGH); // EN high = disabled
  }
  digitalWriteFast(muxEnPins[muxIndex], LOW); // EN low = enable selected mux
}

void setup() {
  Serial.begin(115200);

  // Setup enable pins
  for (uint8_t i = 0; i < 3; i++) {
    pinMode(muxEnPins[i], OUTPUT);
    digitalWriteFast(muxEnPins[i], HIGH);
  }

  // Setup select pins
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(muxSelPins[i], OUTPUT);
    digitalWriteFast(muxSelPins[i], LOW);
  }

  // Setup signal pin
  pinMode(signalPin, OUTPUT);
  //digitalWriteFast(signalPin, LOW);
  digitalWriteFast(signalPin, HIGH);
}

void loop() {
  for (uint8_t mux = 0; mux < 3; mux++) {
    enableMux(mux);
      // --- Your output signal ---
    digitalWriteFast(signalPin, HIGH);  // Turn on
      //0
    digitalWriteFast(muxSelPins[0], LOW);
    digitalWriteFast(muxSelPins[1], LOW);
    digitalWriteFast(muxSelPins[2], LOW);
    digitalWriteFast(muxSelPins[3], LOW);
      //delayMicroseconds(1000);
    delay(10);
    //digitalWriteFast(signalPin, LOW);   // Turn off
    //delayMicroseconds(1000);
    //1
    digitalWriteFast(muxSelPins[0], HIGH);
    digitalWriteFast(muxSelPins[1], LOW);
    digitalWriteFast(muxSelPins[2], LOW);
    digitalWriteFast(muxSelPins[3], LOW);
      //delayMicroseconds(1000);
    delay(10);
      //2
    digitalWriteFast(muxSelPins[0], LOW);
    digitalWriteFast(muxSelPins[1], HIGH);
    digitalWriteFast(muxSelPins[2], LOW);
    digitalWriteFast(muxSelPins[3], LOW);
      //delayMicroseconds(1000);
    delay(10);
      //3
    digitalWriteFast(muxSelPins[0], HIGH);
    digitalWriteFast(muxSelPins[1], HIGH);
    digitalWriteFast(muxSelPins[2], LOW);
    digitalWriteFast(muxSelPins[3], LOW);
      //delayMicroseconds(1000);
    delay(10);
      //4
    digitalWriteFast(muxSelPins[0], LOW);
    digitalWriteFast(muxSelPins[1], LOW);
    digitalWriteFast(muxSelPins[2], HIGH);
    digitalWriteFast(muxSelPins[3], LOW);
      //delayMicroseconds(1000);
    delay(10);
      //5
    digitalWriteFast(muxSelPins[0], HIGH);
    digitalWriteFast(muxSelPins[1], LOW);
    digitalWriteFast(muxSelPins[2], HIGH);
    digitalWriteFast(muxSelPins[3], LOW);
      //delayMicroseconds(1000);
    delay(10);
      //6
    digitalWriteFast(muxSelPins[0], LOW);
    digitalWriteFast(muxSelPins[1], HIGH);
    digitalWriteFast(muxSelPins[2], HIGH);
    digitalWriteFast(muxSelPins[3], LOW);
      //delayMicroseconds(1000);
    delay(10);
      //7
    digitalWriteFast(muxSelPins[0], HIGH);
    digitalWriteFast(muxSelPins[1], HIGH);
    digitalWriteFast(muxSelPins[2], HIGH);
    digitalWriteFast(muxSelPins[3], LOW);
      //delayMicroseconds(1000);
    delay(10);
      //8
    digitalWriteFast(muxSelPins[0], LOW);
    digitalWriteFast(muxSelPins[1], LOW);
    digitalWriteFast(muxSelPins[2], LOW);
    digitalWriteFast(muxSelPins[3], HIGH);
      //delayMicroseconds(1000);
    delay(10);
      //9
    digitalWriteFast(muxSelPins[0], HIGH);
    digitalWriteFast(muxSelPins[1], LOW);
    digitalWriteFast(muxSelPins[2], LOW);
    digitalWriteFast(muxSelPins[3], HIGH);
      //delayMicroseconds(1000);
    delay(10);
      //10
    digitalWriteFast(muxSelPins[0], LOW);
    digitalWriteFast(muxSelPins[1], HIGH);
    digitalWriteFast(muxSelPins[2], LOW);
    digitalWriteFast(muxSelPins[3], HIGH);
      //delayMicroseconds(1000);
    delay(10);
      //11
    digitalWriteFast(muxSelPins[0], HIGH);
    digitalWriteFast(muxSelPins[1], HIGH);
    digitalWriteFast(muxSelPins[2], LOW);
    digitalWriteFast(muxSelPins[3], HIGH);
      //delayMicroseconds(1000);
    delay(10);
      //12
    digitalWriteFast(muxSelPins[0], LOW);
    digitalWriteFast(muxSelPins[1], LOW);
    digitalWriteFast(muxSelPins[2], HIGH);
    digitalWriteFast(muxSelPins[3], HIGH);
      //delayMicroseconds(1000);
    delay(10);
      //13
    digitalWriteFast(muxSelPins[0], HIGH);
    digitalWriteFast(muxSelPins[1], LOW);
    digitalWriteFast(muxSelPins[2], HIGH);
    digitalWriteFast(muxSelPins[3], HIGH);
      //delayMicroseconds(1000);
    delay(10);
      //14
    digitalWriteFast(muxSelPins[0], LOW);
    digitalWriteFast(muxSelPins[1], HIGH);
    digitalWriteFast(muxSelPins[2], HIGH);
    digitalWriteFast(muxSelPins[3], HIGH);
      //delayMicroseconds(1000);
    delay(10);
      //15
    digitalWriteFast(muxSelPins[0], HIGH);
    digitalWriteFast(muxSelPins[1], HIGH);
    digitalWriteFast(muxSelPins[2], HIGH);
    digitalWriteFast(muxSelPins[3], HIGH);
      //delayMicroseconds(1000);
    delay(10);

    //digitalWriteFast(signalPin, LOW);

    //delay(10);
    }
  }
*/


/*
#include <Arduino.h>

//Pin
const uint8_t muxEnPins[3] = {5, 6, 7};   // EN pins for 3 muxes
const uint8_t muxSelPins[4] = {11, 10, 9, 8}; // S0..S3 shared select lines
const uint8_t signalPin = 3;             // Output pin to send HIGH/LOW signal

const uint8_t READmuxEnPins[3] = {16, 17, 18};   // EN pins for 3 muxes
const uint8_t READmuxSelPins[4] = {19, 20, 21, 22}; // S0..S3 shared select lines
const uint8_t READsignalPin = 14;             // Iutput pin to READ HIGH/LOW signal

void selectChannel(uint8_t channel) {
  // S0 = least significant bit (bit 0)
  digitalWrite(muxSelPins[0], channel & 0x01);

  // S1 = bit 1 (shift right by 1, then mask)
  digitalWrite(muxSelPins[1], (channel >> 1) & 0x01);

  // S2 = bit 2
  digitalWrite(muxSelPins[2], (channel >> 2) & 0x01);

  // S3 = most significant bit (bit 3)
  digitalWrite(muxSelPins[3], (channel >> 3) & 0x01);
}

void READselectChannel(uint8_t channel) {
  // S0 = least significant bit (bit 0)
  digitalWrite(READmuxSelPins[0], channel & 0x01);

  // S1 = bit 1 (shift right by 1, then mask)
  digitalWrite(READmuxSelPins[1], (channel >> 1) & 0x01);

  // S2 = bit 2
  digitalWrite(READmuxSelPins[2], (channel >> 2) & 0x01);

  // S3 = most significant bit (bit 3)
  digitalWrite(READmuxSelPins[3], (channel >> 3) & 0x01);
}


void enableMux(uint8_t muxIndex) {
  // Disable all mux first
  for (uint8_t i = 0; i < 3; i++) {
    digitalWrite(muxEnPins[i], HIGH); // EN high = disabled
  }
  digitalWriteFast(muxEnPins[muxIndex], LOW); // EN low = enable selected mux
}

void READenableMux(uint8_t muxIndex) {
  // Disable all mux first
  for (uint8_t i = 0; i < 3; i++) {
    digitalWrite(READmuxEnPins[i], HIGH); // EN high = disabled
  }
  digitalWriteFast(READmuxEnPins[muxIndex], LOW); // EN low = enable selected mux
}

void setup() {
  Serial.begin(115200);


  pinMode(muxEnPins[1], OUTPUT);
  digitalWriteFast(muxEnPins[1], LOW);

  pinMode(muxEnPins[0], OUTPUT);
  digitalWriteFast(muxEnPins[0], HIGH);

  pinMode(muxEnPins[2], OUTPUT);
  digitalWriteFast(muxEnPins[2], HIGH);


  pinMode(READmuxEnPins[1], OUTPUT);
  digitalWriteFast(READmuxEnPins[1], LOW);
  
  pinMode(READmuxEnPins[0], OUTPUT);
  digitalWriteFast(READmuxEnPins[0], HIGH);

  pinMode(READmuxEnPins[2], OUTPUT);
  digitalWriteFast(READmuxEnPins[2], HIGH);


  // Setup select pins
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(muxSelPins[i], OUTPUT);
    digitalWriteFast(muxSelPins[i], LOW);
  }

  // Setup READ select pins
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(READmuxSelPins[i], OUTPUT);
    digitalWriteFast(READmuxSelPins[i], LOW);
  }

  // Setup signal pin
  pinMode(signalPin, OUTPUT);
  //digitalWriteFast(signalPin, LOW);
  digitalWriteFast(signalPin, HIGH);

  // Setup READ signal pin
  pinMode(READsignalPin, INPUT);

}

void loop() {
    for (uint8_t ch = 0; ch < 16; ch++) {
      selectChannel(ch);
      for (uint8_t rch = 0; rch < 16; rch++) {
        READselectChannel(rch);
        Serial.print(digitalReadFast(READsignalPin));
        delay(10);
      }
    }

      //delay(10);
}

*/