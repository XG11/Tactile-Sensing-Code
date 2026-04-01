#include <Arduino.h>

int analog_pin1 = 22;
int analog_pin2 = 21;
int analog_pin3 = 20;
int analog_pin4 = 19;
int analog_pin5 = 18;

int enable1 = 7;
int enable2 = 8;
int enable3 = 9;
int enable4 = 10;
int enable5 = 11;

int baseline[5][5];
int threshold = 10; //to be changed, need to try multipletimes 

void readMatrix(int matrix[5][5]) {

  digitalWriteFast(enable1, HIGH);
  delayMicroseconds(50);
  matrix[0][0] = analogRead(analog_pin1);
  matrix[0][1] = analogRead(analog_pin2);
  matrix[0][2] = analogRead(analog_pin3);
  matrix[0][3] = analogRead(analog_pin4);
  matrix[0][4] = analogRead(analog_pin5);
  digitalWriteFast(enable1, LOW);

  digitalWriteFast(enable2, HIGH);
  delayMicroseconds(50);
  matrix[1][0] = analogRead(analog_pin1);
  matrix[1][1] = analogRead(analog_pin2);
  matrix[1][2] = analogRead(analog_pin3);
  matrix[1][3] = analogRead(analog_pin4);
  matrix[1][4] = analogRead(analog_pin5);
  digitalWriteFast(enable2, LOW);

  digitalWriteFast(enable3, HIGH);
  delayMicroseconds(50);
  matrix[2][0] = analogRead(analog_pin1);
  matrix[2][1] = analogRead(analog_pin2);
  matrix[2][2] = analogRead(analog_pin3);
  matrix[2][3] = analogRead(analog_pin4);
  matrix[2][4] = analogRead(analog_pin5);
  digitalWriteFast(enable3, LOW);

  digitalWriteFast(enable4, HIGH);
  delayMicroseconds(50);
  matrix[3][0] = analogRead(analog_pin1);
  matrix[3][1] = analogRead(analog_pin2);
  matrix[3][2] = analogRead(analog_pin3);
  matrix[3][3] = analogRead(analog_pin4);
  matrix[3][4] = analogRead(analog_pin5);
  digitalWriteFast(enable4, LOW);

  digitalWriteFast(enable5, HIGH);
  delayMicroseconds(50);
  matrix[4][0] = analogRead(analog_pin1);
  matrix[4][1] = analogRead(analog_pin2);
  matrix[4][2] = analogRead(analog_pin3);
  matrix[4][3] = analogRead(analog_pin4);
  matrix[4][4] = analogRead(analog_pin5);
  digitalWriteFast(enable5, LOW);
}

void setup() {

  Serial.begin(115200);

  pinMode(analog_pin1, INPUT);
  pinMode(analog_pin2, INPUT);
  pinMode(analog_pin3, INPUT);
  pinMode(analog_pin4, INPUT);
  pinMode(analog_pin5, INPUT);

  pinMode(enable1, OUTPUT);
  pinMode(enable2, OUTPUT);
  pinMode(enable3, OUTPUT);
  pinMode(enable4, OUTPUT);
  pinMode(enable5, OUTPUT);

  digitalWriteFast(enable1, LOW);
  digitalWriteFast(enable2, LOW);
  digitalWriteFast(enable3, LOW);
  digitalWriteFast(enable4, LOW);
  digitalWriteFast(enable5, LOW);

  delay(1000);

  // capture baseline
  readMatrix(baseline);
}

void loop() {

  int current[5][5];
  readMatrix(current);

  for(int i=0;i<5;i++){
    for(int j=0;j<5;j++){

      int change = abs(current[i][j] - baseline[i][j]);
      
      Serial.print(current[i][j]);

      if(change > threshold){
        Serial.print("1 ");
      }
      else{
        Serial.print("0 ");
      }
    }
    Serial.println();
  }

  Serial.println();
  delay(10);
}