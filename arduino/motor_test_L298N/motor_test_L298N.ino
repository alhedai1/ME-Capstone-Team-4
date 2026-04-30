// // 1 L298N driver, controlling 2 DC motors
// // Pins:
// //   Motor 1: ENA, IN1, IN2
// //   Motor 2: ENB, IN3, IN4
// //
// // To control:
// //   forward:  IN1=HIGH, IN2=LOW,  PWM on ENA
// //   reverse:  IN1=LOW,  IN2=HIGH, PWM on ENA
// //   stop:     IN1=LOW,  IN2=LOW,  PWM=0
// //
// // Same idea for motor 2 using IN3, IN4, ENB
// //
// // Note:
// // If your L298N board has ENA/ENB jumpers installed,
// // remove them if you want PWM speed control from Arduino.

// // ===== Motor 1 (L298N channel A) =====
// const int M1_EN = 5;   // must be PWM-capable
// const int M1_IN1 = 6;
// const int M1_IN2 = 7;

// // ===== Motor 2 (L298N channel B) =====
// const int M2_EN = 9;   // must be PWM-capable
// const int M2_IN1 = 11;
// const int M2_IN2 = 10;

// // Set to true if motor 2 is physically mounted reversed
// bool invertMotor2 = false;

// // Optional trim if one motor is slightly stronger
// int motor1_trim = 0;
// int motor2_trim = 0;

// // speed: -255 to 255
// void setMotor1(int speed) {
//   speed = constrain(speed, -255, 255);

//   if (speed > 0) {
//     digitalWrite(M1_IN1, HIGH);
//     digitalWrite(M1_IN2, LOW);
//     analogWrite(M1_EN, speed);
//   } 
//   else if (speed < 0) {
//     digitalWrite(M1_IN1, LOW);
//     digitalWrite(M1_IN2, HIGH);
//     analogWrite(M1_EN, -speed);
//   } 
//   else {
//     digitalWrite(M1_IN1, LOW);
//     digitalWrite(M1_IN2, LOW);
//     analogWrite(M1_EN, 0);
//   }
// }

// // speed: -255 to 255
// void setMotor2(int speed) {
//   speed = constrain(speed, -255, 255);

//   if (invertMotor2) {
//     speed = -speed;
//   }

//   if (speed > 0) {
//     digitalWrite(M2_IN1, HIGH);
//     digitalWrite(M2_IN2, LOW);
//     analogWrite(M2_EN, speed);
//   } 
//   else if (speed < 0) {
//     digitalWrite(M2_IN1, LOW);
//     digitalWrite(M2_IN2, HIGH);
//     analogWrite(M2_EN, -speed);
//   } 
//   else {
//     digitalWrite(M2_IN1, LOW);
//     digitalWrite(M2_IN2, LOW);
//     analogWrite(M2_EN, 0);
//   }
// }

// // Command both motors together as one wheel
// void setWheel(int speed) {
//   int s1 = constrain(speed + motor1_trim, -255, 255);
//   int s2 = constrain(speed + motor2_trim, -255, 255);

//   setMotor1(s1);
//   setMotor2(s2);
// }

// void stopWheel() {
//   setWheel(0);
// }

// void setup() {
//   pinMode(M1_EN, OUTPUT);
//   pinMode(M1_IN1, OUTPUT);
//   pinMode(M1_IN2, OUTPUT);

//   pinMode(M2_EN, OUTPUT);
//   pinMode(M2_IN1, OUTPUT);
//   pinMode(M2_IN2, OUTPUT);

//   Serial.begin(115200);
//   stopWheel();

//   Serial.println("Starting L298N dual-motor wheel test...");

//   Serial.println("Forward");
//   setWheel(180);
//   delay(3000);
// }

// void loop() {
//   // Serial.println("Forward");
//   // setWheel(180);
//   // delay(3000);

//   // Serial.println("Stop");
//   // stopWheel();
//   // delay(1500);

//   // Serial.println("Reverse");
//   // setWheel(-180);
//   // delay(3000);

//   // Serial.println("Stop");
//   // stopWheel();
//   // delay(3000);
// }




// 1 L298N driver, controlling 2 DC motors
// Pins:
//   Motor 1: ENA, IN1, IN2
//   Motor 2: ENB, IN3, IN4
//
// To control:
//   forward:  IN1=HIGH, IN2=LOW,  PWM on ENA
//   reverse:  IN1=LOW,  IN2=HIGH, PWM on ENA
//   stop:     IN1=LOW,  IN2=LOW,  PWM=0
//
// Same idea for motor 2 using IN3, IN4, ENB
//
// Note:
// If your L298N board has ENA/ENB jumpers installed,
// remove them if you want PWM speed control from Arduino.

// ===== Motor 1 (L298N channel A) =====
const int M1_EN  = 5;   // must be PWM-capable
const int M1_IN1 = 6;
const int M1_IN2 = 7;

// ===== Motor 2 (L298N channel B) =====
const int M2_EN  = 9;   // must be PWM-capable
const int M2_IN1 = 11;
const int M2_IN2 = 10;

// Set to true if motor 2 is physically mounted reversed
bool invertMotor2 = false;

// Optional trim if one motor is slightly stronger
int motor1_trim = 0;
int motor2_trim = 0;

// Current commanded speed
int currentSpeed = 180;

// Buffer for incoming serial command
String cmd = "";

// speed: -255 to 255
void setMotor1(int speed) {
  speed = constrain(speed, -255, 255);
  speed = speed * 0.7;

  if (speed > 0) {
    digitalWrite(M1_IN1, HIGH);
    digitalWrite(M1_IN2, LOW);
    analogWrite(M1_EN, speed);
  } 
  else if (speed < 0) {
    digitalWrite(M1_IN1, LOW);
    digitalWrite(M1_IN2, HIGH);
    analogWrite(M1_EN, -speed);
  } 
  else {
    digitalWrite(M1_IN1, LOW);
    digitalWrite(M1_IN2, LOW);
    analogWrite(M1_EN, 0);
  }
  Serial.println(speed);
}

// speed: -255 to 255
void setMotor2(int speed) {
  // speed = speed * 0.01;
  speed = constrain(speed, -255, 255);

  if (invertMotor2) {
    speed = -speed;
  }

  if (speed > 0) {
    digitalWrite(M2_IN1, HIGH);
    digitalWrite(M2_IN2, LOW);
    analogWrite(M2_EN, speed);
  } 
  else if (speed < 0) {
    digitalWrite(M2_IN1, LOW);
    digitalWrite(M2_IN2, HIGH);
    analogWrite(M2_EN, -speed);
  } 
  else {
    digitalWrite(M2_IN1, LOW);
    digitalWrite(M2_IN2, LOW);
    analogWrite(M2_EN, 0);
  }
  Serial.println(speed);
}

// Command both motors together as one wheel
void setWheel(int speed) {
  int s1 = constrain(speed + motor1_trim, -255, 255);
  int s2 = constrain(speed + motor2_trim, -255, 255);

  setMotor1(s1);
  setMotor2(s2);
}

void stopWheel() {
  setWheel(0);
}

void handleCommand(String input) {
  input.trim();
  input.toLowerCase();

  if (input == "run" || input == "f") {
    setWheel(abs(currentSpeed));
    Serial.print("Running forward at speed ");
    Serial.println(abs(currentSpeed));
  }
  else if (input == "r") {
    setWheel(-abs(currentSpeed));
    Serial.print("Running reverse at speed ");
    Serial.println(abs(currentSpeed));
  }
  else if (input == "s") {
    stopWheel();
    Serial.println("Stopped");
  }
  else if (input.startsWith("speed ")) {
    String valueStr = input.substring(6);
    int newSpeed = valueStr.toInt();

    newSpeed = constrain(newSpeed, -255, 255);
    currentSpeed = newSpeed;

    Serial.print("Set currentSpeed = ");
    Serial.println(currentSpeed);
  }
  else {
    Serial.println("Unknown command");
    Serial.println("Available commands:");
    Serial.println("  run");
    Serial.println("  forward");
    Serial.println("  reverse");
    Serial.println("  stop");
    Serial.println("  speed 0..255");
  }
}

void setup() {
  pinMode(M1_EN, OUTPUT);
  pinMode(M1_IN1, OUTPUT);
  pinMode(M1_IN2, OUTPUT);

  pinMode(M2_EN, OUTPUT);
  pinMode(M2_IN1, OUTPUT);
  pinMode(M2_IN2, OUTPUT);

  Serial.begin(115200);
  stopWheel();

  Serial.println("L298N dual-motor wheel test ready");
  Serial.println("Type a command:");
  Serial.println("  run");
  Serial.println("  f");
  Serial.println("  r");
  Serial.println("  s");
  Serial.println("  speed 180");
}

void loop() {
  if (Serial.available()) {
    cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }
}