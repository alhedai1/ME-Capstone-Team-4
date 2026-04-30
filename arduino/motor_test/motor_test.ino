// 2 Motor DriverS: BTS7960
// Pins: RPWM, LPWM, R_EN, L_EN
// To control:
//   enable both sides (R_EN & L_EN)
//   to rotate: apply PWM to RPWM, set LPWM to 0
//   reverse: apply PWM to LPWM, set RPWM to 0

// Can connect R_EN and L_EN together to 5V instead of arduino pins

// ===== Motor 1 (BTS7960 #1) =====
const int M1_RPWM = 5;   // must be PWM-capable
const int M1_LPWM = 6;   // must be PWM-capable
const int M1_REN  = 7;
const int M1_LEN  = 8;

// ===== Motor 2 (BTS7960 #2) =====
const int M2_RPWM = 9;   // must be PWM-capable
const int M2_LPWM = 10;  // must be PWM-capable
const int M2_REN  = 11;
const int M2_LEN  = 12;

// Set to true if motor 2 is physically mounted reversed
bool invertMotor2 = false;

// Optional trim if one motor is slightly stronger
int motor1_trim = 0;
int motor2_trim = 0;

// speed: -255 to 255
void setMotor1(int speed) {
  speed = constrain(speed, -255, 255);

  if (speed > 0) {
    analogWrite(M1_RPWM, speed);
    analogWrite(M1_LPWM, 0);
  } else if (speed < 0) {
    analogWrite(M1_RPWM, 0);
    analogWrite(M1_LPWM, -speed);
  } else {
    analogWrite(M1_RPWM, 0);
    analogWrite(M1_LPWM, 0);
  }
}

// speed: -255 to 255
void setMotor2(int speed) {
  speed = constrain(speed, -255, 255);

  if (invertMotor2) {
    speed = -speed;
  }

  if (speed > 0) {
    analogWrite(M2_RPWM, speed);
    analogWrite(M2_LPWM, 0);
  } else if (speed < 0) {
    analogWrite(M2_RPWM, 0);
    analogWrite(M2_LPWM, -speed);
  } else {
    analogWrite(M2_RPWM, 0);
    analogWrite(M2_LPWM, 0);
  }
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


void setup() {
  // put your setup code here, to run once:
  pinMode(M1_RPWM, OUTPUT);
  pinMode(M1_LPWM, OUTPUT);
  pinMode(M1_REN, OUTPUT);
  pinMode(M1_LEN, OUTPUT);

  pinMode(M2_RPWM, OUTPUT);
  pinMode(M2_LPWM, OUTPUT);
  pinMode(M2_REN, OUTPUT);
  pinMode(M2_LEN, OUTPUT);

  // Enable both BTS7960 boards
  digitalWrite(M1_REN, HIGH);
  digitalWrite(M1_LEN, HIGH);
  digitalWrite(M2_REN, HIGH);
  digitalWrite(M2_LEN, HIGH);

  Serial.begin(115200);
  stopWheel();

  Serial.println("Starting BTS7960 dual-motor wheel test...");
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println("Forward");
  setWheel(180);
  delay(3000);

  Serial.println("Stop");
  stopWheel();
  delay(1500);

  // Serial.println("Reverse");
  // setWheel(-180);
  // delay(3000);

  // Serial.println("Stop");
  // stopWheel();
  // delay(3000);
}
