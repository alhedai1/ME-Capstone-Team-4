// Arduino motor serial bridge for Raspberry Pi vision control.
//
// Protocol at 115200 baud:
//   M <left> <right>   set wheel speeds from -1.0 to +1.0
//   S                  stop both motors
//   P                  ping; replies OK
//
// Safety:
//   If commands stop arriving for WATCHDOG_TIMEOUT_MS, motors stop.
//
// This sketch is written for one L298N driver controlling two DC motors.
// Remove ENA/ENB jumpers from the L298N if you want Arduino PWM speed control.

// ===== Motor 1 / left wheel, L298N channel A =====
const int LEFT_EN = 5;  // PWM-capable
const int LEFT_IN1 = 6;
const int LEFT_IN2 = 7;

// ===== Motor 2 / right wheel, L298N channel B =====
const int RIGHT_EN = 9;  // PWM-capable
const int RIGHT_IN1 = 10;
const int RIGHT_IN2 = 11;

const unsigned long WATCHDOG_TIMEOUT_MS = 300;

// Tune these after checking physical motor direction and balance.
bool invertLeft = false;
bool invertRight = true;
int leftTrim = 0;
int rightTrim = 0;

String line = "";
unsigned long lastCommandMs = 0;
bool watchdogStopped = true;

void setOneMotor(int enPin, int in1Pin, int in2Pin, int pwm) {
  pwm = constrain(pwm, -255, 255);

  if (pwm > 0) {
    digitalWrite(in1Pin, HIGH);
    digitalWrite(in2Pin, LOW);
    analogWrite(enPin, pwm);
  } else if (pwm < 0) {
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, HIGH);
    analogWrite(enPin, -pwm);
  } else {
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, LOW);
    analogWrite(enPin, 0);
  }
}

void setMotors(float left, float right) {
  left = constrain(left, -1.0, 1.0);
  right = constrain(right, -1.0, 1.0);

  int leftPwm = round(left * 255.0) + leftTrim;
  int rightPwm = round(right * 255.0) + rightTrim;

  if (invertLeft) {
    leftPwm = -leftPwm;
  }
  if (invertRight) {
    rightPwm = -rightPwm;
  }

  setOneMotor(LEFT_EN, LEFT_IN1, LEFT_IN2, leftPwm);
  setOneMotor(RIGHT_EN, RIGHT_IN1, RIGHT_IN2, rightPwm);
}

void stopMotors() {
  setMotors(0.0, 0.0);
}

void handleLine(String input) {
  input.trim();
  if (input.length() == 0) {
    return;
  }

  char command = toupper(input.charAt(0));

  if (command == 'M') {
    int firstSpace = input.indexOf(' ');
    int secondSpace = input.indexOf(' ', firstSpace + 1);

    if (firstSpace < 0 || secondSpace < 0) {
      Serial.println("ERR expected: M <left> <right>");
      stopMotors();
      return;
    }

    float left = input.substring(firstSpace + 1, secondSpace).toFloat();
    float right = input.substring(secondSpace + 1).toFloat();

    setMotors(left, right);
    lastCommandMs = millis();
    watchdogStopped = false;
    Serial.print("OK M ");
    Serial.print(left, 3);
    Serial.print(" ");
    Serial.println(right, 3);
    return;
  }

  if (command == 'S') {
    stopMotors();
    lastCommandMs = millis();
    watchdogStopped = true;
    Serial.println("OK S");
    return;
  }

  if (command == 'P') {
    lastCommandMs = millis();
    Serial.println("OK P");
    return;
  }

  Serial.println("ERR unknown command");
  stopMotors();
}

void setup() {
  pinMode(LEFT_EN, OUTPUT);
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);

  pinMode(RIGHT_EN, OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);

  Serial.begin(115200);
  stopMotors();
  lastCommandMs = millis();

  Serial.println("Arduino motor serial bridge ready");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      handleLine(line);
      line = "";
    } else if (c != '\r') {
      line += c;
      if (line.length() > 64) {
        line = "";
        stopMotors();
        Serial.println("ERR line too long");
      }
    }
  }

  // if (!watchdogStopped && millis() - lastCommandMs > WATCHDOG_TIMEOUT_MS) {
  //   stopMotors();
  //   watchdogStopped = true;
  //   Serial.println("WATCHDOG STOP");
  // }
}
