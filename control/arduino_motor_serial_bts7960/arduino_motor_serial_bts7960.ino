// Arduino motor serial bridge for Raspberry Pi vision control.
//
// BTS7960 version.
//
// Protocol at 115200 baud:
//   M <left> <right>   set wheel speeds from -1.0 to +1.0
//   S                  stop both motors
//   P                  ping; replies OK
//
// Safety:
//   If commands stop arriving for WATCHDOG_TIMEOUT_MS, motors stop.
//
// BTS7960 notes:
//   Each motor driver has RPWM, LPWM, R_EN, and L_EN.
//   Forward: PWM on RPWM, LPWM = 0.
//   Reverse: PWM on LPWM, RPWM = 0.
//   R_EN and L_EN must be HIGH, or tied to 5V externally.

// ===== Left wheel / BTS7960 #1 =====
const int LEFT_RPWM = 5;   // PWM-capable
const int LEFT_LPWM = 6;   // PWM-capable
const int LEFT_REN = 7;
const int LEFT_LEN = 8;

// ===== Right wheel / BTS7960 #2 =====
const int RIGHT_RPWM = 9;   // PWM-capable
const int RIGHT_LPWM = 10;  // PWM-capable
const int RIGHT_REN = 11;
const int RIGHT_LEN = 12;

const unsigned long WATCHDOG_TIMEOUT_MS = 300;

// Tune these after checking physical motor direction and balance.
bool invertLeft = false;
bool invertRight = false;
int leftTrim = 0;
int rightTrim = 0;

String line = "";
unsigned long lastCommandMs = 0;
bool watchdogStopped = true;

void setOneMotor(int rpwmPin, int lpwmPin, int pwm) {
  pwm = constrain(pwm, -255, 255);

  if (pwm > 0) {
    analogWrite(rpwmPin, pwm);
    analogWrite(lpwmPin, 0);
  } else if (pwm < 0) {
    analogWrite(rpwmPin, 0);
    analogWrite(lpwmPin, -pwm);
  } else {
    analogWrite(rpwmPin, 0);
    analogWrite(lpwmPin, 0);
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

  setOneMotor(LEFT_RPWM, LEFT_LPWM, leftPwm);
  setOneMotor(RIGHT_RPWM, RIGHT_LPWM, rightPwm);
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
  pinMode(LEFT_RPWM, OUTPUT);
  pinMode(LEFT_LPWM, OUTPUT);
  pinMode(LEFT_REN, OUTPUT);
  pinMode(LEFT_LEN, OUTPUT);

  pinMode(RIGHT_RPWM, OUTPUT);
  pinMode(RIGHT_LPWM, OUTPUT);
  pinMode(RIGHT_REN, OUTPUT);
  pinMode(RIGHT_LEN, OUTPUT);

  digitalWrite(LEFT_REN, HIGH);
  digitalWrite(LEFT_LEN, HIGH);
  digitalWrite(RIGHT_REN, HIGH);
  digitalWrite(RIGHT_LEN, HIGH);

  Serial.begin(115200);
  stopMotors();
  lastCommandMs = millis();

  Serial.println("Arduino BTS7960 motor serial bridge ready");
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

  if (!watchdogStopped && millis() - lastCommandMs > WATCHDOG_TIMEOUT_MS) {
    stopMotors();
    watchdogStopped = true;
    Serial.println("WATCHDOG STOP");
  }
}
