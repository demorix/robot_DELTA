#include <Arduino.h>
#include <avr/wdt.h>

// ===== ПІНИ =====
#define TRIG_PIN 12
#define ECHO_PIN 13

#define ML_Ctrl 4
#define ML_PWM  6
#define MR_Ctrl 2
#define MR_PWM  5

// ===== НАЛАШТУВАННЯ =====
#define DEFAULT_DISTANCE 100   // см
#define DEFAULT_TURN     45    // градуси
#define DEFAULT_ROTATE   360   // градуси

#define OBSTACLE_DIST 10       // см

// коефіцієнти (ПІДІБРАТИ)
#define MS_PER_CM  28
#define MS_PER_DEG_LEFT 4
#define MS_PER_DEG_RIGHT 4
#define MS_PER_DEG_ROTATE_LEFT 6
#define MS_PER_DEG_ROTATE_RIGHT 5

// ===== СТАНИ =====
enum State {
  IDLE,
  FORWARD,
  BACKWARD,
  LEFT,
  RIGHT,
  ROTATE_LEFT,
  ROTATE_RIGHT
};

State currentState = IDLE;

// ===== ЧАС =====
unsigned long actionStart = 0;
unsigned long actionTime  = 0;

// ===== ДЛЯ УЛЬТРАЗВУКУ (НЕБЛОКУЮЧЕ) =====
unsigned long lastMeasure = 0;
long distance = 100;

// ===== SETUP =====
void setup() {
  Serial.begin(9600);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pinMode(ML_Ctrl, OUTPUT);
  pinMode(ML_PWM, OUTPUT);
  pinMode(MR_Ctrl, OUTPUT);
  pinMode(MR_PWM, OUTPUT);

  stopMotors();

  wdt_enable(WDTO_8S);
}

// ===== LOOP =====
void loop() {
  wdt_reset();

  readCommand();
  updateDistance();
  handleMovement();
}

// ===== ЧИТАННЯ КОМАНД =====
void readCommand() {
  static String cmd = "";

  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      parseCommand(cmd);
      cmd = "";
    } else {
      cmd += c;
    }
  }
}

// ===== ПАРСИНГ =====
void parseCommand(String command) {
  command.trim();

  int value = -1;
  int spaceIndex = command.indexOf(' ');

  if (spaceIndex != -1) {
    value = command.substring(spaceIndex + 1).toInt();
    command = command.substring(0, spaceIndex);
  }

  if (command == "FORWARD") {
    if (value == -1) value = DEFAULT_DISTANCE;
    startAction(FORWARD, value);
  }
  else if (command == "BACKWARD") {
    if (value == -1) value = DEFAULT_DISTANCE;
    startAction(BACKWARD, value);
  }
  else if (command == "LEFT") {
    if (value == -1) value = DEFAULT_TURN;
    startAction(LEFT, value);
  }
  else if (command == "RIGHT") {
    if (value == -1) value = DEFAULT_TURN;
    startAction(RIGHT, value);
  }
  else if (command == "ROTATE_LEFT") {
    if (value == -1) value = DEFAULT_ROTATE;
    startAction(ROTATE_LEFT, value);
  }
  else if (command == "ROTATE_RIGHT") {
    if (value == -1) value = DEFAULT_ROTATE;
    startAction(ROTATE_RIGHT, value);
  }
}

// ===== ЗАПУСК ДІЇ =====
void startAction(State state, int value) {
  currentState = state;
  actionStart = millis();

  switch (state) {
    case FORWARD:
    case BACKWARD:
      actionTime = value * MS_PER_CM;
      break;

    case LEFT:
      actionTime = value * MS_PER_DEG_LEFT;
      break;

    case RIGHT:
      actionTime = value * MS_PER_DEG_RIGHT;
      break;

    case ROTATE_LEFT:
      actionTime = value * MS_PER_DEG_ROTATE_LEFT;
      break;

    case ROTATE_RIGHT:
      actionTime = value * MS_PER_DEG_ROTATE_RIGHT;
      break;

    default:
      actionTime = 0;
      break;
  }

  // запуск моторів
  switch (state) {
    case FORWARD: moveForward(); break;
    case BACKWARD: moveBackward(); break;
    case LEFT: turnLeft(); break;
    case RIGHT: turnRight(); break;
    case ROTATE_LEFT: rotateLeft(); break;
    case ROTATE_RIGHT: rotateRight(); break;
    default: break;
  }
}

// ===== ОСНОВНА ЛОГІКА =====
void handleMovement() {
  if (currentState == IDLE) return;

  // ПЕРЕШКОДА (тільки вперед/назад)
  if (currentState == FORWARD || currentState == BACKWARD) {
    if (distance < OBSTACLE_DIST && distance > 0) {
      stopMotors();
      currentState = IDLE;
      Serial.println("STOP_OBSTACLE");
      return;
    }
  }

  // ЗАВЕРШЕННЯ ДІЇ
  if (millis() - actionStart >= actionTime) {
    stopMotors();
    currentState = IDLE;
    Serial.println("DONE");
  }
}

// ===== УЛЬТРАЗВУК (НЕБЛОКУЮЧИЙ ВИКЛИК) =====
void updateDistance() {
  if (millis() - lastMeasure < 50) return; // кожні 50 мс
  lastMeasure = millis();

  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 20000);

  if (duration > 0) {
    distance = duration * 0.034 / 2;
  }
}

// ===== МОТОРИ =====
void moveForward() {
  digitalWrite(ML_Ctrl, HIGH);
  digitalWrite(MR_Ctrl, HIGH);
  analogWrite(ML_PWM, 100);
  analogWrite(MR_PWM, 100);
}

void moveBackward() {
  digitalWrite(ML_Ctrl, LOW);
  digitalWrite(MR_Ctrl, LOW);
  analogWrite(ML_PWM, 170);
  analogWrite(MR_PWM, 170);
}

void turnLeft() {
  digitalWrite(ML_Ctrl, LOW);
  digitalWrite(MR_Ctrl, HIGH);
  analogWrite(ML_PWM, 200);
  analogWrite(MR_PWM, 55);
}

void turnRight() {
  digitalWrite(ML_Ctrl, HIGH);
  digitalWrite(MR_Ctrl, LOW);
  analogWrite(ML_PWM, 55);
  analogWrite(MR_PWM, 200);
}

void rotateLeft() {
  digitalWrite(ML_Ctrl, LOW);
  digitalWrite(MR_Ctrl, HIGH);
  analogWrite(ML_PWM, 100);
  analogWrite(MR_PWM, 0);
}

void rotateRight() {
  digitalWrite(ML_Ctrl, HIGH);
  digitalWrite(MR_Ctrl, LOW);
  analogWrite(ML_PWM, 0);
  analogWrite(MR_PWM, 140);
}

void stopMotors() {
  digitalWrite(ML_Ctrl, LOW);
  digitalWrite(MR_Ctrl, LOW);
  analogWrite(ML_PWM, 0);
  analogWrite(MR_PWM, 0);
}
