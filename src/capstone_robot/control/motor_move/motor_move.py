import RPi.GPIO as GPIO
import time

# 핀 번호 설정 (BCM 모드 기준)
INA = 23
INB = 24

# GPIO 초기화
GPIO.setmode(GPIO.BCM)
GPIO.setup(INA, GPIO.OUT)
GPIO.setup(INB, GPIO.OUT)

# PWM 객체 변수를 미리 선언
pwm_a = None
pwm_b = None

try:
    # PWM 설정 (주파수 100Hz)
    pwm_a = GPIO.PWM(INA, 100)
    pwm_b = GPIO.PWM(INB, 100)

    pwm_a.start(0)
    pwm_b.start(0)

    print("모터 테스트 시작: 정방향 (50% 속도)")
    pwm_a.ChangeDutyCycle(100)
    pwm_b.ChangeDutyCycle(0)
    time.sleep(10)

    print("정지")
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

except KeyboardInterrupt:
    print("\n사용자에 의해 중단됨")

finally:
    # 에러 방지를 위해 명시적으로 PWM 정지 후 정리
    if pwm_a is not None:
        pwm_a.stop()
    if pwm_b is not None:
        pwm_b.stop()
    GPIO.cleanup()
    print("정리 완료")
