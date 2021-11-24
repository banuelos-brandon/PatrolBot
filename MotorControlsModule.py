#Approach to MotorControlModule based on https://github.com/murtazahassan/Neural-Networks-Self-Driving-Car-Raspberry-Pi/blob/main/Step1-Data-Collection/MotorModule.py
import RPi.GPIO as gpio
from time import sleep

#initializes GPIO mode for RaspberryPi
gpio.setmode(gpio.BCM)
gpio.setwarnings(False)

#Motor class
#Motor class members:
#-------enable_right is the GPIO pin number for the enable of the right motors
#-------input_right1 is the GPIO pin number for the the first input of the right motors
#-------input_right2 is the GPIO pin number for the second input of the right motors
#-------pwm_right is the pulse-width of the right motors
#-------enable_left, input_left1, input_left2, and pwm_left have the same characterists as the above members for the left motors
class Motors():
    def __init__(self, enable_right, input_right1, input_right2, enable_left, input_left1, input_left2):
        self.enable_right = enable_right
        self.input_right1 = input_right1
        self.input_right2 = input_right2
        self.enable_left = enable_left
        self.input_left1 = input_left1
        self.input_left2 = input_left2
        gpio.setup(enable_right, gpio.OUT)
        gpio.setup(input_right1, gpio.OUT)
        gpio.setup(input_right2, gpio.OUT)
        gpio.setup(enable_left, gpio.OUT)
        gpio.setup(input_left1, gpio.OUT)
        gpio.setup(input_left2, gpio.OUT)
        self.pwm_right = gpio.PWM(enable_right, 100)
        self.pwm_right.start(0)
        self.pwm_left = gpio.PWM(enable_left, 100)
        self.pwm_left.start(0)

    #reverse method triggers all motors to move in the direction opposite the camera
    def reverse(self, velocity = 50):
        self.pwm_right.ChangeDutyCycle(velocity)
        self.pwm_left.ChangeDutyCycle(velocity)
        gpio.output(self.input_right1, gpio.LOW)
        gpio.output(self.input_right2, gpio.HIGH)
        gpio.output(self.input_left1, gpio.HIGH)
        gpio.output(self.input_left2, gpio.LOW)

    #right method triggers only right motors to move forward
    def right(self, velocity = 50):
        self.pwm_right.ChangeDutyCycle(velocity)
        self.pwm_left.ChangeDutyCycle(0)
        gpio.output(self.input_right1, gpio.HIGH)
        gpio.output(self.input_right2, gpio.LOW)
        gpio.output(self.input_left1, gpio.LOW)
        gpio.output(self.input_left2, gpio.LOW)

    #left method triggers only left motors to move forward
    def left(self, velocity = 50):
        self.pwm_left.ChangeDutyCycle(velocity)
        self.pwm_right.ChangeDutyCycle(0)
        gpio.output(self.input_left1, gpio.LOW)
        gpio.output(self.input_left2, gpio.HIGH)
        gpio.output(self.input_right1, gpio.LOW)
        gpio.output(self.input_right2, gpio.LOW)

    #forward method triggers all motors to move in the direction the camera is facing
    def forward(self, velocity = 50):
        self.pwm_right.ChangeDutyCycle(velocity)
        self.pwm_left.ChangeDutyCycle(velocity)
        gpio.output(self.input_right1, gpio.HIGH)
        gpio.output(self.input_right2, gpio.LOW)
        gpio.output(self.input_left1, gpio.LOW)
        gpio.output(self.input_left2, gpio.HIGH)


    #stop method ensures the motors stop when inputs from the user stops
    def stop(self, delay = 0):
        self.pwm_right.ChangeDutyCycle(0)
        self.pwm_left.ChangeDutyCycle(0)
        sleep(delay)