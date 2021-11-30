import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.Qt import Qt
from pynput import keyboard
from MotorControlsModule import Motors
import cv2
import numpy as np

class ManualControlPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: blue;")
        layout = QGridLayout()
        self.setLayout(layout)
        self.feed_label = QLabel()
        self.feed_label.setStyleSheet("color: white;")
        layout.addWidget(self.feed_label, 1.5, 0)
        label_logo = QLabel('<font size="12">Click Start: Use WASD Keys to Move PatrolBot </font>')
        label_logo.setStyleSheet("color: white;")
        layout.addWidget(label_logo, 0, 0)

        button_start = QPushButton("Start")
        button_start.setStyleSheet("color: black;")
        button_start.setStyleSheet("background-color: blue;")
        button_start.clicked.connect(self.start)
        layout.addWidget(button_start, 0, 2)

        button_stop = QPushButton("Stop")
        button_stop.setStyleSheet("color: black;")
        button_stop.setStyleSheet("background-color: blue;")
        button_stop.clicked.connect(self.stop)
        layout.addWidget(button_stop, 0, 3)

        button_back = QPushButton("Back")
        button_back.setStyleSheet("color: black;")
        button_back.setStyleSheet("background-color: blue;")
        button_back.clicked.connect(self.back)
        layout.addWidget(button_back, 0, 4)

        self.Worker = Worker()
        self.Worker.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker.start()

    def ImageUpdateSlot(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.feed_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        final_image = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(final_image)

    def start(self):
        global listener_flag
        listener_flag = True

    def stop(self):
        global listener_flag
        listener_flag = False

    def back(self):
        global listener_flag
        listener_flag = False
        self.Worker.stop()
        widget.setCurrentIndex(widget.currentIndex() -1)

class Worker(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.run_flag = True
    def run(self):
        capture = cv2.VideoCapture(0)
        while self.run_flag:
            ret, frame = capture.read()
            if ret:
                self.ImageUpdate.emit(frame)
        capture.release()
    def stop(self):
        self.run_flag = False
        self.quit()

def on_press(key):
    global motor
    global listener_flag
    if listener_flag == True:
        if key.char == 'A':
            motor.left(50)
        elif key.char == 'W':
            motor.forward(50)
        elif key.char == 'S':
            motor.reverse(50)
        elif key.char == 'D':
            motor.right(50)
def on_release(key):
    global motor
    global listener_flag
    if listener_flag == True:
        motor.stop(0.000000001)

motor = Motors(2, 3, 4, 17, 22, 27)
listener_flag = False
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
app = QApplication(sys.argv)
test = ManualControlPage()
widget = QStackedWidget()
widget.setFixedHeight(800)
widget.setFixedWidth(1200)
widget.addWidget(test)
widget.show()
sys.exit(app.exec_())
listener.stop()