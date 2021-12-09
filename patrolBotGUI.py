# File:     patrolBotGUI.py
# Date:     December 3rd 2021
# Authors:  Brandon Banuelos, Jesus Aguilera, Connor Callister, Max Orloff, Michael Stepzinski
# Purpose:  GUI for the CS425 PatrolBot project, works on x86 architecture

import sys
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from functools import partial

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# PyQt5.WebEngine is unsupported on ARM, handle exception for that case
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
    USE_WEB_ENGINE = True
except ImportError:
    USE_WEB_ENGINE = False

# Global variables

#   Toggle bounding boxes and labels
enableFlag = True
#   Toggle object detection
runModel = True
#   Toggle specific objects being labeled
labelFlags = {'Person': True, 'Bike': True, 'Angle Grinder': True, 'Bolt Cutters': True}
#   Toggle specific objects being logged
logFlags = {'Person': True, 'Bike': True, 'Angle Grinder': True, 'Bolt Cutters': True}
#   Action log save directory path
action_dir_path = 'Action_Logs/'
#   Alert log save directory path
alert_dir_path = 'Alert_Logs/'

# Define welcome screen page
class WelcomeScreen(QDialog):
    # PyQt5 page format code adapted from https://github.com/codefirstio/pyqt5-full-app-tutorial-for-beginners/blob/main/main.py

    # Define WelcomeScreen text and buttons
    def __init__(self):
        super(WelcomeScreen, self).__init__()

        self.setStyleSheet('background-color: blue;')
        self.loginIndex = 1
        self.registerIndex = 2
        layout = QGridLayout()

        label_logo = QLabel('<font size="10"> Welcome to PatrolBot </font>')
        label_logo.setStyleSheet("color: white;")
        layout.addWidget(label_logo, 0, 0)

        button_login = QPushButton('Login')
        button_login.setStyleSheet("color: black;")
        button_login.setStyleSheet('background-color: white;')
        button_login.clicked.connect(self.goToLogin)
        layout.addWidget(button_login, 2, 0, 1, 2)
        layout.setRowMinimumHeight(2, 75)

        button_login = QPushButton('Register')
        button_login.setStyleSheet("color: black;")
        button_login.setStyleSheet('background-color: white;')
        button_login.clicked.connect(self.goToCreate)
        layout.addWidget(button_login, 3, 0, 1, 2)
        layout.setRowMinimumHeight(2, 75)

        self.setLayout(layout)

    # Define goToLogin button event
    def goToLogin(self):
        # Set stack index to 1 for login page
        widget.setCurrentIndex(widget.currentIndex() + self.loginIndex)

    # Define goToCreate button event
    def goToCreate(self):
        # Set stack index to 2 for register page
        widget.setCurrentIndex(widget.currentIndex() + self.registerIndex)

# Define create account page
class CreateAccount(QDialog):
    # Define text and buttons
    def __init__(self):
        super().__init__()

        self.setStyleSheet('background-color: blue;')
        layout = QGridLayout()

        label_logo = QLabel('<font size="10"> Create an Account </font>')
        label_logo.setStyleSheet("color: white;")
        layout.addWidget(label_logo, 0, 0)

        button_login = QPushButton('Go back to welcome page')
        button_login.setStyleSheet("color: black;")
        button_login.setStyleSheet('background-color: white;')
        button_login.clicked.connect(self.goBack)
        layout.addWidget(button_login, 1, 0, 1, 2)
        layout.setRowMinimumHeight(2, 75)

        label_name = QLabel('<font size="4"> Username </font>')
        label_name.setStyleSheet("color: white;")
        self.lineEdit_username = QLineEdit()
        self.lineEdit_username.setStyleSheet('background-color: white;')
        self.lineEdit_username.setPlaceholderText('Please enter your username')
        layout.addWidget(label_name, 2, 0)
        layout.addWidget(self.lineEdit_username, 2, 1)

        label_password = QLabel('<font size="4"> Password </font>')
        label_password.setStyleSheet("color: white;")

        self.lineEdit_password = QLineEdit()
        self.lineEdit_password.setEchoMode(QLineEdit.Password)
        self.lineEdit_password.setStyleSheet('background-color: white;')
        self.lineEdit_password.setPlaceholderText('Please enter your password')
        layout.addWidget(label_password, 3, 0)
        layout.addWidget(self.lineEdit_password, 3, 1)

        button_login = QPushButton('Create Account')
        button_login.setStyleSheet("color: black;")
        button_login.setStyleSheet('background-color: white;')
        button_login.clicked.connect(self.goToLogin)
        layout.addWidget(button_login, 4, 0, 1, 2)
        layout.setRowMinimumHeight(2, 75)

        self.setLayout(layout)

    # Define button_login functionality
    def goToLogin(self):
        # Set stack index to 1 for login page
        widget.setCurrentIndex(widget.currentIndex()-1)

    # Define back button functionality
    def goBack(self):
        # Set stack index to 0 for welcome page
        widget.setCurrentIndex(widget.currentIndex() -2 )

# Define login form page
class LoginForm(QDialog):
    # Login form code adapted from https://learndataanalysis.org/create-a-simple-login-form-pyqt5-tutorial/

    # Define text and buttons
    def __init__(self):
        super().__init__()

        self.setStyleSheet('background-color: blue;')

        layout = QGridLayout()

        label_logo = QLabel('<font size="10"> Login to PatrolBot </font>')
        label_logo.setStyleSheet("color: white;")
        layout.addWidget(label_logo, 0, 0)

        button_login = QPushButton('Go back to welcome page')
        button_login.setStyleSheet("color: black;")
        button_login.setStyleSheet('background-color: white;')
        button_login.clicked.connect(self.goBack)
        layout.addWidget(button_login, 1, 0, 1, 2)
        layout.setRowMinimumHeight(2, 75)

        label_name = QLabel('<font size="4"> Username </font>')
        label_name.setStyleSheet("color: white;")
        self.lineEdit_username = QLineEdit()
        self.lineEdit_username.setStyleSheet('background-color: white;')
        self.lineEdit_username.setPlaceholderText('Please enter your username')
        layout.addWidget(label_name, 2, 0)
        layout.addWidget(self.lineEdit_username, 2, 1)

        label_password = QLabel('<font size="4"> Password </font>')
        label_password.setStyleSheet("color: white;")

        self.lineEdit_password = QLineEdit()
        self.lineEdit_password.setEchoMode(QLineEdit.Password)
        self.lineEdit_password.setStyleSheet('background-color: white;')
        self.lineEdit_password.setPlaceholderText('Please enter your password')
        layout.addWidget(label_password, 3, 0)
        layout.addWidget(self.lineEdit_password, 3, 1)

        button_login = QPushButton('Login')
        button_login.setStyleSheet("color: black;")
        button_login.setStyleSheet('background-color: white;')
        button_login.clicked.connect(self.check_password)
        layout.addWidget(button_login, 4, 0, 1, 2)
        layout.setRowMinimumHeight(2, 75)

        self.setLayout(layout)

    # Define on button_login clicked
    def check_password(self):
        msg = QMessageBox()

        # Login is only 'Username' and 'password' for now
        if self.lineEdit_username.text() == 'Username' and self.lineEdit_password.text() == 'password':
            self.goToDashboard()
        else:
            # Output text box indicating unsuccessful login
            msg.setText('Incorrect Password')
            msg.exec_()

    # Define successful login functionality
    def goToDashboard(self):
        # Set stack index to 3 for dashboard page
        widget.setCurrentIndex(widget.currentIndex() + 2)

    # Define back button
    def goBack(self):
        # Set stack index to 0 for welcome page
        widget.setCurrentIndex(widget.currentIndex() -1 )

# Define dashboard page
class ShowDashboard(QDialog):
    # Camera implementation code adapted from https://www.youtube.com/watch?v=dTDgbx-XelY

    # Define text and buttons
    def __init__(self):
        super().__init__()

        self.setStyleSheet('background-color: blue;')
        layout = QGridLayout()
        self.setLayout(layout)

        # Declare CameraFeed thread object within dashboard
        self.camera = CameraFeed()

        # Declare canvas_map_web_view thread object within dashboard
        #  only if the use_web_engine flag is True
        if USE_WEB_ENGINE:
            self.canvas_web_view = CanvasMap(
                view=QWebEngineView(),
                web_attributes=[QWebEngineSettings.JavascriptEnabled],
                file_path="images/mackay.html",
            )

        label_logo = QLabel('<font size="10"> PatrolBot Dashboard </font>')
        label_logo.setStyleSheet("color: white;")
        layout.addWidget(label_logo, 0, 0)

        button_login = QPushButton('Logout')
        button_login.setStyleSheet("color: black;")
        button_login.setStyleSheet('background-color: white;')
        button_login.clicked.connect(self.logout)
        layout.addWidget(button_login, 1, 0)

        self.log_form = QPlainTextEdit('Action Logger')
        self.log_form.setStyleSheet("color: white;")
        self.log_form.setStyleSheet('background-color: white;')
        self.log_form.setReadOnly(True)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = current_time + ': Welcome to Patrol Bot logs'
        self.log_form.appendPlainText(msg)
        layout.addWidget(self.log_form, 2, 1, 1, 2)

        button_save_action_log = QPushButton('Save action log to file')
        button_save_action_log.setStyleSheet('color: black;')
        button_save_action_log.setStyleSheet('background-color: white;')
        button_save_action_log.clicked.connect(self.save_action_log)
        layout.addWidget(button_save_action_log, 3, 1, 1, 2)

        # Notification list to maintain important security alerts
        self.security_alerts = QPlainTextEdit('Security Alerts')
        self.security_alerts.setStyleSheet("color: white;")
        self.security_alerts.setStyleSheet('background-color: white;')
        self.security_alerts.setReadOnly(True)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = current_time + ': No security alerts'
        self.security_alerts.appendPlainText(msg)
        layout.addWidget(self.security_alerts, 4, 1, 1, 2)

        button_save_alert_log = QPushButton('Save alert log to file')
        button_save_alert_log.setStyleSheet('color: black;')
        button_save_alert_log.setStyleSheet('background-color: white;')
        button_save_alert_log.clicked.connect(self.save_alert_log)
        layout.addWidget(button_save_alert_log, 5, 1, 1, 2)

        start_camera = QPushButton('Start Camera')
        start_camera.setStyleSheet("color: black;")
        start_camera.setStyleSheet('background-color: white;')
        start_camera.clicked.connect(self.startCamera)
        layout.addWidget(start_camera, 1, 1)

        stop_camera = QPushButton('Stop Camera')
        stop_camera.setStyleSheet("color: black;")
        stop_camera.setStyleSheet('background-color: white;')
        stop_camera.clicked.connect(self.camera.stop)
        layout.addWidget(stop_camera, 1, 2)

        # Add canvas_map_web_view to layout
        #  only if the use_web_engine flag is True
        if USE_WEB_ENGINE:
            canvas_map_file = self.canvas_web_view.get_map_file()
            view = self.canvas_web_view.view
            view.load(QUrl(canvas_map_file))
            layout.addWidget(self.canvas_web_view.view, 2, 3, 2, 1)

            start_canvas_webview = QPushButton("Canvas Map")
            start_canvas_webview.setStyleSheet("color: black;")
            start_canvas_webview.setStyleSheet("background-color: white;")
            start_canvas_webview.clicked.connect(partial(self.button_listener, self.canvas_web_view.status))
            layout.addWidget(start_canvas_webview, 1, 3)

        button_options = QPushButton('Options')
        button_options.setStyleSheet("color: black;")
        button_options.setStyleSheet('background-color: white;')
        button_options.clicked.connect(self.options)
        layout.addWidget(button_options, 1, 5)

        # Camera feed is output on top of this widget
        self.feed_label = QLabel('Waiting for camera input...')
        self.feed_label.setStyleSheet("color: white;")
        layout.addWidget(self.feed_label, 2,0)

    # Define button_listener for start_canvas_webview
    def button_listener(self, func):
        func()

    # Define logout button functionality
    def logout(self):
        # Set stack index to 0 for welcome page
        widget.setCurrentIndex(widget.currentIndex() - 3)

    # Define options button functionality
    def options(self):
        # Set stack index to 4 for options
        widget.setCurrentIndex(widget.currentIndex() + 1)

    # Define start camera button functionality
    def startCamera(self):
        # Start the camera thread
        self.camera.start()
        # Call to update the camera feed to main thread
        self.camera.ImageUpdate.connect(self.ImageUpdateSlot)
        # Call to update the log form
        self.camera.LogUpdate.connect(self.LogUpdateSlot)

    # Define image updates
    def ImageUpdateSlot(self, Image):
        # Turn the image into a QPixmap
        # This form is readable to PyQt5 as an Image
        # Puts the update image onto the screen
        self.feed_label.setPixmap(QPixmap.fromImage(Image))

    # Define log security alerts
    def LogSecurityAlerts(self, alert='Found potential threat', conf_level=0.0):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self.security_alerts.appendPlainText(current_time + ': ' + alert + '; Confidence Level: ' + str(conf_level))

    # Define text printed when camera updates log
    def LogUpdateSlot(self, label):
        # Get current time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        # Get timezone naive now
        dt = datetime.today()
        # Get current seconds
        seconds = (dt.timestamp() % 10)
        # Every 10 seconds append to log from
        if (int(seconds) == 0):
            if logFlags[label] == True:
                msg = "\n" + current_time + ": " + label + " detected"
                self.log_form.appendPlainText(msg)

    # Define save action log button functionality
    def save_action_log(self):
        # If output directory doesnt exist, create it
        if not os.path.isdir(action_dir_path):
            os.makedirs(action_dir_path)
        # Get current time, then reduce to relevant time components as string
        now = datetime.now()
        now = str(now.year) +           \
            '_' + str(now.month) +      \
            '_' + str(now.day) +        \
            '_h' + str(now.hour) +      \
            '_m' + str(now.minute) +    \
            '_s' + str(now.second) + \
            '.log'
        # Create .txt file at path and output log
        with open(action_dir_path + '/' + now, 'w') as f:
            f.write(self.log_form.toPlainText())

    # Define save alert log button functionality
    def save_alert_log(self):
        # If output directory doesnt exist, create it
        if not os.path.isdir(alert_dir_path):
            os.makedirs(alert_dir_path)
        # Get current time, then reduce to relevant time components as string
        now = datetime.now()
        now = str(now.year) +           \
            '_' + str(now.month) +      \
            '_' + str(now.day) +        \
            '_h' + str(now.hour) +      \
            '_m' + str(now.minute) +    \
            '_s' + str(now.second) + \
            '.log'
        # Create .txt file at path and output log
        with open(alert_dir_path + '/' + now, 'w') as f:
            f.write(self.security_alerts.toPlainText())

# Define canvasmap widget
class CanvasMap(QThread):
    webview_state = pyqtSignal(str)

    def __init__(self, view, web_attributes, file_path):
        super(CanvasMap, self).__init__()
        self.view = view
        self.web_attributes = web_attributes
        self.file_path = file_path

        # Set the webview inactive by default
        self.view.setDisabled(True)

        self.state = self.view.isEnabled()
        self.set_web_settings()

    def set_web_settings(self):
        for i, attr in enumerate(self.web_attributes):
            self.view.settings().setAttribute(attr, True)

    def get_map_file(self):
        return "file://" + os.path.join(os.getcwd(), self.file_path).replace("\\", "/")

    def status(self):
        # Enables/disables webview interactivity
        if not self.state:
            self.state = True
            self.view.setDisabled(False)
            self.webview_state.emit("Started")
        else:
            self.state = False
            self.view.setDisabled(True)
            self.webview_state.emit("Stopped")

# Define notification icon
class NotifcationIcon:
    security_warning = range(1)
    Types = {
        security_warning: None,
    }
    @classmethod
    def initialize(cls):
        cls.Types[cls.security_warning] = QPixmap(os.path.join(os.getcwd(), 'images/baseline_warning_black_24dp.png'))

# Define notification item
class NotificationItem():
    closed = pyqtSignal(QListWidgetItem)

    def __init__(self, title, message, item, *args, notif_type=0, callback=None, **kwargs):
        super(NotificationItem, self).__init__(*args, **kwargs)

# Define notification window
class NotificationWindow(QListWidget):

    def __init__(self, *args, **kwargs):
        super(NotificationWindow, self).__init__(*args, **kwargs)
        self.setSpacing(20)
        self.setMinimumWidth(300)
        self.setMaximumWidth(300)

        QApplication.instance().setQuitOnLastWindowClosed(True)

        self.setFrameShape(self.noFrame)

    @classmethod
    def _createInstance(cls):
        if not cls._instance:
            cls._instance = NotificationWindow()
            cls._instance.show()
            NotifcationIcon.initialize()

    @classmethod
    def security_warning(cls, title, message, callback=None):
        # Creates a notification window for potential security alerts
        cls._createInstance()
        item = QListWidgetItem(cls._instance)
        window = NotificationItem(title, message, item, cls._instance, notif_type=NotifcationIcon.security_warning, callback=callback)
        window.closed.connect(cls._instance.close)
        item.setSizeHint(QSize(cls._instance.width() - cls._instance.spacing(), window.height()))
        cls._instance.addItemWidget(item, window)

# Define camera feed thread
class CameraFeed(QThread):
    # Sends an updated QImage as a signal to the variable ImageUpdate
    ImageUpdate = pyqtSignal(QImage)

    LogUpdate = pyqtSignal(str)

    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)

        global runModel
        modelLoaded = False

        if(runModel == True):
            # Torch code adapted from https://github.com/akash-agni/Real-Time-Object-Detection/blob/main/Object_Detection_Youtube.py
            # Torch implementation. Replace third item with your YoloV5 weight's exact path
            model_weight_path = os.path.join(os.getcwd(), 'model_weights/best.pt')
            model = torch.hub.load('ultralytics/yolov5', 'custom', model_weight_path)
            #model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
            # Extract the names of the classes for trained the YoloV5 model

            classes = model.names
            class_ids = [0,1,2,3]
            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
            modelLoaded = True

        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Flip video frame, so it isn't reversed
                image = cv2.flip(image, 1)

                # If the model is turned on but the object never initialized,
                #  stop the camera feed to prevent crashing
                if(runModel == True and modelLoaded == False):
                    self.stop()

                # If model is turned on and the object is initialized
                #  run object detection on each frame
                if(runModel == True and modelLoaded == True):
                    ################################################################
                    #TORCH OBJECT DETECTION
                    ################################################################

                    # Get dimensions of the current video frame
                    x_shape = image.shape[1]
                    y_shape = image.shape[0]

                    # Apply the Torch YoloV5 model to this frame
                    results = model(image)
                    # Extract the labels and coordinates of the bounding boxes
                    labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

                    numberOfLabels = len(labels)

                    for i in range(numberOfLabels):
                        row = cords[i]
                        # Get the class number of current label
                        class_number = int(labels[i])
                        # Index colors list with current label number
                        color = COLORS[class_ids[class_number]]

                        # If confidence level is greater than 0.2
                        if row[4] >= 0.2:
                            # Get label to send to dashbaord
                            label = classes[class_number]
                            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                            # If global enable flag is set true then show boxes
                            global enableFlag
                            global labelFlags
                            if (enableFlag == True):
                                if (labelFlags[label] == True):
                                    # Draw bounding box
                                    cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                                    # Give bounding box a text label
                                    cv2.putText(image, str(classes[int(labels[i])]), (int(x1)-10, int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                            self.LogUpdate.emit(label)

                # Convert the image to QImage format
                ConvertToQtFormat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                # Send out the new image as a signal to dashboard
                self.ImageUpdate.emit(Pic)

    # Stop the thread's execution
    def stop(self):
        self.ThreadActive = False
        self.quit()

# Define options form page
class OptionsForm(QDialog):
    def __init__(self):
        # Define stylesheet and defaults
        super().__init__()
        StyleSheet = '''
        QCheckBox {
        spacing: 5px;
        font-size:20px;
        color:white;
        }
        QCheckBox::indicator {
        width:  25px;
        height: 25px;
        }
        '''

        self.setStyleSheet('background-color: blue;')
        layout = QGridLayout()
        self.setLayout(layout)

        label_logo = QLabel('<font size="10"> PatrolBot Options </font>')
        label_logo.setStyleSheet("color: white;")
        layout.addWidget(label_logo, 0, 0)

        # Add back button
        button_back = QPushButton('Back')
        button_back.setStyleSheet("color: black;")
        button_back.setStyleSheet('background-color: white;')
        button_back.clicked.connect(self.back)
        layout.addWidget(button_back, 1, 0)

        # Enable overlay checkbox
        self.enable_box = QCheckBox("Enable Overlay",self)
        self.enable_box.setChecked(True)
        self.enable_box.move(20,410)
        self.enable_box.resize(320,40)
        self.enable_box.stateChanged.connect(self.statechanged)

        # Add objects to label section
        self.textlbl = QLabel(self)
        self.textlbl.move(400,410)
        self.textlbl.setText("Objects to Label:")
        self.textlbl.setStyleSheet("color: white; font-size:24px;" )
        self.textlbl.adjustSize()

        # People checkbox for objects to label
        self.People = QCheckBox("People",self)
        self.People.setChecked(True)
        self.People.move(400,450)
        self.People.resize(320,40)
        self.People.setStyleSheet(StyleSheet)
        self.People.stateChanged.connect(self.statechanged1)

        # Bikes checkbox for objects to label
        self.Bikes = QCheckBox("Bikes",self)
        self.Bikes.setChecked(True)
        self.Bikes.move(400,490)
        self.Bikes.resize(320,40)
        self.Bikes.setStyleSheet(StyleSheet)
        self.Bikes.stateChanged.connect(self.statechanged1)
        
        # Angle Grinder checkbox for objects to label
        self.AngleGrinders = QCheckBox("Angle Grinders",self)
        self.AngleGrinders.setChecked(True)
        self.AngleGrinders.move(400,530)
        self.AngleGrinders.resize(320,40)
        self.AngleGrinders.setStyleSheet(StyleSheet)
        self.AngleGrinders.stateChanged.connect(self.statechanged1)

        # Bolt Cutters checkbox for objects to label
        self.BoltCutters = QCheckBox("Bolt Cutters",self)
        self.BoltCutters.setChecked(True)
        self.BoltCutters.move(400,570)
        self.BoltCutters.resize(320,40)
        self.BoltCutters.setStyleSheet(StyleSheet)
        self.BoltCutters.stateChanged.connect(self.statechanged1)
        
        # Check box for turning model on and off
        self.run_model_box = QCheckBox("Run Object Detection",self)
        self.run_model_box.setChecked(True)
        self.run_model_box.move(20,450)
        self.run_model_box.resize(320,40)
        self.run_model_box.stateChanged.connect(self.statechanged)

        self.enable_box.setStyleSheet(StyleSheet)
        self.run_model_box.setStyleSheet(StyleSheet)

        self.textlbl1 = QLabel(self)
        self.textlbl1.move(600,410)
        self.textlbl1.setText("Objects to Log:")
        self.textlbl1.setStyleSheet("color: white; font-size:24px;" )
        self.textlbl1.adjustSize()
        
        # Checkbox to enable and disable logging when people detected
        self.People1 = QCheckBox("People",self)
        self.People1.setChecked(True)
        self.People1.move(600,450)
        self.People1.resize(320,40)
        self.People1.setStyleSheet(StyleSheet)
        self.People1.stateChanged.connect(self.statechanged2)

        # Checkbox to enable and disable logging when bikes detected
        self.Bikes1 = QCheckBox("Bikes",self)
        self.Bikes1.setChecked(True)
        self.Bikes1.move(600,490)
        self.Bikes1.resize(320,40)
        self.Bikes1.setStyleSheet(StyleSheet)
        self.Bikes1.stateChanged.connect(self.statechanged2)

        # Checkbox to enable and disable logging when Angle Grinders detected
        self.AngleGrinders1 = QCheckBox("Angle Grinders",self)
        self.AngleGrinders1.setChecked(True)
        self.AngleGrinders1.move(600,530)
        self.AngleGrinders1.resize(320,40)
        self.AngleGrinders1.setStyleSheet(StyleSheet)
        self.AngleGrinders1.stateChanged.connect(self.statechanged2)

        # Checkbox to enable and disable logging when Bolt Cutters detected
        self.BoltCutters1 = QCheckBox("Bolt Cutters",self)
        self.BoltCutters1.setChecked(True)
        self.BoltCutters1.move(600,570)
        self.BoltCutters1.resize(320,40)
        self.BoltCutters1.setStyleSheet(StyleSheet)
        self.BoltCutters1.stateChanged.connect(self.statechanged2)
        
    # Function to change Enable labels and Run Model flags when boxes are checked and unchecked
    def statechanged(self, int):
        global enableFlag
        global runModel

        # If box is checked bounding boxes enabled
        if self.enable_box.isChecked():
            enableFlag = True
        else :
            enableFlag = False

        # If box is checked model is enabled
        if self.run_model_box.isChecked():
            runModel = True
        else :
            runModel = False
    
    # Function to change label flags when object boxes are checked and unchecked
    def statechanged1(self, int):
        global labelFlags

        # If people box is checked label is enabled
        if self.People.isChecked():
            labelFlags['Person'] = True
        else :
            labelFlags['Person'] = False

        # If bike box is checked label is enabled
        if self.Bikes.isChecked():
            labelFlags['Bike'] = True
        else :
            labelFlags['Bike'] = False

        # If angle grinder box is checked label is enabled
        if self.AngleGrinders.isChecked():
            labelFlags['Angle Grinder'] = True
        else :
            labelFlags['Angle Grinder'] = False

        # If bolt cutter box is checked label is enabled
        if self.BoltCutters.isChecked():
            labelFlags['Bolt Cutters'] = True
        else :
            labelFlags['Bolt Cutters'] = False

    # Function to change log flags when boxes are checked and unchecked
    def statechanged2(self, int):
        global logFlags

        # If people box is checked label is enabled
        if self.People1.isChecked():
            logFlags['Person'] = True
        else :
            logFlags['Person'] = False

        # If bike box is checked label is enabled
        if self.Bikes1.isChecked():
            logFlags['Bike'] = True
        else :
            logFlags['Bike'] = False

        # If angle grinder box is checked label is enabled
        if self.AngleGrinders1.isChecked():
            logFlags['Angle Grinder'] = True
        else :
            logFlags['Angle Grinder'] = False

        # If bolt cutter box is checked label is enabled
        if self.BoltCutters1.isChecked():
            logFlags['Bolt Cutters'] = True
        else :
            logFlags['Bolt Cutters'] = False
    
    # Function to go back to dashboard when back button is pressed
    def back(self):
        # Set stack index to 3 for dashboard page
        widget.setCurrentIndex(widget.currentIndex() - 1)

# Main
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Create welcome screen page
    form = WelcomeScreen()
    # Declare stack of widgets
    widget = QStackedWidget()
    # Add welcome page to stack at index 0
    widget.addWidget(form)
    # Create login page
    login = LoginForm()
    # Add login page to stack at index 1
    widget.addWidget(login)
    # Create register page
    create = CreateAccount()
    # Add register page to stack at index 2
    widget.addWidget(create)
    # Create dashboard page
    dashboard = ShowDashboard()
    # Add dashboard page to stack at index 3
    widget.addWidget(dashboard)
    # Create options page
    options = OptionsForm()
    # Add options page to stack at index 4
    widget.addWidget(options)
    # Set fixed height and width of application window
    widget.setFixedHeight(800)
    widget.setFixedWidth(1200)
    # Display widget at bottom of stack (welcome page)
    widget.show()
    sys.exit(app.exec_())
