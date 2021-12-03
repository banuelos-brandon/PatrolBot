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

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
    USE_WEB_ENGINE = True
except ImportError:
    # Do not import PyQt5.WebEngine if you are using the raspberry pi since it is not supported.
    # Otherwise, use the web engine for every other platform.
    USE_WEB_ENGINE = False

#global variable to toggle bounding boxes and labels
enableFlag = True
#global variable toggle object detection
runModel = True
#global variable to toggle specific objects being labeled
labelFlags = {'Person': True, 'Bike': True, 'Angle Grinder': True, 'Bolt Cutters': True}

class WelcomeScreen(QDialog):
    #PyQt5 page format code adapted from https://github.com/codefirstio/pyqt5-full-app-tutorial-for-beginners/blob/main/main.py

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

    def goToLogin(self):
        #set stack index to 1 which is where the login page is located
        widget.setCurrentIndex(widget.currentIndex() + self.loginIndex)

    def goToCreate(self):
        #set stack index to 2 which is where the register page is located
        widget.setCurrentIndex(widget.currentIndex() + self.registerIndex)

class CreateAccount(QDialog):
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

    def goToLogin(self):
        #set stack index to 1 which is where the login page is located
        widget.setCurrentIndex(widget.currentIndex()-1)

    def goBack(self):
        #set stack index to 0 which is where the welcome page is located
        widget.setCurrentIndex(widget.currentIndex() -2 )

class LoginForm(QDialog):
    #login form code adapted from https://learndataanalysis.org/create-a-simple-login-form-pyqt5-tutorial/

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

    def check_password(self):
        msg = QMessageBox()

        if self.lineEdit_username.text() == 'Username' and self.lineEdit_password.text() == 'password':
            self.goToDashboard()

        else:
            #output text box indicating unsuccessful login
            msg.setText('Incorrect Password')
            msg.exec_()

    def goToDashboard(self):
        #set stack index to 3 which is where the dashboard page is located
        widget.setCurrentIndex(widget.currentIndex() + 2)

    def goBack(self):
        #set stack index to 0 which is where the welcome page is located
        widget.setCurrentIndex(widget.currentIndex() -1 )


class ShowDashboard(QDialog):
    #camera implementation code adapted from https://www.youtube.com/watch?v=dTDgbx-XelY

    def __init__(self):
        super().__init__()

        self.setStyleSheet('background-color: blue;')
        layout = QGridLayout()
        self.setLayout(layout)

        #declare CameraFeed thread object within dashboard
        self.camera = CameraFeed()

        if USE_WEB_ENGINE:
            # declare canvas map web view thread object within dashboard
            # only if the use_web_engine flag is set to True
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

        if USE_WEB_ENGINE:
            # If pyqt5.webengine is available, use it.
            # Add canvas map web view to layout
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

        #camera feed is output on top of this widget
        self.feed_label = QLabel('Waiting for camera input...')
        self.feed_label.setStyleSheet("color: white;")
        layout.addWidget(self.feed_label, 2,0)

    def button_listener(self, func):
        func()

    def logout(self):
        #set stack index to 0 which is where the welcome page is located
        widget.setCurrentIndex(widget.currentIndex() - 3)

    def options(self):
        #set stack index to 4 which is where the welcome page is located
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def startCamera(self):
        #start the camera thread
        self.camera.start()
        #call to update the camera feed to main thread
        self.camera.ImageUpdate.connect(self.ImageUpdateSlot)
        #call to update the log form
        self.camera.LogUpdate.connect(self.LogUpdateSlot)

    def ImageUpdateSlot(self, Image):
        #turn the image into a QPixmap
        #this form is readable to PyQt5 as an Image
        #puts the update image onto the screen
        self.feed_label.setPixmap(QPixmap.fromImage(Image))


    def LogUpdateSlot(self, label):
        #get current time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        #Get timezone naive now
        dt = datetime.today()
        #get current seconds
        seconds = (dt.timestamp() % 10)
        #every 10 seconds append to log from
        if (int(seconds) == 0):
            msg = "\n" + current_time + ": " + label + " detected"
            #append to the log form
            self.log_form.appendPlainText(msg)

class CanvasMap(QThread):
    webview_state = pyqtSignal(str)

    def __init__(self, view, web_attributes, file_path):
        super(CanvasMap, self).__init__()
        self.view = view
        self.web_attributes = web_attributes
        self.file_path = file_path

        # set the webview inactive by default
        self.view.setDisabled(True)

        self.state = self.view.isEnabled()
        self.set_web_settings()

    def set_web_settings(self):
        for i, attr in enumerate(self.web_attributes):
            self.view.settings().setAttribute(attr, True)

    def get_map_file(self):
        return "file://" + os.path.join(os.getcwd(), self.file_path).replace("\\", "/")

    def status(self):
        # enables/disables webview interactivity
        if not self.state:
            self.state = True
            self.view.setDisabled(False)
            self.webview_state.emit("Started")
        else:
            self.state = False
            self.view.setDisabled(True)
            self.webview_state.emit("Stopped")

class CameraFeed(QThread):
    #sends an updated QImage as a signal to the variable ImageUpdate
    ImageUpdate = pyqtSignal(QImage)

    LogUpdate = pyqtSignal(str)

    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)

        global runModel
        modelLoaded = False

        if(runModel == True):
            #Torch code adapted from https://github.com/akash-agni/Real-Time-Object-Detection/blob/main/Object_Detection_Youtube.py
            #Torch implementation. Replace third item with your YoloV5 weight's exact path
            model_weight_path = os.path.join(os.getcwd(), 'model_weights/best.pt')
            model = torch.hub.load('ultralytics/yolov5', 'custom', model_weight_path)
            #model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
            #extract the names of the classes for trained the YoloV5 model

            classes = model.names
            class_ids = [0,1,2,3]
            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
            modelLoaded = True

        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #flip video frame, so it isn't reversed
                image = cv2.flip(image, 1)

                #if the model is turned on but the object never initialized
                #stop the camera feed to prevent crashing
                if(runModel == True and modelLoaded == False):
                    self.stop()

                #if model is turned on and the object is initialized
                #run object detection on each frame
                if(runModel == True and modelLoaded == True):
                    ################################################################
                    #TORCH OBJECT DETECTION
                    ################################################################


                    #get dimensions of the current video frame
                    x_shape = image.shape[1]
                    y_shape = image.shape[0]

                    #apply the Torch YoloV5 model to this frame
                    results = model(image)
                    #extract the labels and coordinates of the bounding boxes
                    labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

                    numberOfLabels = len(labels)

                    for i in range(numberOfLabels):
                        row = cords[i]
                        #get the class number of current label
                        class_number = int(labels[i])
                        #index colors list with current label number
                        color = COLORS[class_ids[class_number]]

                        #if confidence level is greater than 0.2
                        if row[4] >= 0.2:
                            #get label to send to dashbaord
                            label = classes[class_number]
                            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                            #if global enable flag is set true then show boxes
                            global enableFlag
                            global labelFlags
                            if (enableFlag == True):
                                if (labelFlags[label] == True):
                                    #draw bounding box
                                    cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                                    #give bounding box a text label
                                    cv2.putText(image, str(classes[int(labels[i])]), (int(x1)-10, int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                            self.LogUpdate.emit(label)

                #convert the image to QImage format
                ConvertToQtFormat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                #send out the new image as a signal to dashboard
                self.ImageUpdate.emit(Pic)

    #stop the threads execution
    def stop(self):
        self.ThreadActive = False
        self.quit()

class OptionsForm(QDialog):
    def __init__(self):
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

        self.textlbl = QLabel(self)
        self.textlbl.move(400,410)
        self.textlbl.setText("Objects to Label:")
        self.textlbl.setStyleSheet("color: white; font-size:24px;" )
        self.textlbl.resize(200,20)

        button_back = QPushButton('Back')
        button_back.setStyleSheet("color: black;")
        button_back.setStyleSheet('background-color: white;')
        button_back.clicked.connect(self.back)
        layout.addWidget(button_back, 1, 0)

        self.enable_box = QCheckBox("Enable Overlay",self)
        self.enable_box.setChecked(True)
        self.enable_box.move(20,410)
        self.enable_box.resize(320,40)
        self.enable_box.stateChanged.connect(self.statechanged)

        self.People = QCheckBox("People",self)
        self.People.setChecked(True)
        self.People.move(400,450)
        self.People.resize(320,40)
        self.People.setStyleSheet(StyleSheet)
        self.People.stateChanged.connect(self.statechanged1)

        self.Bikes = QCheckBox("Bikes",self)
        self.Bikes.setChecked(True)
        self.Bikes.move(400,490)
        self.Bikes.resize(320,40)
        self.Bikes.setStyleSheet(StyleSheet)
        self.Bikes.stateChanged.connect(self.statechanged1)

        self.AngleGrinders = QCheckBox("Angle Grinders",self)
        self.AngleGrinders.setChecked(True)
        self.AngleGrinders.move(400,530)
        self.AngleGrinders.resize(320,40)
        self.AngleGrinders.setStyleSheet(StyleSheet)
        self.AngleGrinders.stateChanged.connect(self.statechanged1)

        self.BoltCutters = QCheckBox("Bolt Cutters",self)
        self.BoltCutters.setChecked(True)
        self.BoltCutters.move(400,570)
        self.BoltCutters.resize(320,40)
        self.BoltCutters.setStyleSheet(StyleSheet)
        self.BoltCutters.stateChanged.connect(self.statechanged1)

        self.run_model_box = QCheckBox("Run Object Detection",self)
        self.run_model_box.setChecked(True)
        self.run_model_box.move(20,450)
        self.run_model_box.resize(320,40)
        self.run_model_box.stateChanged.connect(self.statechanged)

        self.enable_box.setStyleSheet(StyleSheet)
        self.run_model_box.setStyleSheet(StyleSheet)



    def statechanged(self, int):
        global enableFlag
        global runModel

        #if box is checked bounding boxes enabled
        if self.enable_box.isChecked():
            enableFlag = True
        else :
            enableFlag = False

        #if box is checked model is enabled
        if self.run_model_box.isChecked():
            runModel = True
        else :
            runModel = False

    def statechanged1(self, int):
        global labelFlags

        #if people box is checked label is enabled
        if self.People.isChecked():
            labelFlags['Person'] = True
        else :
            labelFlags['Person'] = False

        #if bike box is checked label is enabled
        if self.Bikes.isChecked():
            labelFlags['Bike'] = True
        else :
            labelFlags['Bike'] = False

        #if angle grinder box is checked label is enabled
        if self.AngleGrinders.isChecked():
            labelFlags['Angle Grinder'] = True
        else :
            labelFlags['Angle Grinder'] = False

        #if bolt cutter box is checked label is enabled
        if self.BoltCutters.isChecked():
            labelFlags['Bolt Cutters'] = True
        else :
            labelFlags['Bolt Cutters'] = False



    def back(self):
        #set stack index to 3 which is where the dashboard page is located
        widget.setCurrentIndex(widget.currentIndex() - 1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    #create welcome screen page
    form = WelcomeScreen()
    #declare stack of widgets
    widget = QStackedWidget()
    #add welcome page to stack at index 0
    widget.addWidget(form)
    #create login page
    login = LoginForm()
    #add login page to stack at index 1
    widget.addWidget(login)
    #create register page
    create = CreateAccount()
    #add register page to stack at index 2
    widget.addWidget(create)
    #create dashboard page
    dashboard = ShowDashboard()
    #add dashboard page to stack at index 3
    widget.addWidget(dashboard)
    #create options page
    options = OptionsForm()
    #add options page to stack at index 4
    widget.addWidget(options)
    widget.setFixedHeight(800)
    widget.setFixedWidth(1200)
    #display widget at bottom of stack (welcome page)
    widget.show()
    sys.exit(app.exec_())