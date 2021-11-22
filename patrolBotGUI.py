import sys
import os
import cv2
import numpy as np
from datetime import datetime

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

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

        button_options = QPushButton('Options')
        button_options.setStyleSheet("color: black;")
        button_options.setStyleSheet('background-color: white;')
        button_options.clicked.connect(self.options)
        layout.addWidget(button_options, 1, 3)

        #camera feed is output on top of this widget
        self.feed_label = QLabel('Waiting for camera input...')
        self.feed_label.setStyleSheet("color: white;")
        layout.addWidget(self.feed_label, 2,0)

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
        #every 10.0 seconds append to log from
        if (int(seconds) == 0 and (int((seconds*10) % 10)) == 0):
            msg = "\n" + current_time + ": " + label + " detected"
            #append to the log form
            self.log_form.appendPlainText(msg)

class CameraFeed(QThread):
    #sends an updated QImage as a signal to the variable ImageUpdate
    ImageUpdate = pyqtSignal(QImage)

    LogUpdate = pyqtSignal(str)
    
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)

        #OpenCV code adapted from https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9

        #I'm using the regular sized YoloV3 in this example
        #You can use whatever size/model as long as you have the .names file
        #.weights file, and .cfg file for it

        #yolov3 requires a .names file with class names. Replace with your own exact path
        labelsPath = "/Users/brandonbanuelos/Documents/CS 425/Patrol Bot/Yolo/coco.names"
        #get class names for coco example 
        labels = open(labelsPath).read().strip().split("\n")
        #implement a list of random colors for each class
        COLORS = np.random.uniform(0, 255, size=(len(labels), 3))
        #get pretrained weights for YoloV3 and OpenCV. Replace with your own exact path
        weightsPath = "/Users/brandonbanuelos/Documents/CS 425/Patrol Bot/yolov3.weights"
        #get get config file for YoloV3 and OpenCV. Replace with your own exact path
        configPath = "/Users/brandonbanuelos/Documents/CS 425/Patrol Bot/yolov3-darknet-master/yolov3.cfg"
        #read in pretrained YoloV3 model with OpenCV
        net = cv2.dnn.readNet(weightsPath, configPath)

        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                #flip video frame, so it isn't reversed
                image = cv2.flip(image, 1)

                ################################################################
                #OPEN CV OBJECT DETECTION
                ################################################################
                
                #get dimensions of the current video frame
                Width = image.shape[1]
                Height = image.shape[0]
                scale = 0.00392
               
                #normalize input image so it works as an input to neural network
                blob = cv2.dnn.blobFromImage(image, scale, (320,320), (0,0,0), True, crop=False)
                #set YoloV3 input as this normalized image
                net.setInput(blob)

                #find the outputs of the network
                layer_names = net.getLayerNames()
                output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
                outs = net.forward(output_layers)
                
                box_count = 0
                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.2
                nms_threshold = 0.3

                #for every output layer
                for out in outs:

                    #for every prediction in each layer
                    for detection in out:
                        #find the class with the highest confidence
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        #ensure confidence is greater than 0.5
                        if confidence > 0.5:
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2

                            #append the information for the best prediction
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])
                            box_count += 1
                
                #apply non maximum supression to the boxes available to ensure
                #only the best one is output      
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                
                #for every prediction found
                for i in indices:
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    x2 = x+w
                    y2 = y+h
                    
                    color = COLORS[class_ids[i]]

                    #draw bounding box
                    cv2.rectangle(image, (int(x),int(y)), (int(x2),int(y2)), color, 2)
                    class_number = class_ids[i]
                    label = str(labels[class_number])
                    #give bounding box a text label
                    cv2.putText(image, label, (int(x)-10,int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                    #send signal to dashboard class with name of label
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

    #method not used currently
    #can be helpful to implement this method later as a way to toggle
    #the printing of labels
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(classes[class_id])

        #color = COLORS[class_id]
        color = (255,0,0)

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

class OptionsForm(QDialog):
    def __init__(self):
        super().__init__()
    
        self.setStyleSheet('background-color: blue;')
        layout = QGridLayout()
        self.setLayout(layout)

        label_logo = QLabel('<font size="10"> PatrolBot Options </font>')
        label_logo.setStyleSheet("color: white;")
        layout.addWidget(label_logo, 0, 0)

        button_back = QPushButton('Back')
        button_back.setStyleSheet("color: black;")
        button_back.setStyleSheet('background-color: white;')
        button_back.clicked.connect(self.back)
        layout.addWidget(button_back, 1, 0)

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