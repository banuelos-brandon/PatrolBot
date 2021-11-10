import sys
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, QGridLayout, QMessageBox, QStackedWidget, QDialog,
    QPlainTextEdit)
from PyQt5.QtGui import QPixmap


class WelcomeScreen(QDialog):
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
    def __init__(self):
        super().__init__()
        
        self.setStyleSheet('background-color: blue;')
        layout = QGridLayout()
        self.setLayout(layout)

        label_logo = QLabel('<font size="10"> PatrolBot Dashboard </font>')
        label_logo.setStyleSheet("color: white;")
        layout.addWidget(label_logo, 0, 0)

        button_login = QPushButton('Logout')
        button_login.setStyleSheet("color: black;")
        button_login.setStyleSheet('background-color: white;')
        button_login.clicked.connect(self.logout)
        layout.addWidget(button_login, 1, 0, 1, 2)
        layout.setRowMinimumHeight(2, 75)

        log_form = QPlainTextEdit('Action Logger')
        log_form.setStyleSheet("color: white;")
        log_form.setStyleSheet('background-color: white;')
        log_form.setReadOnly(True)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = current_time + ': Welcome to Patrol Bot logs'
        log_form.appendPlainText(msg)
        layout.addWidget(log_form, 2, 1, 1, 2)

    def logout(self):
        #set stack index to 0 which is where the welcome page is located
        widget.setCurrentIndex(widget.currentIndex() - 3)


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
    widget.setFixedHeight(800)
    widget.setFixedWidth(1200)
    #display widget at bottom of stack (welcome page)
    widget.show()
    sys.exit(app.exec_())