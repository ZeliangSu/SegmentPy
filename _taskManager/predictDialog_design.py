# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/predictDialog_design.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(559, 300)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.predLabel = QtWidgets.QLabel(Form)
        self.predLabel.setObjectName("predLabel")
        self.verticalLayout_3.addWidget(self.predLabel)
        self.predLine = QtWidgets.QLineEdit(Form)
        self.predLine.setObjectName("predLine")
        self.verticalLayout_3.addWidget(self.predLine)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.predButton = QtWidgets.QPushButton(Form)
        self.predButton.setObjectName("predButton")
        self.horizontalLayout_3.addWidget(self.predButton)
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.metaLabel = QtWidgets.QLabel(Form)
        self.metaLabel.setObjectName("metaLabel")
        self.verticalLayout.addWidget(self.metaLabel)
        self.metaLine = QtWidgets.QLineEdit(Form)
        self.metaLine.setObjectName("metaLine")
        self.verticalLayout.addWidget(self.metaLine)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.metaButton = QtWidgets.QPushButton(Form)
        self.metaButton.setObjectName("metaButton")
        self.horizontalLayout.addWidget(self.metaButton)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.rawLabel = QtWidgets.QLabel(Form)
        self.rawLabel.setObjectName("rawLabel")
        self.verticalLayout_2.addWidget(self.rawLabel)
        self.rawLine = QtWidgets.QLineEdit(Form)
        self.rawLine.setObjectName("rawLine")
        self.verticalLayout_2.addWidget(self.rawLine)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.rawButton = QtWidgets.QPushButton(Form)
        self.rawButton.setObjectName("rawButton")
        self.horizontalLayout_2.addWidget(self.rawButton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.predLabel.setText(_translate("Form", "select a folder to put your predictions"))
        self.predButton.setText(_translate("Form", "..."))
        self.metaLabel.setText(_translate("Form", "select a checkpoint file .meta"))
        self.metaButton.setText(_translate("Form", "..."))
        self.rawLabel.setText(_translate("Form", "select a folder of raw tomogram (*.tif) to predict"))
        self.rawButton.setText(_translate("Form", "..."))

