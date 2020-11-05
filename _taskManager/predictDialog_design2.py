# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/predictDialog_design2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(394, 272)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.predLabel = QtWidgets.QLabel(Dialog)
        self.predLabel.setObjectName("predLabel")
        self.verticalLayout_3.addWidget(self.predLabel)
        self.predLine = QtWidgets.QLineEdit(Dialog)
        self.predLine.setObjectName("predLine")
        self.verticalLayout_3.addWidget(self.predLine)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.predButton = QtWidgets.QPushButton(Dialog)
        self.predButton.setObjectName("predButton")
        self.horizontalLayout_3.addWidget(self.predButton)
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.metaLabel = QtWidgets.QLabel(Dialog)
        self.metaLabel.setObjectName("metaLabel")
        self.verticalLayout.addWidget(self.metaLabel)
        self.metaLine = QtWidgets.QLineEdit(Dialog)
        self.metaLine.setObjectName("metaLine")
        self.verticalLayout.addWidget(self.metaLine)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.metaButton = QtWidgets.QPushButton(Dialog)
        self.metaButton.setObjectName("metaButton")
        self.horizontalLayout.addWidget(self.metaButton)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.rawLabel = QtWidgets.QLabel(Dialog)
        self.rawLabel.setObjectName("rawLabel")
        self.verticalLayout_2.addWidget(self.rawLabel)
        self.rawLine = QtWidgets.QLineEdit(Dialog)
        self.rawLine.setObjectName("rawLine")
        self.verticalLayout_2.addWidget(self.rawLine)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.rawButton = QtWidgets.QPushButton(Dialog)
        self.rawButton.setObjectName("rawButton")
        self.horizontalLayout_2.addWidget(self.rawButton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_4.addWidget(self.buttonBox)
        self.gridLayout_2.addLayout(self.verticalLayout_4, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.predLabel.setText(_translate("Dialog", "select a folder to put your predictions"))
        self.predButton.setText(_translate("Dialog", "..."))
        self.metaLabel.setText(_translate("Dialog", "select a checkpoint file .meta"))
        self.metaButton.setText(_translate("Dialog", "..."))
        self.rawLabel.setText(_translate("Dialog", "select a folder of raw tomogram (*.tif) to predict"))
        self.rawButton.setText(_translate("Dialog", "..."))

