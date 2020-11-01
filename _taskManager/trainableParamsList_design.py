# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/trainableParamsList_design.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_chooseParams(object):
    def setupUi(self, chooseParams):
        chooseParams.setObjectName("chooseParams")
        chooseParams.resize(360, 630)
        self.gridLayout = QtWidgets.QGridLayout(chooseParams)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(chooseParams)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.paramList = QtWidgets.QListWidget(chooseParams)
        self.paramList.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.paramList.setObjectName("paramList")
        self.verticalLayout.addWidget(self.paramList)
        self.label_2 = QtWidgets.QLabel(chooseParams)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(chooseParams)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_2.addWidget(self.buttonBox)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        self.retranslateUi(chooseParams)
        self.buttonBox.accepted.connect(chooseParams.accept)
        self.buttonBox.rejected.connect(chooseParams.reject)
        QtCore.QMetaObject.connectSlotsByName(chooseParams)

    def retranslateUi(self, chooseParams):
        _translate = QtCore.QCoreApplication.translate
        chooseParams.setWindowTitle(_translate("chooseParams", "Dialog"))
        self.label.setText(_translate("chooseParams", "Select the parameters to restore"))
        self.label_2.setText(_translate("chooseParams", "* Not selecting = select all"))

