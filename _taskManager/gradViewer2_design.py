# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/gradViewer2.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_gradPlot(object):
    def setupUi(self, gradPlot):
        gradPlot.setObjectName("gradPlot")
        gradPlot.resize(1156, 773)
        self.gridLayout_2 = QtWidgets.QGridLayout(gradPlot)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(gradPlot)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(gradPlot)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.folderButton = QtWidgets.QToolButton(gradPlot)
        self.folderButton.setObjectName("folderButton")
        self.horizontalLayout.addWidget(self.folderButton)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.plotWidget = gradient_plot(gradPlot)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plotWidget.sizePolicy().hasHeightForWidth())
        self.plotWidget.setSizePolicy(sizePolicy)
        self.plotWidget.setObjectName("plotWidget")
        self.gridLayout.addWidget(self.plotWidget, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(gradPlot)
        QtCore.QMetaObject.connectSlotsByName(gradPlot)

    def retranslateUi(self, gradPlot):
        _translate = QtCore.QCoreApplication.translate
        gradPlot.setWindowTitle(_translate("gradPlot", "Form"))
        self.label.setText(_translate("gradPlot", "Select a training folder to exact the gradients"))
        self.folderButton.setText(_translate("gradPlot", "..."))

from _taskManager.canvas_logic import gradient_plot
