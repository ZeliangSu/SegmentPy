# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/gradViewer2.ui',
# licensing of '_taskManager/gradViewer2.ui' applies.
#
# Created: Fri Nov 13 17:56:43 2020
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

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
        gradPlot.setWindowTitle(QtWidgets.QApplication.translate("gradPlot", "Form", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("gradPlot", "Select a training folder to exact the gradients", None, -1))
        self.folderButton.setText(QtWidgets.QApplication.translate("gradPlot", "...", None, -1))

from segmentpy._taskManager.canvas_logic import gradient_plot
