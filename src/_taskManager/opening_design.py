# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/opening.ui',
# licensing of '_taskManager/opening.ui' applies.
#
# Created: Fri Nov 13 17:58:50 2020
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(571, 355)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("LRCS.jpg"))
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.log = QtWidgets.QLabel(Dialog)
        self.log.setObjectName("log")
        self.gridLayout.addWidget(self.log, 1, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtWidgets.QApplication.translate("Dialog", "Dialog", None, -1))
        self.log.setText(QtWidgets.QApplication.translate("Dialog", "Logging...", None, -1))

