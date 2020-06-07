# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/progressBar.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_standalone_pBar(object):
    def setupUi(self, standalone_pBar):
        standalone_pBar.setObjectName("standalone_pBar")
        standalone_pBar.resize(526, 68)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(standalone_pBar.sizePolicy().hasHeightForWidth())
        standalone_pBar.setSizePolicy(sizePolicy)
        self.gridLayout = QtWidgets.QGridLayout(standalone_pBar)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.processing_title = QtWidgets.QLabel(standalone_pBar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.processing_title.sizePolicy().hasHeightForWidth())
        self.processing_title.setSizePolicy(sizePolicy)
        self.processing_title.setObjectName("processing_title")
        self.verticalLayout.addWidget(self.processing_title)
        self.progressBar = QtWidgets.QProgressBar(standalone_pBar)
        self.progressBar.setMinimumSize(QtCore.QSize(500, 0))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(standalone_pBar)
        QtCore.QMetaObject.connectSlotsByName(standalone_pBar)

    def retranslateUi(self, standalone_pBar):
        _translate = QtCore.QCoreApplication.translate
        standalone_pBar.setWindowTitle(_translate("standalone_pBar", "Dialog"))
        self.processing_title.setText(_translate("standalone_pBar", "Loading..."))
