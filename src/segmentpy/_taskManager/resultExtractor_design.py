# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/segmentpy/_taskManager/resultExtractor.ui',
# licensing of 'src/segmentpy/_taskManager/resultExtractor.ui' applies.
#
# Created: Tue May  4 00:56:39 2021
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Extractor(object):
    def setupUi(self, Extractor):
        Extractor.setObjectName("Extractor")
        Extractor.resize(545, 402)
        self.gridLayout = QtWidgets.QGridLayout(Extractor)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.MPLwidget = sortResult(Extractor)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MPLwidget.sizePolicy().hasHeightForWidth())
        self.MPLwidget.setSizePolicy(sizePolicy)
        self.MPLwidget.setObjectName("MPLwidget")
        self.verticalLayout.addWidget(self.MPLwidget)
        self.legendLabel = QtWidgets.QLabel(Extractor)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.legendLabel.sizePolicy().hasHeightForWidth())
        self.legendLabel.setSizePolicy(sizePolicy)
        self.legendLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.legendLabel.setObjectName("legendLabel")
        self.verticalLayout.addWidget(self.legendLabel)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.extractButton = QtWidgets.QPushButton(Extractor)
        self.extractButton.setObjectName("extractButton")
        self.horizontalLayout.addWidget(self.extractButton)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.barLabel = QtWidgets.QLabel(Extractor)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.barLabel.sizePolicy().hasHeightForWidth())
        self.barLabel.setSizePolicy(sizePolicy)
        self.barLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.barLabel.setObjectName("barLabel")
        self.verticalLayout_2.addWidget(self.barLabel)
        self.progressBar = QtWidgets.QProgressBar(Extractor)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.colorButton = QtWidgets.QPushButton(Extractor)
        self.colorButton.setObjectName("colorButton")
        self.verticalLayout_3.addWidget(self.colorButton)
        self.saveButton = QtWidgets.QPushButton(Extractor)
        self.saveButton.setObjectName("saveButton")
        self.verticalLayout_3.addWidget(self.saveButton)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Extractor)
        QtCore.QMetaObject.connectSlotsByName(Extractor)

    def retranslateUi(self, Extractor):
        Extractor.setWindowTitle(QtWidgets.QApplication.translate("Extractor", "Form", None, -1))
        self.legendLabel.setText(QtWidgets.QApplication.translate("Extractor", "Legend", None, -1))
        self.extractButton.setText(QtWidgets.QApplication.translate("Extractor", "Extract", None, -1))
        self.barLabel.setText(QtWidgets.QApplication.translate("Extractor", "0%", None, -1))
        self.colorButton.setText(QtWidgets.QApplication.translate("Extractor", "colors", None, -1))
        self.saveButton.setText(QtWidgets.QApplication.translate("Extractor", "Save as img and csv", None, -1))

from segmentpy._taskManager.canvas_logic import sortResult
