# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/augmentationViewer.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_augViewer(object):
    def setupUi(self, augViewer):
        augViewer.setObjectName("augViewer")
        augViewer.resize(1063, 605)
        self.gridLayout = QtWidgets.QGridLayout(augViewer)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(augViewer)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(augViewer)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.raw = QtWidgets.QLabel(augViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.raw.sizePolicy().hasHeightForWidth())
        self.raw.setSizePolicy(sizePolicy)
        self.raw.setMinimumSize(QtCore.QSize(500, 500))
        self.raw.setText("")
        self.raw.setObjectName("raw")
        self.horizontalLayout.addWidget(self.raw)
        self.line = QtWidgets.QFrame(augViewer)
        self.line.setLineWidth(8)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.aug = QtWidgets.QLabel(augViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.aug.sizePolicy().hasHeightForWidth())
        self.aug.setSizePolicy(sizePolicy)
        self.aug.setMinimumSize(QtCore.QSize(500, 500))
        self.aug.setText("")
        self.aug.setObjectName("aug")
        self.horizontalLayout.addWidget(self.aug)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.next = QtWidgets.QPushButton(augViewer)
        self.next.setObjectName("next")
        self.verticalLayout.addWidget(self.next)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(augViewer)
        QtCore.QMetaObject.connectSlotsByName(augViewer)

    def retranslateUi(self, augViewer):
        _translate = QtCore.QCoreApplication.translate
        augViewer.setWindowTitle(_translate("augViewer", "Dialog"))
        self.label_3.setText(_translate("augViewer", "Raw"))
        self.label_4.setText(_translate("augViewer", "Augmented"))
        self.next.setText(_translate("augViewer", "Next(Q)"))
        self.next.setShortcut(_translate("augViewer", "Q"))
