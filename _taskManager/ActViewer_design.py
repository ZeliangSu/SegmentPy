# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/ActViewer.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_actViewer(object):
    def setupUi(self, actViewer):
        actViewer.setObjectName("actViewer")
        actViewer.resize(1279, 912)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(actViewer.sizePolicy().hasHeightForWidth())
        actViewer.setSizePolicy(sizePolicy)
        self.gridLayout_2 = QtWidgets.QGridLayout(actViewer)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.ckptPathLabel = QtWidgets.QLabel(actViewer)
        self.ckptPathLabel.setObjectName("ckptPathLabel")
        self.horizontalLayout_3.addWidget(self.ckptPathLabel)
        self.ckptPathLine = QtWidgets.QLineEdit(actViewer)
        self.ckptPathLine.setObjectName("ckptPathLine")
        self.horizontalLayout_3.addWidget(self.ckptPathLine)
        self.ckptButton = QtWidgets.QToolButton(actViewer)
        self.ckptButton.setObjectName("ckptButton")
        self.horizontalLayout_3.addWidget(self.ckptButton)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.inputPathLabel = QtWidgets.QLabel(actViewer)
        self.inputPathLabel.setObjectName("inputPathLabel")
        self.horizontalLayout_4.addWidget(self.inputPathLabel)
        self.inputPathLine = QtWidgets.QLineEdit(actViewer)
        self.inputPathLine.setObjectName("inputPathLine")
        self.horizontalLayout_4.addWidget(self.inputPathLine)
        self.inputButton = QtWidgets.QToolButton(actViewer)
        self.inputButton.setObjectName("inputButton")
        self.horizontalLayout_4.addWidget(self.inputButton)
        self.gridLayout.addLayout(self.horizontalLayout_4, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.ListTitle = QtWidgets.QLabel(actViewer)
        self.ListTitle.setMaximumSize(QtCore.QSize(300, 16777215))
        self.ListTitle.setObjectName("ListTitle")
        self.verticalLayout.addWidget(self.ListTitle)
        self.actList = QtWidgets.QListWidget(actViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.actList.sizePolicy().hasHeightForWidth())
        self.actList.setSizePolicy(sizePolicy)
        self.actList.setMaximumSize(QtCore.QSize(300, 16777215))
        self.actList.setObjectName("actList")
        self.verticalLayout.addWidget(self.actList)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.ImgTitle = QtWidgets.QLabel(actViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImgTitle.sizePolicy().hasHeightForWidth())
        self.ImgTitle.setSizePolicy(sizePolicy)
        self.ImgTitle.setObjectName("ImgTitle")
        self.verticalLayout_2.addWidget(self.ImgTitle)
        self.Images = QtWidgets.QLabel(actViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Images.sizePolicy().hasHeightForWidth())
        self.Images.setSizePolicy(sizePolicy)
        self.Images.setText("")
        self.Images.setObjectName("Images")
        self.verticalLayout_2.addWidget(self.Images)
        self.actSlider = QtWidgets.QSlider(actViewer)
        self.actSlider.setOrientation(QtCore.Qt.Horizontal)
        self.actSlider.setObjectName("actSlider")
        self.verticalLayout_2.addWidget(self.actSlider)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.load = QtWidgets.QPushButton(actViewer)
        self.load.setObjectName("load")
        self.horizontalLayout.addWidget(self.load)
        self.saveButton = QtWidgets.QPushButton(actViewer)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout.addWidget(self.saveButton)
        self.cancelButton = QtWidgets.QPushButton(actViewer)
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout.addWidget(self.cancelButton)
        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(actViewer)
        QtCore.QMetaObject.connectSlotsByName(actViewer)

    def retranslateUi(self, actViewer):
        _translate = QtCore.QCoreApplication.translate
        actViewer.setWindowTitle(_translate("actViewer", "Form"))
        self.ckptPathLabel.setText(_translate("actViewer", "Model Checkpoint path:"))
        self.ckptButton.setText(_translate("actViewer", "..."))
        self.inputPathLabel.setText(_translate("actViewer", "Input path:"))
        self.inputButton.setText(_translate("actViewer", "..."))
        self.ListTitle.setText(_translate("actViewer", "Activation list:"))
        self.ImgTitle.setText(_translate("actViewer", "Activation:"))
        self.load.setText(_translate("actViewer", "Load model(A)"))
        self.load.setShortcut(_translate("actViewer", "A"))
        self.saveButton.setText(_translate("actViewer", "Save activations(S)"))
        self.saveButton.setShortcut(_translate("actViewer", "S"))
        self.cancelButton.setText(_translate("actViewer", "Cancel(C)"))
        self.cancelButton.setShortcut(_translate("actViewer", "C"))
