# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/resumeDialog_designe.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(408, 145)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.ckptLabel = QtWidgets.QLabel(Dialog)
        self.ckptLabel.setObjectName("ckptLabel")
        self.horizontalLayout.addWidget(self.ckptLabel)
        self.ckptLine = QtWidgets.QLineEdit(Dialog)
        self.ckptLine.setObjectName("ckptLine")
        self.horizontalLayout.addWidget(self.ckptLine)
        self.ckptButton = QtWidgets.QPushButton(Dialog)
        self.ckptButton.setObjectName("ckptButton")
        self.horizontalLayout.addWidget(self.ckptButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.epochLabel = QtWidgets.QLabel(Dialog)
        self.epochLabel.setObjectName("epochLabel")
        self.horizontalLayout_2.addWidget(self.epochLabel)
        self.epochLine = QtWidgets.QLineEdit(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.epochLine.sizePolicy().hasHeightForWidth())
        self.epochLine.setSizePolicy(sizePolicy)
        self.epochLine.setObjectName("epochLine")
        self.horizontalLayout_2.addWidget(self.epochLine)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.commentLabel = QtWidgets.QLabel(Dialog)
        self.commentLabel.setObjectName("commentLabel")
        self.horizontalLayout_3.addWidget(self.commentLabel)
        self.commentLine = QtWidgets.QLineEdit(Dialog)
        self.commentLine.setObjectName("commentLine")
        self.horizontalLayout_3.addWidget(self.commentLine)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.ckptLabel.setText(_translate("Dialog", "Checkpoint"))
        self.ckptButton.setText(_translate("Dialog", "..."))
        self.epochLabel.setText(_translate("Dialog", "Extra epoch"))
        self.commentLabel.setText(_translate("Dialog", "Comment"))
