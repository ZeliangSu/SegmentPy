# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/gradViewer_design.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_grad_extractor(object):
    def setupUi(self, grad_extractor):
        grad_extractor.setObjectName("grad_extractor")
        grad_extractor.resize(369, 122)
        self.gridLayout = QtWidgets.QGridLayout(grad_extractor)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pathLine = QtWidgets.QLineEdit(grad_extractor)
        self.pathLine.setObjectName("pathLine")
        self.horizontalLayout.addWidget(self.pathLine)
        self.pathButton = QtWidgets.QPushButton(grad_extractor)
        self.pathButton.setObjectName("pathButton")
        self.horizontalLayout.addWidget(self.pathButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(grad_extractor)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(grad_extractor)
        self.buttonBox.accepted.connect(grad_extractor.accept)
        self.buttonBox.rejected.connect(grad_extractor.reject)
        QtCore.QMetaObject.connectSlotsByName(grad_extractor)

    def retranslateUi(self, grad_extractor):
        _translate = QtCore.QCoreApplication.translate
        grad_extractor.setWindowTitle(_translate("grad_extractor", "Dialog"))
        self.pathButton.setText(_translate("grad_extractor", "..."))
