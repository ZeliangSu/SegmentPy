# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nodes_list_design.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_nodes_list(object):
    def setupUi(self, nodes_list):
        nodes_list.setObjectName("nodes_list")
        nodes_list.resize(347, 726)
        self.gridLayout = QtWidgets.QGridLayout(nodes_list)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(nodes_list)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.listWidget = QtWidgets.QListWidget(nodes_list)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)
        self.label_2 = QtWidgets.QLabel(nodes_list)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.act = QtWidgets.QCheckBox(nodes_list)
        self.act.setChecked(True)
        self.act.setObjectName("act")
        self.verticalLayout.addWidget(self.act)
        self.wt = QtWidgets.QCheckBox(nodes_list)
        self.wt.setObjectName("wt")
        self.verticalLayout.addWidget(self.wt)
        self.tsne = QtWidgets.QCheckBox(nodes_list)
        self.tsne.setObjectName("tsne")
        self.verticalLayout.addWidget(self.tsne)
        self.l2norm = QtWidgets.QCheckBox(nodes_list)
        self.l2norm.setObjectName("l2norm")
        self.verticalLayout.addWidget(self.l2norm)
        self.ang = QtWidgets.QCheckBox(nodes_list)
        self.ang.setObjectName("ang")
        self.verticalLayout.addWidget(self.ang)
        self.hist = QtWidgets.QCheckBox(nodes_list)
        self.hist.setObjectName("hist")
        self.verticalLayout.addWidget(self.hist)
        self.buttonBox = QtWidgets.QDialogButtonBox(nodes_list)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(nodes_list)
        self.buttonBox.accepted.connect(nodes_list.accept)
        self.buttonBox.rejected.connect(nodes_list.reject)
        QtCore.QMetaObject.connectSlotsByName(nodes_list)

    def retranslateUi(self, nodes_list):
        _translate = QtCore.QCoreApplication.translate
        nodes_list.setWindowTitle(_translate("nodes_list", "Dialog"))
        self.label.setText(_translate("nodes_list", "Select the activations to visualize"))
        self.label_2.setText(_translate("nodes_list", "Check the following analysis type"))
        self.act.setText(_translate("nodes_list", "activation"))
        self.wt.setText(_translate("nodes_list", "weight"))
        self.tsne.setText(_translate("nodes_list", "T-SNE"))
        self.l2norm.setText(_translate("nodes_list", "L2-norm of weight evolution"))
        self.ang.setText(_translate("nodes_list", "Angularity of weight evolution"))
        self.hist.setText(_translate("nodes_list", "Histogram of weight evolution"))
