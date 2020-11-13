# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/nodes_list_design.ui',
# licensing of '_taskManager/nodes_list_design.ui' applies.
#
# Created: Fri Nov 13 17:58:20 2020
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

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
        font.setWeight(75)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.listWidget = QtWidgets.QListWidget(nodes_list)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)
        self.label_2 = QtWidgets.QLabel(nodes_list)
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
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
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), nodes_list.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), nodes_list.reject)
        QtCore.QMetaObject.connectSlotsByName(nodes_list)

    def retranslateUi(self, nodes_list):
        nodes_list.setWindowTitle(QtWidgets.QApplication.translate("nodes_list", "Dialog", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("nodes_list", "Select the activations to visualize", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("nodes_list", "Check the following analysis type", None, -1))
        self.act.setText(QtWidgets.QApplication.translate("nodes_list", "activation", None, -1))
        self.wt.setText(QtWidgets.QApplication.translate("nodes_list", "weight", None, -1))
        self.tsne.setText(QtWidgets.QApplication.translate("nodes_list", "T-SNE", None, -1))
        self.l2norm.setText(QtWidgets.QApplication.translate("nodes_list", "L2-norm of weight evolution", None, -1))
        self.ang.setText(QtWidgets.QApplication.translate("nodes_list", "Angularity of weight evolution", None, -1))
        self.hist.setText(QtWidgets.QApplication.translate("nodes_list", "Histogram of weight evolution", None, -1))

