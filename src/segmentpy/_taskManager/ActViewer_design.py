# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/segmentpy/_taskManager/ActViewer.ui',
# licensing of 'src/segmentpy/_taskManager/ActViewer.ui' applies.
#
# Created: Tue May  4 18:22:15 2021
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_actViewer(object):
    def setupUi(self, actViewer):
        actViewer.setObjectName("actViewer")
        actViewer.resize(1280, 711)
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ListTitle.sizePolicy().hasHeightForWidth())
        self.ListTitle.setSizePolicy(sizePolicy)
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
        self.label_2 = QtWidgets.QLabel(actViewer)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.weightLabel = QtWidgets.QLabel(actViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.weightLabel.sizePolicy().hasHeightForWidth())
        self.weightLabel.setSizePolicy(sizePolicy)
        self.weightLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.weightLabel.setObjectName("weightLabel")
        self.verticalLayout.addWidget(self.weightLabel)
        self.weightSlider = QtWidgets.QSlider(actViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.weightSlider.sizePolicy().hasHeightForWidth())
        self.weightSlider.setSizePolicy(sizePolicy)
        self.weightSlider.setOrientation(QtCore.Qt.Horizontal)
        self.weightSlider.setObjectName("weightSlider")
        self.verticalLayout.addWidget(self.weightSlider)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Images.sizePolicy().hasHeightForWidth())
        self.Images.setSizePolicy(sizePolicy)
        self.Images.setText("")
        self.Images.setScaledContents(True)
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
        self.label = QtWidgets.QLabel(actViewer)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.corrector = QtWidgets.QLineEdit(actViewer)
        self.corrector.setObjectName("corrector")
        self.horizontalLayout.addWidget(self.corrector)
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
        actViewer.setWindowTitle(QtWidgets.QApplication.translate("actViewer", "Form", None, -1))
        self.ckptPathLabel.setText(QtWidgets.QApplication.translate("actViewer", "Model Checkpoint path:", None, -1))
        self.ckptButton.setText(QtWidgets.QApplication.translate("actViewer", "(step 1)...", None, -1))
        self.inputPathLabel.setText(QtWidgets.QApplication.translate("actViewer", "Input path:", None, -1))
        self.inputButton.setText(QtWidgets.QApplication.translate("actViewer", "(step 2)...", None, -1))
        self.ListTitle.setText(QtWidgets.QApplication.translate("actViewer", "(step 4) Activation list:", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("actViewer", "Weight:", None, -1))
        self.weightLabel.setText(QtWidgets.QApplication.translate("actViewer", "Weight will display here", None, -1))
        self.ImgTitle.setText(QtWidgets.QApplication.translate("actViewer", "Activation:", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("actViewer", "Correction:", None, -1))
        self.corrector.setPlaceholderText(QtWidgets.QApplication.translate("actViewer", "s3: Enter here the corrector (keep it same as the training)", None, -1))
        self.load.setText(QtWidgets.QApplication.translate("actViewer", "Load model(A)", None, -1))
        self.load.setShortcut(QtWidgets.QApplication.translate("actViewer", "A", None, -1))
        self.saveButton.setText(QtWidgets.QApplication.translate("actViewer", "Save activations(S)", None, -1))
        self.saveButton.setShortcut(QtWidgets.QApplication.translate("actViewer", "S", None, -1))
        self.cancelButton.setText(QtWidgets.QApplication.translate("actViewer", "Cancel(C)", None, -1))
        self.cancelButton.setShortcut(QtWidgets.QApplication.translate("actViewer", "C", None, -1))

