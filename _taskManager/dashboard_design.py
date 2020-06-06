# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/dashboard.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dashboard(object):
    def setupUi(self, dashboard):
        dashboard.setObjectName("dashboard")
        dashboard.resize(1226, 1091)
        dashboard.setLayoutDirection(QtCore.Qt.LeftToRight)
        dashboard.setSizeGripEnabled(False)
        dashboard.setModal(False)
        self.gridLayout = QtWidgets.QGridLayout(dashboard)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(dashboard)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(dashboard)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.mplwidget = MPL(dashboard)
        self.mplwidget.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mplwidget.sizePolicy().hasHeightForWidth())
        self.mplwidget.setSizePolicy(sizePolicy)
        self.mplwidget.setMinimumSize(QtCore.QSize(1200, 600))
        self.mplwidget.setSizeIncrement(QtCore.QSize(0, 0))
        self.mplwidget.setBaseSize(QtCore.QSize(0, 0))
        self.mplwidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mplwidget.setObjectName("mplwidget")
        self.verticalLayout.addWidget(self.mplwidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.refresh_button = QtWidgets.QPushButton(dashboard)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.refresh_button.sizePolicy().hasHeightForWidth())
        self.refresh_button.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("_taskManager/reload.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.refresh_button.setIcon(icon)
        self.refresh_button.setIconSize(QtCore.QSize(16, 32))
        self.refresh_button.setObjectName("refresh_button")
        self.horizontalLayout.addWidget(self.refresh_button)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.live_button = QtWidgets.QPushButton(dashboard)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.live_button.sizePolicy().hasHeightForWidth())
        self.live_button.setSizePolicy(sizePolicy)
        self.live_button.setMaximumSize(QtCore.QSize(16777215, 48))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("_taskManager/live.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.live_button.setIcon(icon1)
        self.live_button.setIconSize(QtCore.QSize(32, 32))
        self.live_button.setCheckable(True)
        self.live_button.setObjectName("live_button")
        self.horizontalLayout.addWidget(self.live_button)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.save_button = QtWidgets.QPushButton(dashboard)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_button.sizePolicy().hasHeightForWidth())
        self.save_button.setSizePolicy(sizePolicy)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("_taskManager/floppy-disk.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_button.setIcon(icon2)
        self.save_button.setIconSize(QtCore.QSize(16, 32))
        self.save_button.setObjectName("save_button")
        self.horizontalLayout.addWidget(self.save_button)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(dashboard)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.save_path_box = QtWidgets.QLineEdit(dashboard)
        self.save_path_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.save_path_box.setDragEnabled(False)
        self.save_path_box.setObjectName("save_path_box")
        self.horizontalLayout_3.addWidget(self.save_path_box)
        self.folder_button = QtWidgets.QPushButton(dashboard)
        self.folder_button.setMaximumSize(QtCore.QSize(54, 32))
        self.folder_button.setObjectName("folder_button")
        self.horizontalLayout_3.addWidget(self.folder_button)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(dashboard)
        QtCore.QMetaObject.connectSlotsByName(dashboard)

    def retranslateUi(self, dashboard):
        _translate = QtCore.QCoreApplication.translate
        dashboard.setWindowTitle(_translate("dashboard", "Dialog"))
        self.label.setText(_translate("dashboard", "Training"))
        self.label_2.setText(_translate("dashboard", "Validation"))
        self.refresh_button.setText(_translate("dashboard", "Refresh(R)"))
        self.refresh_button.setShortcut(_translate("dashboard", "R"))
        self.live_button.setText(_translate("dashboard", "Live(L)"))
        self.live_button.setShortcut(_translate("dashboard", "L"))
        self.save_button.setText(_translate("dashboard", "Save(S)"))
        self.save_button.setShortcut(_translate("dashboard", "S"))
        self.label_3.setText(_translate("dashboard", "Save path :"))
        self.folder_button.setText(_translate("dashboard", "..."))
from _taskManager.canvas_logic import MPL
