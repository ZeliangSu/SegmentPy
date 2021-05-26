# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '_taskManager/volumes_viewer.ui',
# licensing of '_taskManager/volumes_viewer.ui' applies.
#
# Created: Fri Nov 13 18:00:59 2020
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_volViewer(object):
    def setupUi(self, volViewer):
        volViewer.setObjectName("volViewer")
        volViewer.resize(1581, 604)
        volViewer.setMinimumSize(QtCore.QSize(0, 0))
        self.gridLayout = QtWidgets.QGridLayout(volViewer)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.vol1 = QtWidgets.QLabel(volViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vol1.sizePolicy().hasHeightForWidth())
        self.vol1.setSizePolicy(sizePolicy)
        self.vol1.setMinimumSize(QtCore.QSize(500, 500))
        self.vol1.setAcceptDrops(True)
        self.vol1.setFrameShape(QtWidgets.QFrame.Box)
        self.vol1.setAlignment(QtCore.Qt.AlignCenter)
        self.vol1.setObjectName("vol1")
        self.horizontalLayout.addWidget(self.vol1)
        self.vol2 = QtWidgets.QLabel(volViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vol2.sizePolicy().hasHeightForWidth())
        self.vol2.setSizePolicy(sizePolicy)
        self.vol2.setMinimumSize(QtCore.QSize(500, 500))
        self.vol2.setAcceptDrops(True)
        self.vol2.setFrameShape(QtWidgets.QFrame.Box)
        self.vol2.setMidLineWidth(0)
        self.vol2.setAlignment(QtCore.Qt.AlignCenter)
        self.vol2.setObjectName("vol2")
        self.horizontalLayout.addWidget(self.vol2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.actual_slide = QtWidgets.QLCDNumber(volViewer)
        self.actual_slide.setMidLineWidth(0)
        self.actual_slide.setSmallDecimalPoint(False)
        self.actual_slide.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.actual_slide.setObjectName("actual_slide")
        self.horizontalLayout_2.addWidget(self.actual_slide)
        self.label = QtWidgets.QLabel(volViewer)
        font = QtGui.QFont()
        font.setWeight(75)
        font.setItalic(False)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.total_slides = QtWidgets.QLCDNumber(volViewer)
        self.total_slides.setFrameShadow(QtWidgets.QFrame.Raised)
        self.total_slides.setLineWidth(1)
        self.total_slides.setMidLineWidth(0)
        self.total_slides.setSmallDecimalPoint(False)
        self.total_slides.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.total_slides.setObjectName("total_slides")
        self.horizontalLayout_2.addWidget(self.total_slides)
        self.label_2 = QtWidgets.QLabel(volViewer)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.zoom_button = QtWidgets.QPushButton(volViewer)
        self.zoom_button.setMinimumSize(QtCore.QSize(0, 36))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("zoom-in.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.zoom_button.setIcon(icon)
        self.zoom_button.setCheckable(True)
        self.zoom_button.setChecked(False)
        self.zoom_button.setObjectName("zoom_button")
        self.horizontalLayout_2.addWidget(self.zoom_button)
        self.dezoom_button = QtWidgets.QPushButton(volViewer)
        self.dezoom_button.setMinimumSize(QtCore.QSize(0, 36))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("zoom-out.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.dezoom_button.setIcon(icon1)
        self.dezoom_button.setObjectName("dezoom_button")
        self.horizontalLayout_2.addWidget(self.dezoom_button)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.Slider = QtWidgets.QScrollBar(volViewer)
        self.Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Slider.setObjectName("Slider")
        self.verticalLayout.addWidget(self.Slider)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.accumulated_value1 = QtWidgets.QLabel(volViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.accumulated_value1.sizePolicy().hasHeightForWidth())
        self.accumulated_value1.setSizePolicy(sizePolicy)
        self.accumulated_value1.setMinimumSize(QtCore.QSize(100, 100))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.accumulated_value1.setFont(font)
        self.accumulated_value1.setAlignment(QtCore.Qt.AlignCenter)
        self.accumulated_value1.setObjectName("accumulated_value1")
        self.verticalLayout_3.addWidget(self.accumulated_value1)
        self.accum_plot1 = volFracPlotter(volViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.accum_plot1.sizePolicy().hasHeightForWidth())
        self.accum_plot1.setSizePolicy(sizePolicy)
        self.accum_plot1.setMinimumSize(QtCore.QSize(300, 300))
        self.accum_plot1.setObjectName("accum_plot1")
        self.verticalLayout_3.addWidget(self.accum_plot1)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.line = QtWidgets.QFrame(volViewer)
        self.line.setLineWidth(3)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_3.addWidget(self.line)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.accumulated_value2 = QtWidgets.QLabel(volViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.accumulated_value2.sizePolicy().hasHeightForWidth())
        self.accumulated_value2.setSizePolicy(sizePolicy)
        self.accumulated_value2.setMinimumSize(QtCore.QSize(100, 100))
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.accumulated_value2.setFont(font)
        self.accumulated_value2.setAlignment(QtCore.Qt.AlignCenter)
        self.accumulated_value2.setObjectName("accumulated_value2")
        self.verticalLayout_4.addWidget(self.accumulated_value2)
        self.accum_plot2 = volFracPlotter(volViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.accum_plot2.sizePolicy().hasHeightForWidth())
        self.accum_plot2.setSizePolicy(sizePolicy)
        self.accum_plot2.setMinimumSize(QtCore.QSize(300, 300))
        self.accum_plot2.setObjectName("accum_plot2")
        self.verticalLayout_4.addWidget(self.accum_plot2)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.gridLayout.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)

        self.retranslateUi(volViewer)
        QtCore.QMetaObject.connectSlotsByName(volViewer)

    def retranslateUi(self, volViewer):
        volViewer.setWindowTitle(QtWidgets.QApplication.translate("volViewer", "Dialog", None, -1))
        self.vol1.setText(QtWidgets.QApplication.translate("volViewer", "Drop a folder of images here", None, -1))
        self.vol2.setText(QtWidgets.QApplication.translate("volViewer", "Drop a folder of images here", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("volViewer", "/", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("volViewer", "Slides", None, -1))
        self.zoom_button.setText(QtWidgets.QApplication.translate("volViewer", "Zoom-in(A)", None, -1))
        self.zoom_button.setShortcut(QtWidgets.QApplication.translate("volViewer", "A", None, -1))
        self.dezoom_button.setText(QtWidgets.QApplication.translate("volViewer", "Zoom-out(S)", None, -1))
        self.dezoom_button.setShortcut(QtWidgets.QApplication.translate("volViewer", "S", None, -1))
        self.accumulated_value1.setText(QtWidgets.QApplication.translate("volViewer", "Total voxels: Acc. Vol. Frac.:", None, -1))
        self.accumulated_value2.setText(QtWidgets.QApplication.translate("volViewer", "Total voxels: Acc. Vol. Frac.:", None, -1))

from segmentpy._taskManager.canvas_logic import volFracPlotter