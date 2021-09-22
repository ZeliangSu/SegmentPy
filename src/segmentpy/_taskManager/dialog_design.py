# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dialog.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(884, 811)
        self.gridLayout = QGridLayout(Dialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"label")

        self.verticalLayout_2.addWidget(self.label)

        self.modelComBox = QComboBox(Dialog)
        self.modelComBox.addItem("")
        self.modelComBox.addItem("")
        self.modelComBox.addItem("")
        self.modelComBox.addItem("")
        self.modelComBox.addItem("")
        self.modelComBox.setObjectName(u"modelComBox")

        self.verticalLayout_2.addWidget(self.modelComBox)

        self.label_2 = QLabel(Dialog)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_2.addWidget(self.label_2)

        self.ksize = QLineEdit(Dialog)
        self.ksize.setObjectName(u"ksize")

        self.verticalLayout_2.addWidget(self.ksize)

        self.label_3 = QLabel(Dialog)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_2.addWidget(self.label_3)

        self.nbconv = QLineEdit(Dialog)
        self.nbconv.setObjectName(u"nbconv")

        self.verticalLayout_2.addWidget(self.nbconv)

        self.label_4 = QLabel(Dialog)
        self.label_4.setObjectName(u"label_4")

        self.verticalLayout_2.addWidget(self.label_4)

        self.winsize = QLineEdit(Dialog)
        self.winsize.setObjectName(u"winsize")

        self.verticalLayout_2.addWidget(self.winsize)

        self.label_5 = QLabel(Dialog)
        self.label_5.setObjectName(u"label_5")

        self.verticalLayout_2.addWidget(self.label_5)

        self.batsize = QLineEdit(Dialog)
        self.batsize.setObjectName(u"batsize")

        self.verticalLayout_2.addWidget(self.batsize)

        self.label_6 = QLabel(Dialog)
        self.label_6.setObjectName(u"label_6")

        self.verticalLayout_2.addWidget(self.label_6)

        self.nbepoch = QLineEdit(Dialog)
        self.nbepoch.setObjectName(u"nbepoch")

        self.verticalLayout_2.addWidget(self.nbepoch)

        self.label_7 = QLabel(Dialog)
        self.label_7.setObjectName(u"label_7")

        self.verticalLayout_2.addWidget(self.label_7)

        self.batnorm = QComboBox(Dialog)
        self.batnorm.addItem("")
        self.batnorm.addItem("")
        self.batnorm.setObjectName(u"batnorm")

        self.verticalLayout_2.addWidget(self.batnorm)

        self.label_8 = QLabel(Dialog)
        self.label_8.setObjectName(u"label_8")

        self.verticalLayout_2.addWidget(self.label_8)

        self.aug = QComboBox(Dialog)
        self.aug.addItem("")
        self.aug.addItem("")
        self.aug.setObjectName(u"aug")

        self.verticalLayout_2.addWidget(self.aug)

        self.label_9 = QLabel(Dialog)
        self.label_9.setObjectName(u"label_9")

        self.verticalLayout_2.addWidget(self.label_9)

        self.dropout = QLineEdit(Dialog)
        self.dropout.setObjectName(u"dropout")

        self.verticalLayout_2.addWidget(self.dropout)

        self.label_23 = QLabel(Dialog)
        self.label_23.setObjectName(u"label_23")

        self.verticalLayout_2.addWidget(self.label_23)

        self.sampling_gap = QLineEdit(Dialog)
        self.sampling_gap.setObjectName(u"sampling_gap")

        self.verticalLayout_2.addWidget(self.sampling_gap)

        self.label_25 = QLabel(Dialog)
        self.label_25.setObjectName(u"label_25")

        self.verticalLayout_2.addWidget(self.label_25)

        self.correction = QLineEdit(Dialog)
        self.correction.setObjectName(u"correction")

        self.verticalLayout_2.addWidget(self.correction)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.line = QFrame(Dialog)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_10 = QLabel(Dialog)
        self.label_10.setObjectName(u"label_10")

        self.verticalLayout.addWidget(self.label_10)

        self.lrtype = QComboBox(Dialog)
        self.lrtype.addItem("")
        self.lrtype.addItem("")
        self.lrtype.addItem("")
        self.lrtype.setObjectName(u"lrtype")

        self.verticalLayout.addWidget(self.lrtype)

        self.label_11 = QLabel(Dialog)
        self.label_11.setObjectName(u"label_11")

        self.verticalLayout.addWidget(self.label_11)

        self.initlr = QLineEdit(Dialog)
        self.initlr.setObjectName(u"initlr")

        self.verticalLayout.addWidget(self.initlr)

        self.label_12 = QLabel(Dialog)
        self.label_12.setObjectName(u"label_12")

        self.verticalLayout.addWidget(self.label_12)

        self.kparam = QLineEdit(Dialog)
        self.kparam.setObjectName(u"kparam")

        self.verticalLayout.addWidget(self.kparam)

        self.label_13 = QLabel(Dialog)
        self.label_13.setObjectName(u"label_13")

        self.verticalLayout.addWidget(self.label_13)

        self.pparam = QLineEdit(Dialog)
        self.pparam.setObjectName(u"pparam")

        self.verticalLayout.addWidget(self.pparam)

        self.label_14 = QLabel(Dialog)
        self.label_14.setObjectName(u"label_14")

        self.verticalLayout.addWidget(self.label_14)

        self.actfn = QComboBox(Dialog)
        self.actfn.addItem("")
        self.actfn.addItem("")
        self.actfn.addItem("")
        self.actfn.addItem("")
        self.actfn.addItem("")
        self.actfn.setObjectName(u"actfn")

        self.verticalLayout.addWidget(self.actfn)

        self.label_15 = QLabel(Dialog)
        self.label_15.setObjectName(u"label_15")

        self.verticalLayout.addWidget(self.label_15)

        self.lossfn = QComboBox(Dialog)
        self.lossfn.addItem("")
        self.lossfn.addItem("")
        self.lossfn.addItem("")
        self.lossfn.setObjectName(u"lossfn")

        self.verticalLayout.addWidget(self.lossfn)

        self.label_16 = QLabel(Dialog)
        self.label_16.setObjectName(u"label_16")

        self.verticalLayout.addWidget(self.label_16)

        self.clsReg = QComboBox(Dialog)
        self.clsReg.addItem("")
        self.clsReg.addItem("")
        self.clsReg.setObjectName(u"clsReg")

        self.verticalLayout.addWidget(self.clsReg)

        self.label_17 = QLabel(Dialog)
        self.label_17.setObjectName(u"label_17")

        self.verticalLayout.addWidget(self.label_17)

        self.svsteps = QLineEdit(Dialog)
        self.svsteps.setObjectName(u"svsteps")

        self.verticalLayout.addWidget(self.svsteps)

        self.label_18 = QLabel(Dialog)
        self.label_18.setObjectName(u"label_18")

        self.verticalLayout.addWidget(self.label_18)

        self.tbstep = QLineEdit(Dialog)
        self.tbstep.setObjectName(u"tbstep")

        self.verticalLayout.addWidget(self.tbstep)

        self.label_24 = QLabel(Dialog)
        self.label_24.setObjectName(u"label_24")

        self.verticalLayout.addWidget(self.label_24)

        self.criterion = QLineEdit(Dialog)
        self.criterion.setObjectName(u"criterion")

        self.verticalLayout.addWidget(self.criterion)


        self.horizontalLayout.addLayout(self.verticalLayout)


        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.comment = QLineEdit(Dialog)
        self.comment.setObjectName(u"comment")

        self.verticalLayout_3.addWidget(self.comment)

        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout_3.addWidget(self.buttonBox)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_22 = QLabel(Dialog)
        self.label_22.setObjectName(u"label_22")

        self.horizontalLayout_5.addWidget(self.label_22)

        self.log_dir_line = QLineEdit(Dialog)
        self.log_dir_line.setObjectName(u"log_dir_line")

        self.horizontalLayout_5.addWidget(self.log_dir_line)

        self.log_dir_button = QPushButton(Dialog)
        self.log_dir_button.setObjectName(u"log_dir_button")

        self.horizontalLayout_5.addWidget(self.log_dir_button)


        self.verticalLayout_3.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_20 = QLabel(Dialog)
        self.label_20.setObjectName(u"label_20")

        self.horizontalLayout_3.addWidget(self.label_20)

        self.trn_dir_line = QLineEdit(Dialog)
        self.trn_dir_line.setObjectName(u"trn_dir_line")

        self.horizontalLayout_3.addWidget(self.trn_dir_line)

        self.trn_dir_button = QPushButton(Dialog)
        self.trn_dir_button.setObjectName(u"trn_dir_button")

        self.horizontalLayout_3.addWidget(self.trn_dir_button)


        self.verticalLayout_3.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_19 = QLabel(Dialog)
        self.label_19.setObjectName(u"label_19")

        self.horizontalLayout_2.addWidget(self.label_19)

        self.val_dir_line = QLineEdit(Dialog)
        self.val_dir_line.setObjectName(u"val_dir_line")

        self.horizontalLayout_2.addWidget(self.val_dir_line)

        self.val_dir_button = QPushButton(Dialog)
        self.val_dir_button.setObjectName(u"val_dir_button")

        self.horizontalLayout_2.addWidget(self.val_dir_button)


        self.verticalLayout_3.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_21 = QLabel(Dialog)
        self.label_21.setObjectName(u"label_21")

        self.horizontalLayout_4.addWidget(self.label_21)

        self.test_dir_line = QLineEdit(Dialog)
        self.test_dir_line.setObjectName(u"test_dir_line")

        self.horizontalLayout_4.addWidget(self.test_dir_line)

        self.test_dir_button = QPushButton(Dialog)
        self.test_dir_button.setObjectName(u"test_dir_button")

        self.horizontalLayout_4.addWidget(self.test_dir_button)


        self.verticalLayout_3.addLayout(self.horizontalLayout_4)


        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 1)


        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"model", None))
        self.modelComBox.setItemText(0, QCoreApplication.translate("Dialog", u"Unet", None))
        self.modelComBox.setItemText(1, QCoreApplication.translate("Dialog", u"LRCS-Net(shallow)", None))
        self.modelComBox.setItemText(2, QCoreApplication.translate("Dialog", u"LRCS-Net(deep)", None))
        self.modelComBox.setItemText(3, QCoreApplication.translate("Dialog", u"SegNet", None))
        self.modelComBox.setItemText(4, QCoreApplication.translate("Dialog", u"Xlearn", None))

        self.label_2.setText(QCoreApplication.translate("Dialog", u"kernel size", None))
        self.ksize.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 3, 5, 7...)", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"number of convolution minimum per layer", None))
        self.nbconv.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 16, 32,...) depand on memory", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"window size", None))
        self.winsize.setText("")
        self.winsize.setPlaceholderText(QCoreApplication.translate("Dialog", u"(80, 512...) multiple of 8 if 3x MaxPool", None))
        self.label_5.setText(QCoreApplication.translate("Dialog", u"batch size", None))
        self.batsize.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 8, 16...) depend on memory", None))
        self.label_6.setText(QCoreApplication.translate("Dialog", u"number of epoch", None))
        self.nbepoch.setText("")
        self.nbepoch.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 5, 10...) integer", None))
        self.label_7.setText(QCoreApplication.translate("Dialog", u"batch normalization", None))
        self.batnorm.setItemText(0, QCoreApplication.translate("Dialog", u"True", None))
        self.batnorm.setItemText(1, QCoreApplication.translate("Dialog", u"False", None))

        self.label_8.setText(QCoreApplication.translate("Dialog", u"data augmentation", None))
        self.aug.setItemText(0, QCoreApplication.translate("Dialog", u"True", None))
        self.aug.setItemText(1, QCoreApplication.translate("Dialog", u"False", None))

        self.label_9.setText(QCoreApplication.translate("Dialog", u"dropout probability", None))
        self.dropout.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 0.1, 0.5, 1.0...) float", None))
        self.label_23.setText(QCoreApplication.translate("Dialog", u"sampling gap", None))
        self.sampling_gap.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 5, 100, 200...) integer", None))
        self.label_25.setText(QCoreApplication.translate("Dialog", u"input image correction (beamline dependant)", None))
        self.correction.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 1e3 if input image ranges at 1e-3; 0.0039 for 0-255) float", None))
        self.label_10.setText(QCoreApplication.translate("Dialog", u"learning rate decay type", None))
        self.lrtype.setItemText(0, QCoreApplication.translate("Dialog", u"ramp", None))
        self.lrtype.setItemText(1, QCoreApplication.translate("Dialog", u"exp", None))
        self.lrtype.setItemText(2, QCoreApplication.translate("Dialog", u"constant", None))

        self.label_11.setText(QCoreApplication.translate("Dialog", u"initial learning rate", None))
        self.initlr.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 1e-4, 0.01,...) float", None))
        self.label_12.setText(QCoreApplication.translate("Dialog", u"k parameter in decay type", None))
        self.kparam.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 0.3, 0.1, 0.5) float", None))
        self.label_13.setText(QCoreApplication.translate("Dialog", u"decay periode / decay every n epoch", None))
        self.pparam.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 1, 0.5, 4) float ", None))
        self.label_14.setText(QCoreApplication.translate("Dialog", u"activation function type", None))
        self.actfn.setItemText(0, QCoreApplication.translate("Dialog", u"leaky", None))
        self.actfn.setItemText(1, QCoreApplication.translate("Dialog", u"relu", None))
        self.actfn.setItemText(2, QCoreApplication.translate("Dialog", u"sigmoid", None))
        self.actfn.setItemText(3, QCoreApplication.translate("Dialog", u"tanh", None))
        self.actfn.setItemText(4, QCoreApplication.translate("Dialog", u"custom", None))

        self.label_15.setText(QCoreApplication.translate("Dialog", u"loss function type", None))
        self.lossfn.setItemText(0, QCoreApplication.translate("Dialog", u"DSC", None))
        self.lossfn.setItemText(1, QCoreApplication.translate("Dialog", u"cross_entropy", None))
        self.lossfn.setItemText(2, QCoreApplication.translate("Dialog", u"MSE", None))

        self.label_16.setText(QCoreApplication.translate("Dialog", u"classification / regression", None))
        self.clsReg.setItemText(0, QCoreApplication.translate("Dialog", u"classification", None))
        self.clsReg.setItemText(1, QCoreApplication.translate("Dialog", u"regression", None))

        self.label_17.setText(QCoreApplication.translate("Dialog", u"save model every n steps", None))
        self.svsteps.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 500...) integer", None))
        self.label_18.setText(QCoreApplication.translate("Dialog", u"tb: gradients and weights every n steps", None))
        self.tbstep.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 50...) integer", None))
        self.label_24.setText(QCoreApplication.translate("Dialog", u"stopping criterion", None))
        self.criterion.setPlaceholderText(QCoreApplication.translate("Dialog", u"(e.g. 0.002) float", None))
        self.comment.setPlaceholderText(QCoreApplication.translate("Dialog", u"enter extra comment here", None))
        self.label_22.setText(QCoreApplication.translate("Dialog", u"save model to:", None))
        self.log_dir_line.setPlaceholderText(QCoreApplication.translate("Dialog", u"default: <SegmentPy installation folder>/logs/", None))
        self.log_dir_button.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.label_20.setText(QCoreApplication.translate("Dialog", u"Trn. ds. repo.:", None))
        self.trn_dir_line.setPlaceholderText(QCoreApplication.translate("Dialog", u"default: <SegmentPy installation folder>/train/", None))
        self.trn_dir_button.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.label_19.setText(QCoreApplication.translate("Dialog", u"Val. ds. repo.:", None))
        self.val_dir_line.setPlaceholderText(QCoreApplication.translate("Dialog", u"default: <SegmentPy installation folder>/valid/", None))
        self.val_dir_button.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.label_21.setText(QCoreApplication.translate("Dialog", u"Tst. ds. repo.:", None))
        self.test_dir_line.setPlaceholderText(QCoreApplication.translate("Dialog", u"default: <SegmentPy installation folder>/test/", None))
        self.test_dir_button.setText(QCoreApplication.translate("Dialog", u"...", None))
    # retranslateUi

