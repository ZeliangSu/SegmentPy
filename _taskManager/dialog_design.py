# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(560, 582)
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(11, 11, 536, 577))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.mdl = QtWidgets.QLineEdit(self.layoutWidget)
        self.mdl.setObjectName("mdl")
        self.verticalLayout_2.addWidget(self.mdl)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.ksize = QtWidgets.QLineEdit(self.layoutWidget)
        self.ksize.setObjectName("ksize")
        self.verticalLayout_2.addWidget(self.ksize)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.nbconv = QtWidgets.QLineEdit(self.layoutWidget)
        self.nbconv.setObjectName("nbconv")
        self.verticalLayout_2.addWidget(self.nbconv)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.winsize = QtWidgets.QLineEdit(self.layoutWidget)
        self.winsize.setText("")
        self.winsize.setObjectName("winsize")
        self.verticalLayout_2.addWidget(self.winsize)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.batsize = QtWidgets.QLineEdit(self.layoutWidget)
        self.batsize.setObjectName("batsize")
        self.verticalLayout_2.addWidget(self.batsize)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.nbepoch = QtWidgets.QLineEdit(self.layoutWidget)
        self.nbepoch.setText("")
        self.nbepoch.setObjectName("nbepoch")
        self.verticalLayout_2.addWidget(self.nbepoch)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.batnorm = QtWidgets.QComboBox(self.layoutWidget)
        self.batnorm.setObjectName("batnorm")
        self.batnorm.addItem("")
        self.batnorm.addItem("")
        self.verticalLayout_2.addWidget(self.batnorm)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.aug = QtWidgets.QComboBox(self.layoutWidget)
        self.aug.setObjectName("aug")
        self.aug.addItem("")
        self.aug.addItem("")
        self.verticalLayout_2.addWidget(self.aug)
        self.label_9 = QtWidgets.QLabel(self.layoutWidget)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_2.addWidget(self.label_9)
        self.dropout = QtWidgets.QLineEdit(self.layoutWidget)
        self.dropout.setObjectName("dropout")
        self.verticalLayout_2.addWidget(self.dropout)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.line = QtWidgets.QFrame(self.layoutWidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget)
        self.label_10.setObjectName("label_10")
        self.verticalLayout.addWidget(self.label_10)
        self.lrtype = QtWidgets.QComboBox(self.layoutWidget)
        self.lrtype.setObjectName("lrtype")
        self.lrtype.addItem("")
        self.lrtype.addItem("")
        self.lrtype.addItem("")
        self.verticalLayout.addWidget(self.lrtype)
        self.label_11 = QtWidgets.QLabel(self.layoutWidget)
        self.label_11.setObjectName("label_11")
        self.verticalLayout.addWidget(self.label_11)
        self.initlr = QtWidgets.QLineEdit(self.layoutWidget)
        self.initlr.setObjectName("initlr")
        self.verticalLayout.addWidget(self.initlr)
        self.label_12 = QtWidgets.QLabel(self.layoutWidget)
        self.label_12.setObjectName("label_12")
        self.verticalLayout.addWidget(self.label_12)
        self.kparam = QtWidgets.QLineEdit(self.layoutWidget)
        self.kparam.setObjectName("kparam")
        self.verticalLayout.addWidget(self.kparam)
        self.label_13 = QtWidgets.QLabel(self.layoutWidget)
        self.label_13.setObjectName("label_13")
        self.verticalLayout.addWidget(self.label_13)
        self.pparam = QtWidgets.QLineEdit(self.layoutWidget)
        self.pparam.setObjectName("pparam")
        self.verticalLayout.addWidget(self.pparam)
        self.label_14 = QtWidgets.QLabel(self.layoutWidget)
        self.label_14.setObjectName("label_14")
        self.verticalLayout.addWidget(self.label_14)
        self.actfn = QtWidgets.QComboBox(self.layoutWidget)
        self.actfn.setObjectName("actfn")
        self.actfn.addItem("")
        self.actfn.addItem("")
        self.actfn.addItem("")
        self.actfn.addItem("")
        self.actfn.addItem("")
        self.verticalLayout.addWidget(self.actfn)
        self.label_15 = QtWidgets.QLabel(self.layoutWidget)
        self.label_15.setObjectName("label_15")
        self.verticalLayout.addWidget(self.label_15)
        self.lossfn = QtWidgets.QComboBox(self.layoutWidget)
        self.lossfn.setObjectName("lossfn")
        self.lossfn.addItem("")
        self.lossfn.addItem("")
        self.lossfn.addItem("")
        self.verticalLayout.addWidget(self.lossfn)
        self.label_16 = QtWidgets.QLabel(self.layoutWidget)
        self.label_16.setObjectName("label_16")
        self.verticalLayout.addWidget(self.label_16)
        self.clsReg = QtWidgets.QComboBox(self.layoutWidget)
        self.clsReg.setObjectName("clsReg")
        self.clsReg.addItem("")
        self.clsReg.addItem("")
        self.verticalLayout.addWidget(self.clsReg)
        self.label_17 = QtWidgets.QLabel(self.layoutWidget)
        self.label_17.setObjectName("label_17")
        self.verticalLayout.addWidget(self.label_17)
        self.svsteps = QtWidgets.QLineEdit(self.layoutWidget)
        self.svsteps.setObjectName("svsteps")
        self.verticalLayout.addWidget(self.svsteps)
        self.label_18 = QtWidgets.QLabel(self.layoutWidget)
        self.label_18.setObjectName("label_18")
        self.verticalLayout.addWidget(self.label_18)
        self.tbstep = QtWidgets.QLineEdit(self.layoutWidget)
        self.tbstep.setObjectName("tbstep")
        self.verticalLayout.addWidget(self.tbstep)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.comment = QtWidgets.QLineEdit(self.layoutWidget)
        self.comment.setObjectName("comment")
        self.verticalLayout_3.addWidget(self.comment)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_3.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "model name"))
        self.mdl.setPlaceholderText(_translate("Dialog", "(e.g. LRCS, Unet, custom...)"))
        self.label_2.setText(_translate("Dialog", "kernel size"))
        self.ksize.setPlaceholderText(_translate("Dialog", "(e.g. 3, 5, 7...)"))
        self.label_3.setText(_translate("Dialog", "number of convolution minimum per layer"))
        self.nbconv.setPlaceholderText(_translate("Dialog", "(e.g. 16, 32,...) depand on memory"))
        self.label_4.setText(_translate("Dialog", "window size"))
        self.winsize.setPlaceholderText(_translate("Dialog", "(80, 512...) multiple of 8 if 3x MaxPool"))
        self.label_5.setText(_translate("Dialog", "batch size"))
        self.batsize.setPlaceholderText(_translate("Dialog", "(e.g. 8, 16...) depend on memory"))
        self.label_6.setText(_translate("Dialog", "number of epoch"))
        self.nbepoch.setPlaceholderText(_translate("Dialog", "(e.g. 5, 10...) integer"))
        self.label_7.setText(_translate("Dialog", "batch normalization"))
        self.batnorm.setItemText(0, _translate("Dialog", "True"))
        self.batnorm.setItemText(1, _translate("Dialog", "False"))
        self.label_8.setText(_translate("Dialog", "data augmentation"))
        self.aug.setItemText(0, _translate("Dialog", "True"))
        self.aug.setItemText(1, _translate("Dialog", "False"))
        self.label_9.setText(_translate("Dialog", "dropout probability"))
        self.dropout.setPlaceholderText(_translate("Dialog", "(e.g. 0.1, 0.5, 1.0...) float"))
        self.label_10.setText(_translate("Dialog", "learning rate decay type"))
        self.lrtype.setItemText(0, _translate("Dialog", "ramp"))
        self.lrtype.setItemText(1, _translate("Dialog", "exp"))
        self.lrtype.setItemText(2, _translate("Dialog", "constant"))
        self.label_11.setText(_translate("Dialog", "initial learning rate"))
        self.initlr.setPlaceholderText(_translate("Dialog", "(e.g. 1e-4, 0.01,...) float"))
        self.label_12.setText(_translate("Dialog", "k parameter in decay type"))
        self.kparam.setPlaceholderText(_translate("Dialog", "(e.g. 0.3, 0.1, 0.5) float"))
        self.label_13.setText(_translate("Dialog", "decay periode / decay every n epoch"))
        self.pparam.setPlaceholderText(_translate("Dialog", "(e.g. 1, 0.5, 4) float "))
        self.label_14.setText(_translate("Dialog", "activation function type"))
        self.actfn.setItemText(0, _translate("Dialog", "relu"))
        self.actfn.setItemText(1, _translate("Dialog", "leaky"))
        self.actfn.setItemText(2, _translate("Dialog", "sigmoid"))
        self.actfn.setItemText(3, _translate("Dialog", "tanh"))
        self.actfn.setItemText(4, _translate("Dialog", "custom"))
        self.label_15.setText(_translate("Dialog", "loss function type"))
        self.lossfn.setItemText(0, _translate("Dialog", "DSC"))
        self.lossfn.setItemText(1, _translate("Dialog", "cross_entropy"))
        self.lossfn.setItemText(2, _translate("Dialog", "MSE"))
        self.label_16.setText(_translate("Dialog", "classification / regression"))
        self.clsReg.setItemText(0, _translate("Dialog", "classification"))
        self.clsReg.setItemText(1, _translate("Dialog", "regression"))
        self.label_17.setText(_translate("Dialog", "save model every n steps"))
        self.svsteps.setPlaceholderText(_translate("Dialog", "(e.g. 500...) integer"))
        self.label_18.setText(_translate("Dialog", "tb: gradients and weights every n steps"))
        self.tbstep.setPlaceholderText(_translate("Dialog", "(e.g. 50...) integer"))
        self.comment.setPlaceholderText(_translate("Dialog", "enter extra comment here"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

