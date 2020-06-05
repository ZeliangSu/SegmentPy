from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as toolbar
from tensorboard_extractor import lr_curve_extractor
from util import duplicate_event
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QT5Agg')

import re


# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


class MPL(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        # back end attributes
        self.paths = {}  # {1:'/.../.../', 2:'/.../.../'}
        self.curves = {}  # {1:pd.df, 2:pd.df

        # front end
        self.setAcceptDrops(True)

        # init curves
        figure_acc_train = plt.figure(dpi=50)
        figure_lss_train = plt.figure(dpi=50)
        figure_acc_val = plt.figure(dpi=50)
        figure_lss_val = plt.figure(dpi=50)

        self.canvas_acc_train = canvas(figure_acc_train)
        self.canvas_lss_train = canvas(figure_lss_train)
        self.canvas_acc_val = canvas(figure_acc_val)
        self.canvas_lss_val = canvas(figure_lss_val)

        # note: set parent to allow super class mainwindow control these canvas
        self.canvas_acc_train.setParent(parent)
        self.canvas_lss_train.setParent(parent)
        self.canvas_acc_val.setParent(parent)
        self.canvas_lss_val.setParent(parent)

        # init toolbar
        Toolbar1 = toolbar(self.canvas_acc_train, self)
        Toolbar2 = toolbar(self.canvas_acc_val, self)
        Toolbar3 = toolbar(self.canvas_lss_train, self)
        Toolbar4 = toolbar(self.canvas_lss_val, self)

        # set layout
        QVBL1 = QtWidgets.QVBoxLayout()
        QVBL1.addWidget(self.canvas_acc_train)
        QVBL1.addWidget(Toolbar1)

        QVBL2 = QtWidgets.QVBoxLayout()
        QVBL2.addWidget(self.canvas_acc_val)
        QVBL2.addWidget(Toolbar2)

        QVBL3 = QtWidgets.QVBoxLayout()
        QVBL3.addWidget(self.canvas_lss_val)
        QVBL3.addWidget(Toolbar3)

        QVBL4 = QtWidgets.QVBoxLayout()
        QVBL4.addWidget(self.canvas_lss_val)
        QVBL4.addWidget(Toolbar4)

        # set layout
        self.QHBL = QtWidgets.QHBoxLayout()
        self.QHBL.addLayout(QVBL1)
        self.QHBL.addLayout(QVBL2)
        self.setLayout(self.QHBL)

        self.QHBL2 = QtWidgets.QHBoxLayout()
        self.QHBL2.addLayout(QVBL3)
        self.QHBL2.addLayout(QVBL4)
        self.setLayout(self.QHBL2)

        # together
        self.QVBL_all = QtWidgets.QVBoxLayout()
        self.QVBL_all.addLayout(self.QHBL)
        self.QVBL_all.addLayout(self.QHBL2)
        self.setLayout(self.QVBL_all)

        # finally draw the canvas
        self.canvas_acc_train.draw()
        self.canvas_lss_train.draw()
        self.canvas_acc_val.draw()
        self.canvas_lss_val.draw()

    def load_event(self, key):
        if key not in self.curves.keys():
            duplicate_event(self.paths[key])
            try:
                ac_tn, ac_val, ls_tn, ls_val = lr_curve_extractor(self.paths[key] + 'event/')
                self.curves[key] = [ac_tn, ac_val, ls_tn, ls_val]
            except Exception as e:
                logger.error(e)

    def plot(self):
        if self.paths.__len__() == 0:
            return

        fig_acc_tn = self.canvas_acc_train.figure
        fig_acc_val = self.canvas_acc_val.figure
        fig_lss_tn = self.canvas_lss_train.figure
        fig_lss_val = self.canvas_lss_val.figure

        fig_acc_tn.clear()
        fig_lss_tn.clear()
        fig_acc_val.clear()
        fig_lss_val.clear()

        ac_tn_ax = fig_acc_tn.add_subplot(111)
        ls_tn_ax = fig_lss_tn.add_subplot(111)
        ac_val_ax = fig_acc_val.add_subplot(111)
        ls_val_ax = fig_lss_val.add_subplot(111)

        # sometimes no event conducts to NoneType
        self.setAcceptDrops(False)
        self.setCursor(Qt.WaitCursor)
        for k, v in self.paths.items():
            self.load_event(k)
            try:
                ac_tn_ax.plot(self.curves[k][0].step, self.curves[k][0].value, label=k)
                ls_tn_ax.plot(self.curves[k][2].step, self.curves[k][2].value, label=k)
                ac_val_ax.plot(self.curves[k][1].step, self.curves[k][1].value, label=k)
                ls_val_ax.plot(self.curves[k][3].step, self.curves[k][3].value, label=k)
            except Exception as e:
                logger.debug(e)
        self.setCursor(Qt.ArrowCursor)
        self.setAcceptDrops(True)

        fig_acc_tn.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)
        fig_lss_tn.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)
        fig_acc_val.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)
        fig_lss_val.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)

        self.canvas_acc_train.draw()
        self.canvas_lss_train.draw()
        self.canvas_acc_val.draw()
        self.canvas_lss_val.draw()

    def dragEnterEvent(self, QDragEnterEvent):
        print('detected: ', QDragEnterEvent.mimeData().text())
        QDragEnterEvent.accept()  # jump to dropEvent

    def dropEvent(self, QDropEvent):
        # note: $ ends of the input
        tmp = re.search(
        '(.*hour\d+_gpu\d+)',
        QDropEvent.mimeData().text()
        )
        if tmp is not None:
            tmp = tmp.group(1)
            if not tmp.endswith('/'):
                tmp += '/'

            max_id = 0
            if self.paths.__len__() != 0:
                for i in self.paths.keys():
                    max_id = max(max_id, int(i))

            # add event folder path
            path = tmp.replace('file://', '')
            logger.debug(path)
            self.paths[max_id + 1] = path

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Oooppssss")
            msg.setInformativeText('Drag a folder which ends with format: (hour?_gpu?/) \nand which contains train/test of tf events folder!')
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        self.plot()