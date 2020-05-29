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
        figure_train = plt.figure(dpi=50)
        figure_val = plt.figure(dpi=50)
        self.canvas_train = canvas(figure_train)
        self.canvas_val = canvas(figure_val)

        # note: set parent to allow super class mainwindow control these canvas
        self.canvas_train.setParent(parent)
        self.canvas_val.setParent(parent)

        # init toolbar
        Toolbar1 = toolbar(self.canvas_train, self)
        Toolbar2 = toolbar(self.canvas_val, self)

        # set layout
        QVBL1 = QtWidgets.QVBoxLayout()
        QVBL1.addWidget(self.canvas_train)
        QVBL1.addWidget(Toolbar1)

        QVBL2 = QtWidgets.QVBoxLayout()
        QVBL2.addWidget(self.canvas_val)
        QVBL2.addWidget(Toolbar2)

        self.QHBL = QtWidgets.QHBoxLayout()
        self.QHBL.addLayout(QVBL1)
        self.QHBL.addLayout(QVBL2)
        self.setLayout(self.QHBL)

        # finally draw the canvas
        self.canvas_train.draw()
        self.canvas_val.draw()

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

        fig_tn = self.canvas_train.figure
        fig_val = self.canvas_val.figure
        fig_tn.clear()
        fig_val.clear()
        tn_ax = fig_tn.add_subplot(111)
        val_ax = fig_val.add_subplot(111)
        try:
            # sometimes no event conducts to NoneType
            self.setAcceptDrops(False)
            self.setCursor(Qt.WaitCursor)
            for k, v in self.paths.items():
                self.load_event(k)
                tn_ax.plot(self.curves[k][0].step, self.curves[k][0].value, label=k)
                val_ax.plot(self.curves[k][1].step, self.curves[k][1].value, label=k)
            self.setCursor(Qt.ArrowCursor)
            self.setAcceptDrops(True)

        except Exception as e:
            logger.debug(e)
            self.setCursor(Qt.ArrowCursor)
            self.setAcceptDrops(True)

        fig_tn.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)
        fig_val.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)
        self.canvas_train.draw()
        self.canvas_val.draw()

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