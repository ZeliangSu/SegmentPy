from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as toolbar
from tensorboard_extractor import lr_curve_extractor

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QT5Agg')

import re
import pandas as pd
import numpy as np
import string

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
        QVBL1.addWidget(self.canvas_lss_train)
        QVBL1.addWidget(Toolbar3)

        QVBL2 = QtWidgets.QVBoxLayout()
        QVBL2.addWidget(self.canvas_acc_val)
        QVBL2.addWidget(Toolbar2)
        QVBL2.addWidget(self.canvas_lss_val)
        QVBL2.addWidget(Toolbar4)

        # set layout
        self.QHBL = QtWidgets.QHBoxLayout()
        self.QHBL.addLayout(QVBL1)
        self.QHBL.addLayout(QVBL2)
        self.setLayout(self.QHBL)

        # finally draw the canvas
        self.canvas_acc_train.draw()
        self.canvas_lss_train.draw()
        self.canvas_acc_val.draw()
        self.canvas_lss_val.draw()

    def load_event(self, key):
        if key not in self.curves.keys():
            try:
                ac_tn, _, ls_tn, _ = lr_curve_extractor(self.paths[key] + 'train/')
                _, ac_val, _, ls_val = lr_curve_extractor(self.paths[key] + 'test/')
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

        for k, v in self.paths.items():
            self.load_event(k)
            try:
                ac_tn_ax.plot(self.curves[k][0].step, self.curves[k][0].value, label=k)
                ls_tn_ax.plot(self.curves[k][2].step, self.curves[k][2].value, label=k)
                ac_val_ax.plot(self.curves[k][1].step, self.curves[k][1].value, label=k)
                ls_val_ax.plot(self.curves[k][3].step, self.curves[k][3].value, label=k)
            except Exception as e:
                logger.debug(e)

        fig_acc_tn.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)
        fig_lss_tn.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)
        fig_acc_val.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)
        fig_lss_val.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), shadow=True, ncol=2)

        self.canvas_acc_train.draw()
        self.canvas_lss_train.draw()
        self.canvas_acc_val.draw()
        self.canvas_lss_val.draw()


class volFracPlotter(QWidget):
    '''on disk volFracPlotter'''
    def __init__(self, parent):
        super().__init__(parent)

        # back end attributes
        self.total_vs = 0  # {'total_vs': int, 'index': [1, 2, 4, 5...]}
        self.accum_nb_vx = None  # pd.DataFrame{A: [1234, 2345, ], B: [123, 234],...}
        self.plan_vs = None  # pd.DataFrame{A: [1234, 2345, ], B: [123, 234],...}

        # init curve
        volfracplot = plt.figure(dpi=50)
        self.canvas_volfrac = canvas(volfracplot)
        self.canvas_volfrac.setParent(parent)

        # init toolbar
        # Toolbar = toolbar(self.canvas_volfrac, self)

        # set layout
        self.QHBL = QtWidgets.QHBoxLayout()
        # self.QHBL.addWidget(Toolbar)
        self.QHBL.addWidget(self.canvas_volfrac)
        self.setLayout(self.QHBL)

        # draw
        self.canvas_volfrac.draw()

    def plot(self):
        plot_volfrac = self.canvas_volfrac.figure
        plot_volfrac.clear()
        vf_ax = plot_volfrac.add_subplot(111)
        self.plan_vs = self.accum_nb_vx.iloc[0, 1:].sum()

        for col in self.accum_nb_vx.columns[1:]:
            vf_ax.plot(self.accum_nb_vx.index,
                       self.accum_nb_vx[col] / self.plan_vs,
                       label=col,
                       linewidth=0.5)
        vf_ax.set_ylim([0., 1.])

        vf_ax.legend(loc='best')
        self.canvas_volfrac.draw()

