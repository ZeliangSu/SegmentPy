from PySide2 import QtWidgets
from PySide2.QtWidgets import QWidget, QMessageBox
from PySide2.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as toolbar
from segmentpy.tf114.score_extractor import lr_curve_extractor

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QT5Agg')

import re
import pandas as pd
import numpy as np
import string

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


def heatmapPreparation(x, y, z):
    pass


class MPL(QWidget):
    """learning curves plot"""
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
                ac_tn_ax.set_ylabel('accuracy')
                ls_tn_ax.plot(self.curves[k][2].step, self.curves[k][2].value, label=k)
                ls_tn_ax.set_ylabel('loss')
                ac_val_ax.plot(self.curves[k][1].step, self.curves[k][1].value, label=k)
                ac_val_ax.set_ylabel('accuracy')
                ls_val_ax.plot(self.curves[k][3].step, self.curves[k][3].value, label=k)
                ls_val_ax.set_ylabel('loss')
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

    def thicken(self, idxes: list):
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
        for i, (k, v) in enumerate(self.paths.items()):
            try:
                if i not in idxes:
                    ac_tn_ax.plot(self.curves[k][0].step, self.curves[k][0].value, label=k, alpha=0.5)
                    ls_tn_ax.plot(self.curves[k][2].step, self.curves[k][2].value, label=k, alpha=0.5)
                    ac_val_ax.plot(self.curves[k][1].step, self.curves[k][1].value, label=k, alpha=0.5)
                    ls_val_ax.plot(self.curves[k][3].step, self.curves[k][3].value, label=k, alpha=0.5)

                else:
                    ac_tn_ax.plot(self.curves[k][0].step, self.curves[k][0].value, label=k, linewidth=2)
                    ls_tn_ax.plot(self.curves[k][2].step, self.curves[k][2].value, label=k, linewidth=2)
                    ac_val_ax.plot(self.curves[k][1].step, self.curves[k][1].value, label=k, linewidth=2)
                    ls_val_ax.plot(self.curves[k][3].step, self.curves[k][3].value, label=k, linewidth=2)
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
        self.accum_interf = None  # pd.DataFrame{'0-1': [0.1, 0.5, 0.3 ...], '0-2': [0.6, 0.02, ...] ...}
        self.plan_vs = None  # pd.DataFrame{A: [1234, 2345, ], B: [123, 234],...}

        # init curve
        volfracplot = plt.figure(dpi=40)
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
        plot = self.canvas_volfrac.figure
        plot.clear()

        # plot vol frac
        vf_ax = plot.add_subplot(211)
        vf_ax.set_title('Volume Fractions')
        self.plan_vs = self.accum_nb_vx.iloc[0, 1:].sum()

        for col in self.accum_nb_vx.columns[1:]:
            vf_ax.plot(self.accum_nb_vx.index,
                       self.accum_nb_vx[col] / self.plan_vs,
                       label=col,
                       linewidth=0.5)
        vf_ax.set_ylim([0., 1.])
        vf_ax.legend(loc='best')

        # plot interface
        interf_ax = plot.add_subplot(212)
        interf_ax.set_title('Interfaces')
        for col in self.accum_interf.columns[1:]:
            interf_ax.plot(self.accum_interf.index,
                       self.accum_interf[col],
                       label=col,
                       linewidth=0.5)
        interf_ax.legend(loc='best')

        plot.tight_layout()
        self.canvas_volfrac.draw()


class gradient_plot(QWidget):
    """gradient plot"""
    def __init__(self, parent):
        super().__init__(parent)

        self.w = {}
        self.betaOrBias = {}
        self.gamma = {}
        self.beta = {}
        self.step = {}

        # todo: w/gamma/beta plots if using batch norm (gamma beta), or 2 plots w/b without batch norm
        figure_w = plt.figure(dpi=50)
        # figure_gamma = plt.figure(dpi=50)
        # figure_beta = plt.figure(dpi=50)

        # set canvas
        self.canvas_w = canvas(figure_w)
        # self.canvas_gamma = canvas(figure_gamma)
        # self.canvas_beta = canvas(figure_beta)

        # layout
        self.QHBL = QtWidgets.QHBoxLayout()
        self.QHBL.addWidget(self.canvas_w)
        # self.QHBL.addWidget(self.canvas_gamma)
        # self.QHBL.addWidget(self.canvas_beta)
        self.setLayout(self.QHBL)

        # todo: scrollable
        # self.QHBL.layout().setContentsMargins(0, 0, 0, 0)
        # self.QHBL.layout().setSpacing(0)


        # draw blancket
        self.canvas_w.draw()
        # self.canvas_gamma.draw()
        # self.canvas_beta.draw()

    def plot(self):
        if self.w.__len__() == 0:
            return

        # weight
        fig = self.canvas_w.figure
        fig.clear()
        w_ax = fig.add_subplot(131)
        w_ax.set_title('weights', fontsize=20)
        df = pd.DataFrame(self.w, index=self.step)
        w_ax.set_xticklabels([n.replace('summary/', '').replace('/w_0/grad', '') for n in self.w.keys()],
                             rotation=90, fontsize=20, ha='left')  # center
        w_ax.set_xticks(np.arange(len(self.w.keys())))
        w_ax.tick_params(axis='y', which='major', labelsize=20)
        c = w_ax.pcolor(df)  #, cmap='RdBu'
        cbar = fig.colorbar(c)
        cbar.ax.tick_params(labelsize=20)

        # gamma
        g_ax = fig.add_subplot(132)
        g_ax.set_title('gammas', fontsize=20)
        df2 = pd.DataFrame(self.gamma, index=self.step)
        g_ax.set_xticklabels([n.replace('summary/', '').replace('/gamma_0/grad', '').replace('_BN/batch_norm', '')
                              for n in self.gamma.keys()],
                             minor=False, rotation=90, fontsize=20, ha='left')
        g_ax.set_xticks(np.arange(len(self.gamma.keys())))
        g_ax.tick_params(axis='y', which='major', labelsize=20)
        c2 = g_ax.pcolor(df2)
        cbar2 = fig.colorbar(c2)
        cbar2.ax.tick_params(labelsize=20)

        # beta
        b_ax = fig.add_subplot(133)
        b_ax.set_title('betas or bias', fontsize=20)
        df3 = pd.DataFrame(self.betaOrBias, index=self.step)
        b_ax.set_xticklabels([n.replace('summary/', '').replace('_0/grad', '').replace('_BN/batch_norm', '')
                              for n in self.betaOrBias.keys()],
                             minor=False, rotation=90, fontsize=20, ha='left')
        b_ax.set_xticks(np.arange(len(self.betaOrBias.keys())))
        b_ax.tick_params(axis='y', which='major', labelsize=20)
        c3 = b_ax.pcolor(df3, vmin=df3.min().min(), vmax=df3.max().max())
        cbar3 = fig.colorbar(c3)
        cbar3.ax.tick_params(labelsize=20)

        fig.tight_layout()
        self.canvas_w.draw()


class sortResult(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.df = None
        fig1 = plt.figure()
        self.canvas1 = canvas(fig1)
        self.canvas1.setParent(parent)
        ########## the following code is mandatory despite the layout being added in QtDesigner, or the MPLwidget sticks on the cornor
        self.QHBL = QtWidgets.QHBoxLayout()
        self.QHBL.addWidget(self.canvas1)
        self.setLayout(self.QHBL)
        ########## the above code is mandatory despite the layout being added  in QtDesigner, or the MPLwidget sticks on the cornor
        self.canvas1.draw()

    def plot(self):
        # clear
        fig1 = self.canvas1.figure
        fig1.clear()

        # plot
        ax = fig1.add_subplot(111)
        ax.scatter(x=np.arange(len(self.df.acc_tts_max)), y=self.df.acc_tts_max)
        ax_min, ax_max = ax.get_xlim()
        ax.set_xlim(ax_min, ax_max)
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        fig1.tight_layout()
        self.canvas1.draw()

