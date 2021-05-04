from PySide2.QtWidgets import QApplication, QDialog, QWidget, QColorDialog, QVBoxLayout, QHBoxLayout
from PySide2 import QtWidgets
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import Qt

from segmentpy._taskManager.resultExtractor_design import Ui_Extractor
from segmentpy._taskManager.blanketColorPalette_logic import clrPalette_logic
from segmentpy._taskManager.file_dialog import file_dialog
from segmentpy.tf114.hypParser import string_to_data

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as toolbar

import pandas as pd
import numpy as np
import os
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from operator import mul


import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


class rltExtractor_logic(QWidget, Ui_Extractor):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setupUi(self)

        # connect button
        self.folderButton.clicked.connect(self.chooseFolder)
        self.extractButton.clicked.connect(self.start_extract)
        self.saveButton.clicked.connect(self.saveResult)
        self.colorButton.clicked.connect(self.defineColor)

        # attrs
        self.dir = None
        self.pd_df = pd.DataFrame()
        self.progressBar.setValue(0)
        self.colors = None
        self.colorsPal = None

    def chooseFolder(self):
        self.dir = file_dialog(title='choose a folder to save the plot/legend/data', type='/').openFolderDialog()
        if not os.path.isdir(self.dir):
            logger.error('cannot find training files in this folder')

    def start_extract(self):
        l_fn = os.listdir(self.dir)
        self.progressBar.setRange(0, 100)
        for i, folder in enumerate(l_fn):
            if not folder.startswith('.'):  # MacOS: avoid './.DS_Store/'
                hypers = string_to_data(os.path.join(self.dir, folder))
                tmp = hypers.hyper_to_DataFrame()
                # pd_df.columns = tmp.columns
                self.pd_df = self.pd_df.append(tmp, ignore_index=True)
            self.progressBar.setValue(i / len(l_fn) * 100)
        logger.info('finished extracting')
        self.progressBar.setValue(100)
        self.sort()
        self.defineColor()

    def sort(self):
        self.pd_df = self.pd_df.sort_values(by='acc_tts_max', ascending=False)
        self.pd_df = pd.concat([self.pd_df.model,
                             self.pd_df.batch_size,
                             self.pd_df.lr_decay,
                             self.pd_df.kernel_size,
                             self.pd_df.nb_conv,
                             self.pd_df.lr_init,
                             self.pd_df.acc_tts_max], axis=1).copy()

    @staticmethod
    def enumerated_product(*args):
        yield from zip(product(*(range(len(x)) for x in args)), product(*args))

    def defineColor(self):
        self.uniqueness = {
            'mdl': self.pd_df.model.unique().tolist(),
            'bs': self.pd_df.batch_size.unique().tolist(),
            'lk': self.pd_df.lr_decay.unique().tolist(),
            'ks': self.pd_df.kernel_size.unique().tolist(),
            'nc': self.pd_df.nb_conv.unique().tolist(),
            'li': self.pd_df.lr_init.unique().tolist(),
        }
        logger.debug(self.uniqueness)
        if self.colors is None:
            self.colors = {}

        # construct the color palette
        if self.colorsPal is None:
            self.colorsPal = clrPalette_logic(None)
            mdl0_set = False
            bs0_set = False
            lk0_set = False
            ks0_set = False
            nc0_set = False
            li0_set = False

            for idx, (mdl, bs, lk, ks, nc, li) in self.enumerated_product(self.uniqueness['mdl'],
                                                                          self.uniqueness['bs'],
                                                                          self.uniqueness['lk'],
                                                                          self.uniqueness['ks'],
                                                                          self.uniqueness['nc'],
                                                                          self.uniqueness['li'],
                                                                          ):

                # mdl
                logger.debug('idx: {}\nmdl: {}'.format(idx, mdl))
                if idx[0] == 0:
                    if not mdl0_set:
                        self.colorsPal.label2310.setText(mdl)
                        self.colorsPal.pushButton2310.setText('color')
                        self.colorsPal.pushButton2310.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[mdl] = '#ff0000'
                        self.colorsPal.pushButton2310.clicked.connect(self.chooseColor(self.colorsPal.pushButton2310, mdl))
                        mdl0_set = True
                else:
                    if not hasattr(self.colorsPal, 'label231{}'.format(idx[0])):
                        # create label
                        setattr(self.colorsPal, 'label231{}'.format(idx[0]), QtWidgets.QLabel(self))
                        tmp_label = getattr(self.colorsPal, 'label231{}'.format(idx[0]))
                        tmp_label.setText(mdl)
                        # button
                        setattr(self.colorsPal, 'pushButton231{}'.format(idx[0]), QtWidgets.QPushButton(self))
                        tmp_button = getattr(self.colorsPal, 'pushButton231{}'.format(idx[0]), QtWidgets.QPushButton(self))
                        tmp_button.setObjectName('pushButton231{}'.format(idx[0]))
                        tmp_button.setText('color')
                        tmp_button.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[mdl] = '#ff0000'
                        tmp_button.clicked.connect(self.chooseColor(tmp_button, mdl))
                        # add into the layout
                        setattr(self.colorsPal, 'horizontalLayout231{}'.format(idx[0]), QHBoxLayout())
                        tmp_hblo = getattr(self.colorsPal, 'horizontalLayout231{}'.format(idx[0]))
                        tmp_hblo.addWidget(tmp_label)
                        tmp_hblo.addWidget(tmp_button)
                        # tmp_hblo.show()  #
                        tmp_vblo = getattr(self.colorsPal, 'verticalLayout231')
                        tmp_vblo.addLayout(tmp_hblo)
                        self.colorsPal.setLayout(tmp_vblo)
                        # self.tmp_vblo.show()
                        logger.debug(self.colorsPal.verticalLayout_global)
                # bs
                if idx[1] == 0:
                    if not bs0_set:
                        self.colorsPal.label2320.setText(bs)
                        self.colorsPal.pushButton2320.setText('color')
                        self.colorsPal.pushButton2320.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[bs] = '#ff0000'
                        self.colorsPal.pushButton2320.clicked.connect(self.chooseColor(self.colorsPal.pushButton2320, bs))
                        bs0_set = True
                else:
                    if not hasattr(self.colorsPal, 'label232{}'.format(idx[1])):
                        # create label
                        setattr(self.colorsPal, 'label232{}'.format(idx[1]), QtWidgets.QLabel(self))
                        tmp_label = getattr(self.colorsPal, 'label232{}'.format(idx[1]))
                        tmp_label.setText(bs)
                        # button
                        setattr(self.colorsPal, 'pushButton232{}'.format(idx[1]), QtWidgets.QPushButton(self))
                        tmp_button = getattr(self.colorsPal, 'pushButton232{}'.format(idx[1]), QtWidgets.QPushButton(self))
                        tmp_button.setObjectName('pushButton232{}'.format(idx[1]))
                        tmp_button.setText('color')
                        tmp_button.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[bs] = '#ff0000'
                        tmp_button.clicked.connect(self.chooseColor(tmp_button, bs))
                        # add into the layout
                        setattr(self.colorsPal, 'horizontalLayout232{}'.format(idx[1]), QHBoxLayout())
                        tmp_hblo = getattr(self.colorsPal, 'horizontalLayout232{}'.format(idx[1]))
                        tmp_hblo.addWidget(tmp_label)
                        tmp_hblo.addWidget(tmp_button)
                        # tmp_hblo.show()  #
                        tmp_vblo = getattr(self.colorsPal, 'verticalLayout232')
                        tmp_vblo.addLayout(tmp_hblo)
                        self.colorsPal.setLayout(tmp_vblo)
                        # self.tmp_vblo.show()
                        logger.debug(self.colorsPal.verticalLayout_global)
                # lk
                if idx[2] == 0:
                    if not lk0_set:
                        self.colorsPal.label2330.setText(lk)
                        self.colorsPal.pushButton2330.setText('color')
                        self.colorsPal.pushButton2330.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[lk] = '#ff0000'
                        self.colorsPal.pushButton2330.clicked.connect(self.chooseColor(self.colorsPal.pushButton2330, lk))
                        lk0_set = True
                else:
                    if not hasattr(self.colorsPal, 'label233{}'.format(idx[2])):
                        # create label
                        setattr(self.colorsPal, 'label233{}'.format(idx[2]), QtWidgets.QLabel(self))
                        tmp_label = getattr(self.colorsPal, 'label233{}'.format(idx[2]))
                        tmp_label.setText(lk)
                        # button
                        setattr(self.colorsPal, 'pushButton233{}'.format(idx[2]), QtWidgets.QPushButton(self))
                        tmp_button = getattr(self.colorsPal, 'pushButton232{}'.format(idx[2]), QtWidgets.QPushButton(self))
                        tmp_button.setObjectName('pushButton233{}'.format(idx[2]))
                        tmp_button.setText('color')
                        tmp_button.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[lk] = '#ff0000'
                        tmp_button.clicked.connect(self.chooseColor(tmp_button, lk))
                        # add into the layout
                        setattr(self.colorsPal, 'horizontalLayout233{}'.format(idx[2]), QHBoxLayout())
                        tmp_hblo = getattr(self.colorsPal, 'horizontalLayout233{}'.format(idx[2]))
                        tmp_hblo.addWidget(tmp_label)
                        tmp_hblo.addWidget(tmp_button)
                        # tmp_hblo.show()  #
                        tmp_vblo = getattr(self.colorsPal, 'verticalLayout233')
                        tmp_vblo.addLayout(tmp_hblo)
                        self.colorsPal.setLayout(tmp_vblo)
                        # self.tmp_vblo.show()
                        logger.debug(self.colorsPal.verticalLayout_global)
                # ks
                if idx[3] == 0:
                    if not ks0_set:
                        self.colorsPal.label2340.setText(ks)
                        self.colorsPal.pushButton2340.setText('color')
                        self.colorsPal.pushButton2340.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[ks] = '#ff0000'
                        self.colorsPal.pushButton2340.clicked.connect(self.chooseColor(self.colorsPal.pushButton2340, ks))
                        ks0_set = True
                else:
                    if not hasattr(self.colorsPal, 'label234{}'.format(idx[3])):
                        # create label
                        setattr(self.colorsPal, 'label234{}'.format(idx[3]), QtWidgets.QLabel(self))
                        tmp_label = getattr(self.colorsPal, 'label234{}'.format(idx[3]))
                        tmp_label.setText(ks)
                        # button
                        setattr(self.colorsPal, 'pushButton234{}'.format(idx[3]), QtWidgets.QPushButton(self))
                        tmp_button = getattr(self.colorsPal, 'pushButton234{}'.format(idx[3]), QtWidgets.QPushButton(self))
                        tmp_button.setObjectName('pushButton234{}'.format(idx[3]))
                        tmp_button.setText('color')
                        tmp_button.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[ks] = '#ff0000'
                        tmp_button.clicked.connect(self.chooseColor(tmp_button, ks))
                        # add into the layout
                        setattr(self.colorsPal, 'horizontalLayout234{}'.format(idx[3]), QHBoxLayout())
                        tmp_hblo = getattr(self.colorsPal, 'horizontalLayout234{}'.format(idx[3]))
                        tmp_hblo.addWidget(tmp_label)
                        tmp_hblo.addWidget(tmp_button)
                        # tmp_hblo.show()  #
                        tmp_vblo = getattr(self.colorsPal, 'verticalLayout234')
                        tmp_vblo.addLayout(tmp_hblo)
                        self.colorsPal.setLayout(tmp_vblo)
                        # self.tmp_vblo.show()
                        logger.debug(self.colorsPal.verticalLayout_global)
                # nc
                if idx[4] == 0:
                    if not nc0_set:
                        self.colorsPal.label2350.setText(nc)
                        self.colorsPal.pushButton2350.setText('color')
                        self.colorsPal.pushButton2350.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[nc] = '#ff0000'
                        self.colorsPal.pushButton2350.clicked.connect(self.chooseColor(self.colorsPal.pushButton2350, nc))
                        nc0_set = True
                else:
                    if not hasattr(self.colorsPal, 'label235{}'.format(idx[4])):
                        # create label
                        setattr(self.colorsPal, 'label235{}'.format(idx[4]), QtWidgets.QLabel(self))
                        tmp_label = getattr(self.colorsPal, 'label235{}'.format(idx[4]))
                        tmp_label.setText(nc)
                        # button
                        setattr(self.colorsPal, 'pushButton235{}'.format(idx[4]), QtWidgets.QPushButton(self))
                        tmp_button = getattr(self.colorsPal, 'pushButton235{}'.format(idx[4]), QtWidgets.QPushButton(self))
                        tmp_button.setObjectName('pushButton235{}'.format(idx[4]))
                        tmp_button.setText('color')
                        tmp_button.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[nc] = '#ff0000'
                        tmp_button.clicked.connect(self.chooseColor(tmp_button, nc))
                        # add into the layout
                        setattr(self.colorsPal, 'horizontalLayout235{}'.format(idx[4]), QHBoxLayout())
                        tmp_hblo = getattr(self.colorsPal, 'horizontalLayout235{}'.format(idx[4]))
                        tmp_hblo.addWidget(tmp_label)
                        tmp_hblo.addWidget(tmp_button)
                        # tmp_hblo.show()  #
                        tmp_vblo = getattr(self.colorsPal, 'verticalLayout235')
                        tmp_vblo.addLayout(tmp_hblo)
                        self.colorsPal.setLayout(tmp_vblo)
                        # self.tmp_vblo.show()
                        logger.debug(self.colorsPal.verticalLayout_global)
                # li
                if idx[5] == 0:
                    if not li0_set:
                        self.colorsPal.label2360.setText(li)
                        self.colorsPal.pushButton2360.setText('color')
                        self.colorsPal.pushButton2360.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[li] = '#ff0000'
                        self.colorsPal.pushButton2360.clicked.connect(self.chooseColor(self.colorsPal.pushButton2360, li))
                        li0_set = True
                else:
                    if not hasattr(self.colorsPal, 'label236{}'.format(idx[5])):
                        # create label
                        setattr(self.colorsPal, 'label236{}'.format(idx[5]), QtWidgets.QLabel(self))
                        tmp_label = getattr(self.colorsPal, 'label236{}'.format(idx[5]))
                        tmp_label.setText(li)
                        # button
                        setattr(self.colorsPal, 'pushButton236{}'.format(idx[5]), QtWidgets.QPushButton(self))
                        tmp_button = getattr(self.colorsPal, 'pushButton236{}'.format(idx[5]), QtWidgets.QPushButton(self))
                        tmp_button.setObjectName('pushButton236{}'.format(idx[5]))
                        tmp_button.setText('color')
                        tmp_button.setStyleSheet("background-color:{};".format('#ff0000'))
                        self.colors[li] = '#ff0000'
                        tmp_button.clicked.connect(self.chooseColor(tmp_button, li))
                        # add into the layout
                        setattr(self.colorsPal, 'horizontalLayout236{}'.format(idx[5]), QHBoxLayout())
                        tmp_hblo = getattr(self.colorsPal, 'horizontalLayout236{}'.format(idx[5]))
                        tmp_hblo.addWidget(tmp_label)
                        tmp_hblo.addWidget(tmp_button)
                        # tmp_hblo.show()  #
                        tmp_vblo = getattr(self.colorsPal, 'verticalLayout236')
                        tmp_vblo.addLayout(tmp_hblo)
                        self.colorsPal.setLayout(tmp_vblo)
                        # self.tmp_vblo.show()
                        logger.debug(self.colorsPal.verticalLayout_global)

        try:
            self.colorsPal.exec()
        except Exception as e:
            logger.error(e)

        # define the colors
        if self.colorsPal.result() == 1:
            # QColorDialog
            logger.debug(self.colors)
            self.display()

    def chooseColor(self, qtbutton, name):
        def showColorPicker():
            logger.debug(qtbutton)
            c = QColorDialog.getColor()

            logger.debug(c.name())
            qtbutton.setStyleSheet("background-color:{};".format(c.name()))

            logger.debug(name)
            self.colors[str(name)] = c.name()
        return showColorPicker

    def display(self):
        # show plot
        self.MPLwidget.df = self.pd_df
        self.MPLwidget.plot()
        # show legend
        self.makeLegend()

    def makeLegend(self):
        # make the grid in matplotlib
        logger.debug(self.pd_df.info())
        logger.debug(self.pd_df.head())
        grid = self.pd_df.to_numpy()[:, :-1]
        fig2 = plt.figure(figsize=(4, 2))
        ax = fig2.add_subplot(111)
        ax.imshow(np.zeros((10 * grid.shape[1], 10 * grid.shape[0], 3)))
        for i in range(grid.shape[1]):
            # print(i, tb)
            for j in range(grid.shape[0]):
                # print(j)
                color = self.colors[str(grid[j, i])]
                # print(color)
                rect = patches.Rectangle((j * 10, i * 10), 10, 10,
                                         linewidth=1, edgecolor='black',
                                         facecolor=color, snap=False) # snap=False anti-aliasing
                ax.add_patch(rect)
        plt.axis('off')

        # convert matplotlib to img
        fig2.canvas.draw()
        data = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8)
        logger.debug('grid shape: {}'.format(data.shape))
        logger.debug(fig2.canvas.get_width_height())
        logger.debug(fig2.canvas.get_width_height()[::-1])
        data = data.reshape(
            # fig2.canvas.get_width_height()[::-1]
            tuple([i for i in fig2.canvas.get_width_height()[::-1]])
            + (3,))
        logger.debug('grid shape: {}'.format(data.shape))
        # data = data.transpose(1, 0, 2).copy()
        # logger.debug('grid shape: {}'.format(data.shape))
        self.fig2 = data
        del fig2

        # plot weight
        self.q = QImage(data,
                        data.shape[1],
                        data.shape[0],
                        data.shape[1] * 3, QImage.Format_RGB888
                        )
        self.p = QPixmap(self.q)
        self.p.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.legendLabel.setScaledContents(True)
        self.legendLabel.setPixmap(self.p)
        self.legendLabel.update()
        self.legendLabel.repaint()

    def saveResult(self):
        save_path = file_dialog(title='choose a folder to save the plot/legend/data', type='/').openFolderDialog()
        self.pd_df.to_csv(os.path.join(save_path, 'data.csv'))
        self.MPLwidget.canvas_acc.figure.savefig(os.path.join(save_path, 'plot.png'))
        Image.fromarray(np.asarray(self.fig2)).save(os.path.join(save_path, 'legend.png'))


