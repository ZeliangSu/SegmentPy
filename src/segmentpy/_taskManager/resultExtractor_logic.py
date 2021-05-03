from PySide2.QtWidgets import QApplication, QWidget, QColorDialog, QVBoxLayout, QHBoxLayout
from PySide2 import QtWidgets

from segmentpy._taskManager.resultExtractor_design import Ui_Extractor
from segmentpy._taskManager.blanketColorPalette_logic import clrPalette_logic
from segmentpy._taskManager.file_dialog import file_dialog
from segmentpy.tf114.hypParser import string_to_data

import pandas as pd
import os
from itertools import product


import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


class rltExtractor_logic(QWidget, Ui_Extractor):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)

        self.setupUi(self)
        self.dir = file_dialog(title='choose a folder to save the plot/legend/data', type='/').openFolderDialog()
        if not os.path.isdir(self.dir):
            logger.error('cannot find training files in this folder')
        self.pd_df = pd.DataFrame()
        self.progressBar.setValue(0)
        self.extractButton.clicked.connect(self.start_extract)
        self.saveButton.clicked.connect(self.saveResult)
        self.colorButton.clicked.connect(self.defineColor)

    def start_extract(self):
        l_fn = os.listdir(self.dir)
        self.progressBar.setRange(0, 100)
        for i, folder in enumerate(l_fn):
            if not folder.startswith('.'):  # MacOS: avoid './.DS_Store/'
                hypers = string_to_data(os.path.join(self.dir, folder))
                tmp = hypers.hyper_to_DataFrame()
                # pd_df.columns = tmp.columns
                self.pd_df = self.pd_df.append(tmp, ignore_index=True)
            self.progressBar.setValue(i / len(l_fn))
        logger.info('finished extracting')
        self.defineColor()

    def sort(self):
        self.pd_df = self.pd_df.sort_values(by='acc_tts_max', ascending=False).head(20)
        self.pd_df = pd.concat([self.pd_df.model,
                             self.pd_df.batch_size,
                             self.pd_df.lr_decay,
                             self.pd_df.kernel_size,
                             self.pd_df.nb_conv,
                             self.pd_df.lr_init,
                             self.pd_df.acc_tts_max], axis=1)

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

        # construct the color palette
        self.colorsPal = clrPalette_logic(None)
        mdl_set = False
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
                if not mdl_set:
                    self.colorsPal.label2310.setText(mdl)
                    self.colorsPal.pushButton2310.setText('color')
                    self.colorsPal.pushButton2310.setStyleSheet("background-color:rgb({})".format('255,0,0'))
                    self.colorsPal.pushButton2310.clicked.connect(self.chooseColor(self.colorsPal.pushButton2310))
                    mdl_set = True
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
                    tmp_button.setStyleSheet("background-color:rgbf({})".format('255,0,0'))
                    tmp_button.clicked.connect(self.chooseColor(tmp_button))
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
            # for the others

        try:
            self.colorsPal.show()
        except Exception as e:
            logger.error(e)

        # define the colors

        # QColorDialog
        self.display()

    def chooseColor(self, qtbutton):
        def showColorPicker():
            logger.debug(qtbutton)
            c = QColorDialog.getColor()
            logger.debug(c.name())
            qtbutton.setStyleSheet("background-color:{};".format(c.name()))
        return showColorPicker

    def display(self):
        # show plot
        # show legend
        pass

    def saveResult(self):
        save_path = file_dialog(title='choose a folder to save the plot/legend/data', type='/').openFolderDialog()
        self.pd_df.to_csv(os.path.join(save_path, 'data.csv'))


