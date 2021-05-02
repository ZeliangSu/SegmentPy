from segmentpy._taskManager.dialog_design import Ui_Dialog
from segmentpy._taskManager.file_dialog import file_dialog

import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)  #changeHere: debug level
from PySide2.QtWidgets import QDialog, QMessageBox
import json
import os
import re


class dialog_logic(QDialog, Ui_Dialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        # dialog UI
        self.setupUi(self)
        self.latestPath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', 'latest.json')

        if os.path.exists(self.latestPath):
            with open(self.latestPath, 'r') as file:
                params = json.load(file)

            # set line editors
            try:
                self.mdl.setText(params['mdl'])
                self.ksize.setText(params['conv_size'])
                self.nbconv.setText(params['nb_conv'])
                self.winsize.setText(params['win_size'])
                self.batsize.setText(params['bat_size'])
                self.nbepoch.setText(params['nb_epoch'])
                self.dropout.setText(params['dropout'])
                self.initlr.setText(params['lr_init'])
                self.kparam.setText(params['lr_k'])
                self.pparam.setText(params['lr_p'])
                self.svsteps.setText(params['sv_step'])
                self.tbstep.setText(params['tb_step'])
                self.comment.setText(params['comment'])
                self.trn_dir_line.setText(params['train_dir'])
                self.train_dir = params['train_dir']
                self.val_dir_line.setText(params['val_dir'])
                self.val_dir = params['val_dir']
                self.test_dir_line.setText(params['test_dir'])
                self.tst_dir = params['test_dir']
                self.log_dir_line.setText(params['log_dir'])
                self.log_dir = params['log_dir']
                self.sampling_gap.setText(params['gap'])
                self.gap = params['gap']
                self.criterion.setText(params['condition'])
                self.cond = params['condition']
                self.correction.setText(params['correction'])
                self.corr = params['correction']
            except Exception as e:
                logger.error(e)

            # set QComboBox
            self.batnorm.setCurrentIndex(self.batnorm.findText(params['batch_norm']))
            self.aug.setCurrentIndex(self.aug.findText(params['aug']))
            self.lrtype.setCurrentIndex(self.lrtype.findText(params['lr_type']))
            self.actfn.setCurrentIndex(self.actfn.findText(params['act_fn']))
            self.lossfn.setCurrentIndex(self.lossfn.findText(params['lss_fn']))
            self.clsReg.setCurrentIndex(self.clsReg.findText(params['cls_reg']))

        # buttons
        self.buttonBox.accepted.connect(self.accept)  # ok button
        self.buttonBox.rejected.connect(self.reject)  # cancel button
        self.trn_dir_button.clicked.connect(self.set_train_dir)
        self.val_dir_button.clicked.connect(self.set_val_dir)
        self.test_dir_button.clicked.connect(self.set_tst_dir)
        self.log_dir_button.clicked.connect(self.set_log_dir)

        # remove automatically the space
        self.mdl.editingFinished.connect(lambda: self.removeSpace(self.mdl))
        self.ksize.editingFinished.connect(lambda: self.removeSpace(self.ksize))
        self.nbconv.editingFinished.connect(lambda: self.removeSpace(self.nbconv))
        self.batsize.editingFinished.connect(lambda: self.removeSpace(self.batsize))
        self.nbepoch.editingFinished.connect(lambda: self.removeSpace(self.nbepoch))
        self.dropout.editingFinished.connect(lambda: self.removeSpace(self.dropout))
        self.initlr.editingFinished.connect(lambda: self.removeSpace(self.initlr))
        self.kparam.editingFinished.connect(lambda: self.removeSpace(self.kparam))
        self.pparam.editingFinished.connect(lambda: self.removeSpace(self.pparam))
        self.sampling_gap.editingFinished.connect(lambda: self.removeSpace(self.sampling_gap))
        self.criterion.editingFinished.connect(lambda: self.removeSpace(self.criterion))
        self.correction.editingFinished.connect(lambda: self.removeSpace(self.correction))

    @staticmethod
    def removeSpace(qline):
        qline.setText(qline.text().replace(" ", ""))

    def log_window(self, title: str, Msg: str):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(Msg)
        msg.setWindowTitle(title)
        msg.exec_()

    def return_params(self):
        output = {
            'mdl': self.mdl.text(),
            'conv_size': self.ksize.text(),
            'nb_conv': self.nbconv.text(),
            'win_size': self.winsize.text(),
            'bat_size': self.batsize.text(),
            'nb_epoch': self.nbepoch.text(),
            'batch_norm': str(self.batnorm.currentText()),
            'aug': str(self.aug.currentText()),
            'dropout': self.dropout.text(),
            'lr_type': str(self.lrtype.currentText()),
            'lr_init': self.initlr.text(),
            'lr_k': self.kparam.text(),
            'lr_p': self.pparam.text(),
            'act_fn': str(self.actfn.currentText()),
            'lss_fn': str(self.lossfn.currentText()),
            'cls_reg': str(self.clsReg.currentText()),
            'sv_step': self.svsteps.text(),
            'tb_step': self.tbstep.text(),
            'comment': self.comment.text(),
            'gap': self.sampling_gap.text(),
            'condition': self.criterion.text(),
            'correction': self.correction.text(),
        }

        if ',' in output['gap'] or ';' in output['gap']:
            self.log_window(title='Not supported',
                            Msg='Do not support multiple gap yet')
        if ',' in output['condition'] or ';' in output['condition']:
            self.log_window(title='Not supported',
                            Msg='Do not support multiple condition yet')
        if ',' in output['correction'] or ';' in output['correction']:
            self.log_window(title='Not supported',
                            Msg='Do not support multiple correction yet')

        if hasattr(self, 'train_dir'):
            output['train_dir'] = self.trn_dir_line.text()
        else:
            output['train_dir'] = './train/'

        if hasattr(self, 'val_dir'):
            output['val_dir'] = self.val_dir_line.text()
        else:
            output['val_dir'] = './valid/'

        if hasattr(self, 'tst_dir'):
            output['test_dir'] = self.test_dir_line.text()
        else:
            output['test_dir'] = './test/'

        if hasattr(self, 'log_dir'):
            output['log_dir'] = self.log_dir_line.text()
        else:
            output['log_dir'] = './logs/'

        with open(self.latestPath, 'w') as file:
            json.dump(output, file)

        return output

    def set_train_dir(self):
        train_dir_dial = file_dialog(title='select a folder where includes the training dataset', type='/')
        self.train_dir = train_dir_dial.openFolderDialog()
        self.trn_dir_line.setText(self.train_dir)

    def set_val_dir(self):
        val_dir_dial = file_dialog(title='select a folder where includes the validation dataset', type='/')
        self.val_dir = val_dir_dial.openFolderDialog()
        self.val_dir_line.setText(self.val_dir)

    def set_tst_dir(self):
        tst_dir_dial = file_dialog(title='select a folder where includes the testing dataset', type='/')
        self.tst_dir = tst_dir_dial.openFolderDialog()
        self.test_dir_line.setText(self.tst_dir)

    def set_log_dir(self):
        log_dir_dial = file_dialog(title='select a folder to save the data', type='/')
        self.log_dir = log_dir_dial.openFolderDialog()
        self.log_dir_line.setText(self.log_dir)
