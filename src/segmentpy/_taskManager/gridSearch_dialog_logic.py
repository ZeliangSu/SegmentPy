from segmentpy._taskManager.gridSearch_dialog_design import Ui_gridSearch_dialog
from segmentpy._taskManager.file_dialog import file_dialog

from PySide2.QtWidgets import QDialog
import json
import os
import re


class gS_dialog_logic(QDialog, Ui_gridSearch_dialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        # dialog UI
        self.setupUi(self)

        self.loggerPath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', 'latest_gS.json')
        if os.path.exists(self.loggerPath):
            with open(self.loggerPath, 'r') as file:
                params = json.load(file)
                try:
                    self.trn_dir_line.setText(params['train_dir'])
                    self.train_dir = params['train_dir']
                    self.val_dir_line.setText(params['val_dir'])
                    self.val_dir = params['val_dir']
                    self.test_dir_line.setText(params['test_dir'])
                    self.tst_dir = params['test_dir']
                    self.log_dir_line.setText(params['log_dir'])
                    self.log_dir = params['log_dir']
                    self.correction = params['correction']
                    self.sample_gap = params['sample gap']
                    self.stop_criterion = params['criterion']
                except KeyError as e:
                    print(e)
                    pass

        self.buttonBox.accepted.connect(self.accept)  # ok button
        self.buttonBox.rejected.connect(self.reject)  # cancel button
        self.trn_dir_button.clicked.connect(self.set_train_dir)
        self.val_dir_button.clicked.connect(self.set_val_dir)
        self.test_dir_button.clicked.connect(self.set_tst_dir)
        self.log_dir_button.clicked.connect(self.set_log_dir)

    def return_params(self):
        output = {
            'mdl': self.model_line.text(),
            'ks': re.split('\,|\:|\/|\ ', self.kernel_size.text()),
            'nc': re.split('\,|\:|\/|\ ', self.nb_conv.text()),
            'bs': re.split('\,|\:|\/|\ ', self.batch_size.text()),
            'ilr': re.split('\,|\:|\/|\ ', self.init_lr.text()),
            'lrdecay': re.split('\,|\:|\/|\ ', self.decay_ratio.text()),
            'cmt': self.comment.text(),
            'correction': str(self.correctionLine.text()),
            'sample gap': str(self.sample_gapLine.text()),
            'criterion': str(self.correctionLine.text()),
        }
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

        with open(self.loggerPath, 'w') as file:
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