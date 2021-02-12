from _taskManager.dialog_design import Ui_Dialog
from _taskManager.file_dialog import file_dialog

from PySide2.QtWidgets import QDialog
import json
import os


class dialog_logic(QDialog, Ui_Dialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        # dialog UI
        self.setupUi(self)

        if os.path.exists('./_taskManager/latest.json'):
            with open('./_taskManager/latest.json', 'r') as file:
                params = json.load(file)

            # set line editors
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
            'comment': self.comment.text()
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

        with open('./_taskManager/latest.json', 'w') as file:
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
