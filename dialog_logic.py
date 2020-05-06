from _taskManager.dialog_design import Ui_Dialog
from PyQt5.QtWidgets import QDialog
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

        # buttons
        self.buttonBox.accepted.connect(self.accept)  # ok button
        self.buttonBox.rejected.connect(self.reject)  # cancel button

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

        with open('./_taskManager/latest.json', 'w') as file:
            json.dump(output, file)

        return output

