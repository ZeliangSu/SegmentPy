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
            self.batnorm.setText(params['batch_norm'])
            self.aug.setText(params['aug'])
            self.dropout.setText(params['dropout'])
            self.lrtype.setText(params['lr_type'])
            self.initlr.setText(params['lr_init'])
            self.kparam.setText(params['lr_k'])
            self.pparam.setText(params['lr_p'])
            self.actfn.setText(params['act_fn'])
            self.lossfn.setText(params['lss_fn'])
            self.clsReg.setText(params['cls_reg'])
            self.svsteps.setText(params['sv_step'])
            self.tbstep.setText(params['tb_step'])
        else:
            self.mdl.setPlaceholderText('model name')
            self.ksize.setPlaceholderText('kernel size')
            self.nbconv.setPlaceholderText('number of convolution minimum per layer')
            self.winsize.setPlaceholderText('window size')
            self.batsize.setPlaceholderText('batch size')
            self.nbepoch.setPlaceholderText('number of epoch')
            self.batnorm.setPlaceholderText('batch normalization')
            self.aug.setPlaceholderText('data augmentation')
            self.dropout.setPlaceholderText('dropout probability')
            self.lrtype.setPlaceholderText('learning rate decay type')
            self.initlr.setPlaceholderText('initial learning rate')
            self.kparam.setPlaceholderText('k parameter in decay type')
            self.pparam.setPlaceholderText('decay periode / decay every n epoch')
            self.actfn.setPlaceholderText('activation function type')
            self.lossfn.setPlaceholderText('loss function type')
            self.clsReg.setPlaceholderText('classification / regression')
            self.svsteps.setPlaceholderText('save model every n steps')
            self.tbstep.setPlaceholderText('visualize gradients/weights every n steps')

        # buttons
        self.buttonBox.accepted.connect(self.accept)  # ok button
        self.buttonBox.rejected.connect(self.reject)  # cancel button

    def return_params(self):
        output = {
            'mdl': self.mdl.toPlainText(),
            'conv_size': self.ksize.toPlainText(),
            'nb_conv': self.nbconv.toPlainText(),
            'win_size': self.winsize.toPlainText(),
            'bat_size': self.batsize.toPlainText(),
            'nb_epoch': self.nbepoch.toPlainText(),
            'batch_norm': self.batnorm.toPlainText(),
            'aug': self.aug.toPlainText(),
            'dropout': self.dropout.toPlainText(),
            'lr_type': self.lrtype.toPlainText(),
            'lr_init': self.initlr.toPlainText(),
            'lr_k': self.kparam.toPlainText(),
            'lr_p': self.pparam.toPlainText(),
            'act_fn': self.actfn.toPlainText(),
            'lss_fn': self.lossfn.toPlainText(),
            'cls_reg': self.clsReg.toPlainText(),
            'sv_step': self.svsteps.toPlainText(),
            'tb_step': self.tbstep.toPlainText(),
        }

        with open('./_taskManager/latest.json', 'w') as file:
            json.dump(output, file)

        return output

