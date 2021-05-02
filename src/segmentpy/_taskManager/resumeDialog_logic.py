from segmentpy._taskManager.resumeDialog_design import Ui_Dialog
from segmentpy._taskManager.file_dialog import file_dialog

from PySide2.QtWidgets import QDialog

import json
import os


class resumeDialog_logic(QDialog, Ui_Dialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.ckpt_path = None
        self.extra_ep = None
        self.new_cmt = None
        self.trnPath = None
        self.valPath = None
        self.tstPath = None
        self.wherePath = None

        self.loggerDir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', 'latest_resume.json')

        if os.path.exists(self.loggerDir):
            with open(self.loggerDir, 'r') as f:
                tmp = json.load(f)
            try:
                # set attribute
                self.ckpt_path = tmp['ckpt_path']
                self.extra_ep = tmp['extra_ep']
                self.new_cmt = tmp['new_cmt']
                self.trnPath = tmp['trn repo. path']
                self.valPath = tmp['val repo. path']
                self.tstPath = tmp['tst repo. path']
                self.wherePath = tmp['mdl. saved path']

                self.init_lr = tmp["lr_init"]
                self.lr_dec_ratio = tmp["lr_decay"]
                self.lr_dec_period = tmp["lr_period"]
                self.gap = tmp["gap"]
                self.cond = tmp["condition"]

                self.svstep = tmp["sv step"]
                self.tbstep = tmp['tb step']

                # set lines
                self.ckptLine.setText(self.ckpt_path)
                self.epochLine.setText(self.extra_ep)
                self.commentLine.setText(self.new_cmt)
                self.trnPathLine.setText(self.trnPath)
                self.valPathLine.setText(self.valPath)
                self.tstPathLine.setText(self.tstPath)
                self.whereToSave.setText(self.wherePath)
                self.new_init_lr.setText(self.init_lr)
                self.new_lr_dec_ratio.setText(self.lr_dec_ratio)
                self.new_lr_dec_period.setText(self.lr_dec_period)
                self.sampling_gap.setText(self.gap)
                self.stop_cond.setText(self.cond)
                self.svstepLine.setText(self.svstep)
                self.tbstepLine.setText(self.tbstep)
            except KeyError as e:
                print(e)
                pass

        self.ckptButton.clicked.connect(self.selectckpt)
        self.trnPathButton.clicked.connect(self.selectTrnPath)
        self.valPathButton.clicked.connect(self.selectValPath)
        self.tstPathButton.clicked.connect(self.selectTstPath)
        self.whereToSaveButton.clicked.connect(self.selectWherePath)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def selectckpt(self):
        ckpt_dialog = file_dialog(title='select a checkpoint file .meta', type='.meta')
        ckpt_path = ckpt_dialog.openFileNameDialog()
        self.ckpt_path = ckpt_path
        self.ckptLine.setText(self.ckpt_path)

    def selectTrnPath(self):
        trnPath_dialog = file_dialog(title='select training data included directory', type='.meta')
        trnPath = trnPath_dialog.openFolderDialog()
        self.trnPath = trnPath
        self.trnPathLine.setText(self.trnPath)

    def selectValPath(self):
        valPath_dialog = file_dialog(title='select validation data included directory', type='.meta')
        valPath = valPath_dialog.openFolderDialog()
        self.valPath = valPath
        self.valPathLine.setText(self.valPath)

    def selectTstPath(self):
        tstPath_dialog = file_dialog(title='select testing data included directory', type='.meta')
        tstPath = tstPath_dialog.openFolderDialog()
        self.tstPath = tstPath
        self.tstPathLine.setText(self.tstPath)

    def selectWherePath(self):
        wherePath_dialog = file_dialog(title='select a directory to save your models', type='.meta')
        wherePath = wherePath_dialog.openFolderDialog()
        self.wherePath = wherePath
        self.whereToSave.setText(self.wherePath)

    def return_params(self):
        output = {
            "trn repo. path": self.trnPathLine.text(),
            "val repo. path": self.valPathLine.text(),
            "tst repo. path": self.tstPathLine.text(),
            "mdl. saved path": self.whereToSave.text(),
            "ckpt_path": self.ckptLine.text(),
            "extra_ep": self.epochLine.text(),
            "new_cmt": self.commentLine.text(),
            "lr_init": self.new_init_lr.text(),
            "lr_decay": self.new_lr_dec_ratio.text(),
            "lr_period": self.new_lr_dec_period.text(),
            "gap": self.sampling_gap.text(),
            "condition": self.stop_cond.text(),
            "sv step": self.svstepLine.text(),
            'tb step': self.tbstepLine.text(),

        }
        with open(self.loggerDir, 'w') as f:
            json.dump(output, f)
        return output

