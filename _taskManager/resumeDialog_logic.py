from _taskManager.resumeDialog_design import Ui_Dialog
from _taskManager.file_dialog import file_dialog

from PyQt5.QtWidgets import QDialog

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

        if os.path.exists('./_taskManager/latest_resume.json'):
            with open('./_taskManager/latest_resume.json', 'r') as f:
                tmp = json.load(f)
            try:
                self.ckpt_path = tmp['ckpt_path']
                self.extra_ep = tmp['extra_ep']
                self.new_cmt = tmp['new_cmt']
                self.trnPath = tmp['trn repo. path']
                self.valPath = tmp['val repo. path']
                self.ckptLine.setText(self.ckpt_path)
                self.epochLine.setText(self.extra_ep)
                self.commentLine.setText(self.new_cmt)
                self.trnPathLine.setText(self.trnPath)
                self.valPathLine.setText(self.valPath)
            except KeyError as e:
                print(e)
                pass

        self.ckptButton.clicked.connect(self.selectckpt)
        self.trnPathButton.clicked.connect(self.selectTrnPath)
        self.valPathButton.clicked.connect(self.selectValPath)
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

    def return_params(self):
        output = {
            "trn repo. path": self.trnPathLine.text(),
            "val repo. path": self.valPathLine.text(),
            "ckpt_path": self.ckptLine.text(),
            "extra_ep": self.epochLine.text(),
            "new_cmt": self.commentLine.text()
        }
        with open('./_taskManager/latest_resume.json', 'w') as f:
            json.dump(output, f)
        return output

