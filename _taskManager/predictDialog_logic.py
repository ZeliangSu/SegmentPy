from _taskManager.predictDialog_design2 import Ui_Form
from _taskManager.file_dialog import file_dialog

from PyQt5.QtWidgets import QWidget, QMessageBox

import json
import os


class predictDialog_logic(QWidget, Ui_Form):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.meta_path = None
        self.raw_folder = None
        self.pred_folder = None

        if os.path.exists('./_taskManager/latest_resume.json'):
            with open('./_taskManager/latest_resume.json', 'r') as f:
                tmp = json.load(f)
            try:
                self.meta_path = tmp['meta_path']
                self.raw_folder = tmp['raw_folder']
                self.pred_folder = tmp['pred_folder']
            except KeyError as e:
                print(e)
                pass
        self.metaButton.clicekd.connect(self.selectMeta)
        self.rawButton.clicekd.connect(self.selectRaw)
        self.predButton.clicekd.connect(self.selectPred)
        self.buttonBox.accepted.connect(self.get_returns)
        self.buttonBox.rejected.connect(self.reject)

    def selectMeta(self):
        self.meta_path = file_dialog(title='choose .meta file', type='.meta').openFileNameDialog()
        self.metaLine.setText(self.meta_path)

    def selectRaw(self):
        self.raw_folder = file_dialog(title='choose a folder where contains the raw tomogram .tif', type='.meta').openFileNameDialog()
        self.rawLine.setText(self.raw_folder)

    def selectPred(self):
        self.pred_folder = file_dialog(title='choose a folder to put the segmentation', type='.meta').openFileNameDialog()
        self.predLine.setText(self.pred_folder)

    def logWindow(self, msg='Error', title='Error'):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(msg)
        msg.setWindowTitle(title)
        msg.exec_()

    def get_returns(self):
        if self.meta_path is None:
            self.logWindow(msg='please select a .meta file')
        elif self.raw_folder is None:
            self.logWindow(msg='please select a raw folder')
        elif self.pred_folder is None:
            self.logWindow(msg='please select a prediction folder')
        else:
            with open('./_taskManager/latest_resume.json', 'w') as f:
                json.dump({
                    'meta_path': self.meta_path,
                    'raw_folder': self.raw_folder,
                    'pred_folder': self.pred_folder,
                }, f)
            self.accept()

    def get_params(self):
        return self.meta_path, self.raw_folder, self.pred_folder

