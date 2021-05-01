from _taskManager.predictDialog_design2 import Ui_Dialog
from _taskManager.file_dialog import file_dialog

from PySide2.QtWidgets import QDialog, QMessageBox

import json
import os


class predictDialog_logic(QDialog, Ui_Dialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.meta_path = None
        self.raw_folder = None
        self.pred_folder = None
        self.corrector = None
        self.latestPath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', 'latest_pred.json')

        if os.path.exists(self.latestPath):
            with open(self.latestPath, 'r') as f:
                tmp = json.load(f)
            try:
                self.meta_path = tmp['meta_path']
                self.metaLine.setText(self.meta_path)
                self.raw_folder = tmp['raw_folder']
                self.rawLine.setText(self.raw_folder)
                self.pred_folder = tmp['pred_folder']
                self.predLine.setText(self.pred_folder)
            except KeyError as e:
                print(e)
                pass
        self.metaButton.clicked.connect(self.selectMeta)
        self.rawButton.clicked.connect(self.selectRaw)
        self.predButton.clicked.connect(self.selectPred)
        self.buttonBox.accepted.connect(self.get_returns)
        self.buttonBox.rejected.connect(self.reject)

    def selectMeta(self):
        self.meta_path = file_dialog(title='choose .meta file', type='.meta').openFileNameDialog()
        self.metaLine.setText(self.meta_path)

    def selectRaw(self):
        self.raw_folder = file_dialog(title='choose a folder where contains the raw tomogram .tif').openFolderDialog()
        self.rawLine.setText(self.raw_folder)

    def selectPred(self):
        self.pred_folder = file_dialog(title='choose a folder to put the segmentation').openFolderDialog()
        self.predLine.setText(self.pred_folder)

    def logWindow(self, Msg='Error', title='Error'):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(Msg)
        msg.setWindowTitle(title)
        msg.exec_()

    def get_returns(self):
        if self.meta_path is None:
            self.logWindow(Msg='please select a .meta file')
        elif self.raw_folder is None:
            self.logWindow(Msg='please select a raw folder')
        elif self.pred_folder is None:
            self.logWindow(Msg='please select a prediction folder')
        elif self.corrector is None:
            self.logWindow(Msg='please set a corrector')
        else:
            with open(self.latestPath, 'w') as f:
                json.dump({
                    'meta_path': self.meta_path,
                    'raw_folder': self.raw_folder,
                    'pred_folder': self.pred_folder,
                    'corrector': self.corrector,
                }, f)
            self.accept()

    def get_params(self):
        return self.meta_path, self.raw_folder, self.pred_folder, self.corrector

