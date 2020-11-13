from PySide2.QtWidgets import QWidget, QDialog

from _taskManager.gradViewer2_design import Ui_gradPlot
from _taskManager.file_dialog import file_dialog

from tensorboard_extractor import gradient_extractor

import json
import os


class gradView_logic(QDialog, Ui_gradPlot):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)

        # front end config
        self.setupUi(self)
        self.folderButton.clicked.connect(self.set_grad_path)

        if os.path.exists('./_taskManager/latest_gradView.json'):
            with open('./_taskManager/latest_gradView.json', 'r') as f:
                tmp = json.load(f)['path']
                self.lineEdit.setText(tmp)
                self.path = tmp

        else:
            self.path = None

        self.lineEdit.returnPressed.connect(self.extract_gradient)

    def return_grad_path(self):
        try:
            return self.path
        except Exception as e:
            print(e)
            return None

    def set_grad_path(self):
        grad_dial = file_dialog(title='select a training that you want to view its gradients', type='/')
        self.path = grad_dial.openFolderDialog()
        if self.path:
            self.lineEdit.setText(self.path)
            self.extract_gradient()

    def extract_gradient(self):
        _, _, gamma, betaOrBias, w, step = gradient_extractor(self.path)
        with open('./_taskManager/latest_gradView.json', 'w') as f:
            json.dump({"path": self.path}, f)

        self.plotWidget.gamma = gamma
        self.plotWidget.betaOrBias = betaOrBias
        self.plotWidget.w = w
        self.plotWidget.step = step
        self.plotWidget.plot()






