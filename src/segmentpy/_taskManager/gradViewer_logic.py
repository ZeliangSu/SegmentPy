from PySide2.QtWidgets import QWidget, QDialog

from segmentpy._taskManager.gradViewer2_design import Ui_gradPlot
from segmentpy._taskManager.file_dialog import file_dialog

from segmentpy.tf114.score_extractor import gradient_extractor

import json
from os.path import join, exists, dirname


class gradView_logic(QDialog, Ui_gradPlot):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)

        # front end config
        self.setupUi(self)
        self.folderButton.clicked.connect(self.set_grad_path)

        self.loggerPath = join(dirname(dirname(__file__)), 'log', 'latest_gradView.json')
        if exists(self.loggerPath):
            with open(self.loggerPath, 'r') as f:
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
        if self.path is not None:
            _, _, gamma, betaOrBias, w, step = gradient_extractor(self.path)
            with open(self.loggerPath, 'w') as f:
                json.dump({"path": self.path}, f)

            self.plotWidget.gamma = gamma
            self.plotWidget.betaOrBias = betaOrBias
            self.plotWidget.w = w
            self.plotWidget.step = step
            self.plotWidget.plot()






