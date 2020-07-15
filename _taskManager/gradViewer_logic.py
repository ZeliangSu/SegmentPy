from PyQt5.QtWidgets import QDialog

from _taskManager.gradViewer_design import Ui_grad_extractor
from _taskManager.file_dialog import file_dialog

from tensorboard_extractor import gradient_extractor


class gradView_logic(QDialog, Ui_grad_extractor):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)

        # front end config
        self.setupUi(self)
        self.pathButton.clicked.connect(self.set_grad_path)

    def return_grad_path(self):
        try:
            return self.path
        except Exception as e:
            print(e)
            return None

    def set_grad_path(self):
        grad_dial = file_dialog(title='select a training that you want to view its gradients', type='/')
        self.path = grad_dial.openFolderDialog()
        self.pathLine.setText(self.path)

    def extract_gradient(self):
        if hasattr(self, 'path'):
            self.set_grad_path()
        gradient_extractor(self.path)


