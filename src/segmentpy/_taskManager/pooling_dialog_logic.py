from segmentpy._taskManager.pooling_dialog_design import Ui_Dialog

from PySide2.QtWidgets import QDialog


class dialog_logic(QDialog, Ui_Dialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        # dialog UI
        self.setupUi(self)

    def return_nb_max(self):
        return self.lineEdit.text()