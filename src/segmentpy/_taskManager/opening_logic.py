from segmentpy._taskManager.opening_design import Ui_Dialog
from PySide2.QtWidgets import QDialog


class opening_logic(QDialog, Ui_Dialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        self.setupUi(self)

