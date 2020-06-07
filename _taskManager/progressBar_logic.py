from PyQt5.QtWidgets import QDialog

from _taskManager.progressBar_design import Ui_standalone_pBar


class procBar_logic(QDialog, Ui_standalone_pBar):
    def __init__(self, title: str, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.processing_title.setText(title)

    def update_progress(self, val: int):
        self.progressBar.setValue(val)

    def set_total(self, total: int):
        self.progressBar.setMaximum(total)