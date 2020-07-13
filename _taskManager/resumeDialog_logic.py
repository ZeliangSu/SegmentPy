from _taskManager.resumeDialog_design import Ui_Dialog
from _taskManager.file_dialog import file_dialog

from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QLabel, QWidget, QProgressDialog
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QThreadPool, QRunnable, pyqtSlot, pyqtSignal, QObject


class resumeDialog_logic(QDialog, Ui_Dialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.ckpt_path = None
        self.extra_ep = None
        self.new_cmt = None

        self.ckptButton.clicked.connect(self.selectckpt)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def selectckpt(self):
        ckpt_dialog = file_dialog(title='select a checkpoint file .meta', type='.meta')
        ckpt_path = ckpt_dialog.openFileNameDialog()
        self.ckpt_path = ckpt_path
        self.ckptLine.setText(self.ckpt_path)

    def return_params(self):
        output = {
            'ckpt_path': self.ckptLine.text(),
            'extra_ep': self.epochLine.text(),
            'new_cmt': self.commentLine.text()
        }
        return output

