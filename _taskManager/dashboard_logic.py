from _taskManager.dashboard_design import Ui_dashboard
from _taskManager.file_dialog import file_dialog
from PyQt5.QtWidgets import QDialog, QApplication, QLineEdit
from PyQt5.QtCore import QThread, QObject, pyqtSlot, pyqtSignal

# from tensorboard_extractor import lr_curve_extractor

import matplotlib
matplotlib.use('QT5Agg')
import sys
import pandas as pd
import numpy as np
from time import sleep

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


class save_path_box(QLineEdit):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, QDragEnterEvent):
        print('detected: ', QDragEnterEvent.mimeData().text())
        QDragEnterEvent.accept()

    def dropEvent(self, QDropEvent):
        self.setText(QDropEvent.mimeData().text().replace('file://', ''))


class sideloop(QThread):
    def __init__(self):
        super().__init__()
        self.toggle = True
        self.signal =simpleSignal()

    @pyqtSlot()
    def run(self):
        self.toggle = True
        while self.toggle:
            self.signal.launch.emit(1)
            sleep(20)

    @pyqtSlot()
    def stop(self):
        self.toggle = False

class simpleSignal(QObject):
    launch = pyqtSignal(object)


class dashboard_logic(QDialog, Ui_dashboard):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)  # inherited two classes so do QDialog instead of super()?

        # front end config
        self.setupUi(self)
        # self.save_path_box = save_path_box(self.save_path_box)  # todo: enable the drag&drop

        # complete the total layout with the partial MPL widget layout
        # self.verticalLayout.addChildLayout(self.mplwidget.QHBL)
        # self.setLayout(self.mplwidget.QHBL)
        # self.setLayout(self.verticalLayout)

        # note: dashboard dialog as parent to FigureCanvasQTAgg to give drawing permission to dashboard
        # self.mplwidget.canvas1.setParent(self)  #note: this line will align the canvas upon the top-left corner

        self.sideLoop = sideloop()

        self.refresh_button.clicked.connect(self.refresh)
        self.save_button.clicked.connect(self.save_csv)
        self.folder_button.clicked.connect(self.choose_save_path)
        self.live_button.clicked.connect(self.check_live_button)
        self.signal = simpleSignal()
        self.signal.launch.connect(self.mplwidget.plot)

    def check_live_button(self):
        if self.live_button.isChecked():
            self.sideLoop.start()
        else:
            self.sideLoop.terminate()

    def refresh(self):
        fig_acc_tn = self.mplwidget.canvas_acc_train.figure
        fig_acc_val = self.mplwidget.canvas_acc_val.figure
        fig_lss_tn = self.mplwidget.canvas_lss_train.figure
        fig_lss_val = self.mplwidget.canvas_lss_val.figure

        fig_acc_tn.clear()
        fig_acc_val.clear()
        fig_lss_tn.clear()
        fig_lss_val.clear()

        self.mplwidget.canvas_acc_train.draw()
        self.mplwidget.canvas_acc_val.draw()
        self.mplwidget.canvas_lss_train.draw()
        self.mplwidget.canvas_lss_val.draw()

        self.mplwidget.paths = {}
        self.mplwidget.curves = {}
        print('refresh')

    def choose_save_path(self):
        folder_path = file_dialog(title='select folder to save curves .csv', type='/').openFolderDialog()
        self.save_path_box.setText(folder_path)

    def get_save_path(self):
        return self.save_path_box.text()

    def save_csv(self):
        ac_tn, ac_val, ls_tn, ls_val = {}, {}, {}, {}
        for k, v in self.mplwidget.curves.items():
            if 'step' not in ac_tn.keys():
                ac_tn['step'] = np.asarray(v[0].step)
                ac_val['step'] = np.asarray(v[1].step)
                ls_tn['step'] = np.asarray(v[2].step)
                ls_val['step'] = np.asarray(v[3].step)
            ac_tn[k] = np.asarray(v[0].value)
            ac_val[k] = np.asarray(v[1].value)
            ls_tn[k] = np.asarray(v[2].value)
            ls_val[k] = np.asarray(v[3].value)
        try:
            pd.DataFrame(ac_tn).to_csv(self.get_save_path() + 'acc_train.csv', header=True, index=False, sep=';')
            pd.DataFrame(ac_val).to_csv(self.get_save_path() + 'acc_val.csv', header=True, index=False, sep=';')
            pd.DataFrame(ls_tn).to_csv(self.get_save_path() + 'loss_train.csv', header=True, index=False, sep=';')
            pd.DataFrame(ls_val).to_csv(self.get_save_path() + 'loss_val.csv', header=True, index=False, sep=';')
        except Exception as e:
            # fixme: length or number of steps different with throw pandas bug
            logger.debug(e)


def test():
    app = QApplication(sys.argv)

    # set ui
    ui = dashboard_logic()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test()

