from _taskManager.dashboard_design import Ui_dashboard
from _taskManager.file_dialog import file_dialog
from PyQt5.QtWidgets import QDialog, QApplication, QLineEdit, QMessageBox
from PyQt5.QtCore import QThread, QObject, pyqtSlot, pyqtSignal, Qt

import matplotlib
matplotlib.use('QT5Agg')
import sys
import os
import pandas as pd
import numpy as np
from time import sleep
import re

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


class simpleSignal(QObject):
    launch = pyqtSignal(object)


class sideloop(QThread):
    def __init__(self, signal: simpleSignal):
        super().__init__()
        self.toggle = True
        self.signal = signal

    @pyqtSlot(name='loop')
    def run(self):
        self.toggle = True
        while self.toggle:
            sleep(300)
            self.signal.launch.emit(1)


    @pyqtSlot(name='terminate')
    def stop(self):
        self.toggle = False


class dashboard_logic(QDialog, Ui_dashboard):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)  # inherited two classes so do QDialog instead of super()?

        # front end config
        self.setupUi(self)
        # self.save_path_box = save_path_box(self.save_path_box)  # todo: enable the drag&drop
        self.setAcceptDrops(True)

        self.signal = simpleSignal()
        self.sideLoop = sideloop(signal=self.signal)

        self.refresh_button.clicked.connect(self.clean)
        self.save_button.clicked.connect(self.save_csv)
        self.folder_button.clicked.connect(self.choose_save_path)
        self.live_button.clicked.connect(self.check_live_button)

        self.signal.launch.connect(self.refresh)

    def check_live_button(self):
        if self.live_button.isChecked():
            self.sideLoop.start()
        else:
            self.sideLoop.stop()

    def refresh(self):
        print('refresh')
        self.mplwidget.curves = {}  # note: clean curves but not paths to let it reloads curves
        self.mplwidget.plot()

    def clean(self):
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
        self.mplwidget.repaint()

    def dragEnterEvent(self, event):
        print('detected: ', event.mimeData().text())
        event.accept()  # jump to dropEvent

    def dropEvent(self, event):
        path = event.mimeData().text()
        if self.mplwidget.geometry().contains(event.pos()):
            # note: $ ends of the input
            tmp = re.search(
            '(.*hour\d+_gpu-?\d+)',
            path
            )
            if tmp is not None:
                # protect
                self.setAcceptDrops(False)
                self.setCursor(Qt.WaitCursor)

                tmp = tmp.group(1)
                if not tmp.endswith('/'):
                    tmp += '/'

                max_id = 0
                if self.mplwidget.paths.__len__() != 0:
                    for i in self.mplwidget.paths.keys():
                        max_id = max(max_id, int(i))

                # add event folder path
                path = tmp.replace('file://', '')
                logger.debug(path)
                self.mplwidget.paths[max_id + 1] = path

                # add curve name in the list
                self.curves_list.clear()
                for k, v in self.mplwidget.paths.items():
                    self.curves_list.addItem(str(k))
                    self.curves_list.item(k - 1).setToolTip(v)

                self.setCursor(Qt.ArrowCursor)
                self.setAcceptDrops(True)

            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Oooppssss")
                msg.setInformativeText('Drag a folder which ends with format: (hour?_gpu?/) \nand which contains train/test of tf events folder!')
                msg.setWindowTitle("Error")
                msg.exec_()

            self.mplwidget.plot()

        elif self.save_path_box.geometry().contains(event.pos()):
            if os.path.isdir(path.replace('file://', '').replace('\r','').replace('\n','')):
                if not path.endswith('/'):
                    path += '/'
                self.save_path_box.setText(path)

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
            pd.DataFrame(ac_tn).to_csv(self.get_save_path() + 'acc_train.csv', header=True, index=False, sep=',')
            pd.DataFrame(ac_val).to_csv(self.get_save_path() + 'acc_val.csv', header=True, index=False, sep=',')
            pd.DataFrame(ls_tn).to_csv(self.get_save_path() + 'loss_train.csv', header=True, index=False, sep=',')
            pd.DataFrame(ls_val).to_csv(self.get_save_path() + 'loss_val.csv', header=True, index=False, sep=',')
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

