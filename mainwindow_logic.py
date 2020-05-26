from PyQt5.QtCore import pyqtSignal, QThreadPool, QThread, QObject, QRunnable, pyqtSlot
from PyQt5.QtWidgets import QMainWindow,  QApplication, QTableWidgetItem, QErrorMessage, QMessageBox, QLabel
from PyQt5 import QtCore, QtGui

from _taskManager.mainwindow_design import Ui_LRCSNet
from dialog_logic import dialog_logic
from _taskManager.file_dialog import file_dialog
from dashboard_logic import dashboard_logic

import traceback, sys, os
from queue import Queue
from time import sleep
import subprocess
from threading import Thread

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


def get_available_gpus_wrapper():
    """this threading wrapper can get rid of residus tensorflow in gpus"""
    gpu_list = []
    t = Thread(target=get_available_gpus, args=(gpu_list,))
    t.start()
    t.join()
    return gpu_list


def get_available_gpus(l: list):
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    for x in local_device_protos:
        if x.device_type == 'GPU':
            l.append(int(x.name.split(':')[-1]))
    return l


class queueManager(QThread):
    def __init__(self, gpu_queue: Queue):
        super().__init__()
        self.enqueueListener = gpu_queue
        self.signals = WorkerSignals()
        self.toggle = False

    @pyqtSlot()
    def run(self):
        self.toggle = True
        while self.toggle:
            if self.enqueueListener.empty():
                continue  # note: don't use continu here, or saturate the CPU
            else:
                _gpu = list(self.enqueueListener.queue)[0]
                self.signals.available_gpu.emit(_gpu)
            sleep(20)  # note: at least wait 2 min for thread security, unknown GPU/inputpipeline bug

    @pyqtSlot()
    def stop(self):
        self.toggle = False

class WorkerSignals(QObject):
    error = pyqtSignal(tuple)
    released_gpu = pyqtSignal(object)
    start_proc = pyqtSignal(tuple)
    released_proc = pyqtSignal(tuple)
    available_gpu = pyqtSignal(int)


class predict_Worker(QRunnable):
    def __init__(self, ckpt_path: str, pred_dir: list, save_dir: str, *args, **kwargs):
        super(predict_Worker, self).__init__()
        self.ckpt_path = ckpt_path
        self.pred_dir = pred_dir
        self.save_dir = save_dir
        self.device = 'cpu'
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        thread_name = QThread.currentThread().objectName()
        thread_id = int(QThread.currentThreadId())
        print('On CPU')
        print('running name:{} on id:{}'.format(thread_name, thread_id))

        terminal = [
            'python', 'main_inference.py',
            '--ckpt', self.ckpt_path,
            '--raw', self.pred_dir,
            '--pred', self.save_dir
        ]

        # terminal = ['python', 'test.py']  # todo: uncomment here for similation
        # terminal = ['mpiexec', '--use-hwthread-cpus', 'python', 'test.py']  # todo: uncomment here for mpi similation

        process = subprocess.Popen(
            terminal,
        )
        signal = ('pred on {}: pid:{}'.format(self.device, process.pid), self.device, process, self.ckpt_path)
        # put proc queue pid and proc
        self.signals.start_proc.emit(signal)
        o, e = process.communicate()

        if e:
            logger.debug(e)
            self.signals.error.emit((e, traceback.format_exc()))

        self.signals.released_proc.emit(signal)


class training_Worker(QRunnable):
    def __init__(self, *args, **kwargs):
        super(training_Worker, self).__init__()
        self.using_gpu = str(args[0])
        self.params = args[1]
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        thread_name = QThread.currentThread().objectName()
        thread_id = int(QThread.currentThreadId())
        print('On GPU: {}'.format(self.using_gpu))
        print('running name:{} on id:{}'.format(thread_name, thread_id))

        terminal = [
            'python', 'main_train.py',
            '-nc', self.params['conv nb'],
            '-bs', self.params['batch size'],
            '-ws', self.params['window size'],
            '-ep', self.params['nb epoch'],
            '-cs', self.params['kernel size'],
            '-lr', self.params['lr type'],
            '-ilr', self.params['lr init'],
            '-klr', self.params['k param'],
            '-plr', self.params['period'],
            '-bn', self.params['batch norm'],
            '-do', self.params['dropout'],
            '-ag', self.params['augmentation'],
            '-fn', self.params['loss fn'],
            '-af', self.params['act fn'],
            '-mdl', self.params['model'],
            '-mode', self.params['cls/reg'],
            '-dv', self.using_gpu,
            '-st', self.params['sv step'],
            '-tb', self.params['tb step'],
            '-cmt', self.params['comment']
        ]

        print(self.params)
        print('\n', terminal)

        # terminal = ['python', 'dummy.py']  # todo: uncomment here for similation
        terminal = ['mpiexec', '--use-hwthread-cpus', 'python', 'test.py']  # todo: uncomment here for mpi similation

        process = subprocess.Popen(
            terminal,
        )
        # set signal
        signal = ('train on {}: pid:{}'.format(self.using_gpu, process.pid), self.using_gpu, process, self.params)

        # put proc queue pid and proc
        self.signals.start_proc.emit(signal)
        o, error = process.communicate()

        if error:
            self.signals.error.emit((error, traceback.format_exc()))

        self.signals.released_gpu.emit(self.using_gpu)
        self.signals.released_proc.emit(signal)


class mainwindow_logic(QMainWindow, Ui_LRCSNet):
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.menubar.setNativeMenuBar(False)

        # init the gpu queue and proc list
        self.gpu_queue = Queue()
        gpu_list = get_available_gpus_wrapper()

        for i in gpu_list:
            self.gpu_queue.put(i)
        if len(gpu_list) == 0:
            self.gpu_queue.put('cpu')
            # self.gpu_queue.put(1)  # todo: uncomment here for similation
            # self.gpu_queue.put(2)  # todo: uncomment here for similation
            # self.gpu_queue.put(3)  # todo: uncomment here for similation

        self.qManager = queueManager(gpu_queue=self.gpu_queue)
        self.refresh_gpu_list()
        self.proc_list = []  # tuple of (str: gpu, str: pid, subprocess)
        self.refresh_proc_list()

        self.threadpool = QThreadPool()
        self.qManager.signals.available_gpu.connect(self.start)

        _translate = QtCore.QCoreApplication.translate
        # init the hyper-params tablewidget
        item = self.tableWidget.item(0, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)  #note: make it read only
        item.setText(_translate("LRCSNet", "model"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(1, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "kernel size"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(2, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "conv nb"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(3, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "window size"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(4, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "batch size"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(5, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "nb epoch"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(6, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "batch norm"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(7, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "augmentation"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(8, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "dropout"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(9, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "lr type"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(10, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "lr init"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(11, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "k param"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(12, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "period"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(13, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "act fn"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(14, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "loss fn"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(15, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "cls/reg"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(16, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "sv step"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(17, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "tb step"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(18, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "comment"))
        item.setBackground(QtGui.QColor(128, 128, 128))

        self.tableWidget.setHorizontalHeaderLabels(['Hyper-parameter', 'next training'])

        # set the the buttons
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.add_button.clicked.connect(self.openDialog)
        self.clean_button.clicked.connect(self.clean)
        self.loop_button.clicked.connect(self.loop_state)
        self.forward_button.clicked.connect(self.forward)
        self.dashboard_button.clicked.connect(self.openDashboard)
        self.predict_button.clicked.connect(self.predict)

        # todo: plug in for activations

    def openDialog(self):
        self.dialog = dialog_logic(None)
        self.dialog.exec()  #.show() won't return
        if self.dialog.result() == 1:  #cancel: 0, ok: 1
            output = self.dialog.return_params()
            nb_col = self.tableWidget.columnCount()
            self.tableWidget.setColumnCount(nb_col + 1)
            for i, (k, v) in enumerate(output.items()):
                self.tableWidget.setItem(i, nb_col - 1, QTableWidgetItem(v))
            # bold first column
            self.bold(column=1)
        else:
            pass

    def predict(self):
        # define data folder path
        ckpt_dialog = file_dialog(title='select a checkpoint file .meta', type='.meta')
        ckpt_path = ckpt_dialog.openFileNameDialog()
        print(ckpt_path)

        # get to predict .tif
        predict_dialog = file_dialog(title='select folder of raw tomograms (*.tif) to predict', type='/')
        predict_dir = predict_dialog.openFolderDialog()
        print(predict_dir)

        # define predict folder path (can create new folder)
        save_dialog = file_dialog(title='select folder to put prediction', type='/')
        save_dir = save_dialog.openFolderDialog()
        print(save_dir)

        # spawn sub process
        _Worker = predict_Worker(ckpt_path=ckpt_path, pred_dir=predict_dir, save_dir=save_dir)
        self.threadpool.start(_Worker)
        _Worker.signals.start_proc.connect(self.add_proc_surveillance)
        _Worker.signals.released_proc.connect(self.remove_process_from_list)

    def setHeader(self):
        self.tableWidget.setHorizontalHeaderLabels(['Hyper-parameter', 'next training'])

    def bold(self, column):
        font = QtGui.QFont()
        font.setBold(True)
        for row_id in range(self.tableWidget.rowCount()):
            item = self.tableWidget.item(row_id, column)
            if item is not None:
                item.setFont(font)

    def unbold(self, column):
        font = QtGui.QFont()
        font.setBold(False)
        for row_id in range(self.tableWidget.rowCount()):
            item = self.tableWidget.item(row_id, column)
            if item is not None:
                item.setFont(font)

    def start(self):
        if not self.gpu_queue.empty() and self.verify_column_not_None():
            gpu = self.gpu_queue.get()
            self.refresh_gpu_list()

            # start a thread
            _Worker = training_Worker(gpu, self.grab_params())
            self.threadpool.start(_Worker)
            _Worker.signals.start_proc.connect(self.add_proc_surveillance)

            # release gpu and process
            _Worker.signals.error.connect(self.print_in_log)
            _Worker.signals.released_gpu.connect(self.enqueue)
            _Worker.signals.released_proc.connect(self.remove_process_from_list)

        elif not self.verify_column_not_None():
            print('Should fulfill the first column. \r')
        else:
            print('Waiting for available gpu \r')

    def print_in_log(self, content):
        # todo: in the log window
        print(content)

    def loop_state(self):
        if self.loop_button.isChecked():
            self.loop()
        else:
            self.stop_loop()

    def loop(self):
        self.qManager.start()

    def stop_loop(self):
        self.qManager.terminate()

    def stop(self):
        # refers to: https://stackoverflow.com/questions/37601672/how-can-i-get-the-indices-of-qlistwidgetselecteditems
        selected = self.ongoing_process.selectionModel().selectedIndexes()
        for item in selected:
            # options
            # os.kill(self.proc_list[item.row()][2].pid, sig.SIGTERM)
            # self.proc_list[item.row()][2].terminate()
            self.proc_list[item.row()][2].kill()

    def clean(self):
        column = self.tableWidget.currentColumn()
        self.tableWidget.removeColumn(column)
        if column == 1:
            self.setHeader()
            self.bold(column=1)

    def forward(self):
        column = self.tableWidget.currentColumn()
        if column > 1:
            self.tableWidget.insertColumn(column - 1)
            for i in range(self.tableWidget.rowCount()):
                self.tableWidget.setItem(i, column - 1, self.tableWidget.takeItem(i, column + 1))
                self.tableWidget.setCurrentCell(i, column - 1)
            self.tableWidget.removeColumn(column + 1)

            if column == 2:
                self.bold(column=1)
                self.unbold(column=2)

    def openDashboard(self):
        self.Dashboard = dashboard_logic(None)
        self.Dashboard.exec()

    def grab_params(self, column=1):
            nb_row = self.tableWidget.rowCount()
            out = {}
            for row in range(nb_row):
                out[self.tableWidget.item(row, 0).text()] = self.tableWidget.item(row, column).text()
            self.tableWidget.removeColumn(column)
            self.setHeader()
            self.bold(column=1)
            return out

    def enqueue(self, gpu):
        self.gpu_queue.put(gpu)
        self.refresh_gpu_list()

    def add_proc_surveillance(self, signal):
        self.proc_list.append(signal)
        self.refresh_proc_list()

    def remove_process_from_list(self, signal: tuple):
        # (str, subprocess.Popen)
        try:
            self.proc_list.remove(signal)
            self.refresh_proc_list()
        except Exception as e:
            logger.error(e)

    def refresh_gpu_list(self):
        self.AvailableGPUs.clear()
        self.AvailableGPUs.addItems(
            [str(i) for i in self.gpu_queue.queue]  # QListWidget only takes string not int
        )

    def refresh_proc_list(self):
        # this method only manipulate str in QlistWidget
        self.ongoing_process.clear()
        self.ongoing_process.addItems(['{}'.format(t[0]) for t in self.proc_list])
        for i, sig in zip(range(self.ongoing_process.count()), self.proc_list):
            self.ongoing_process.item(i).setToolTip(str(sig[3]).replace(',', '\n'))

    def verify_column_not_None(self, column=1):
        nb_row = self.tableWidget.rowCount()
        for row in range(nb_row):
            item = self.tableWidget.item(row, column)
            if item is None:
                return False
        return True


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('./_taskManager/logo.png'))

    # set ui
    ui = mainwindow_logic()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

