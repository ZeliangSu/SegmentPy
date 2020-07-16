from PyQt5.QtCore import pyqtSignal, QThreadPool, QThread, QObject, QRunnable, pyqtSlot, Qt
from PyQt5.QtWidgets import QMainWindow,  QApplication, QTableWidgetItem, QMessageBox
from PyQt5 import QtCore, QtGui

from _taskManager.mainwindow_design import Ui_LRCSNet
from _taskManager.dialog_logic import dialog_logic
from _taskManager.file_dialog import file_dialog
from _taskManager.dashboard_logic import dashboard_logic
from _taskManager.nodes_list_logic import node_list_logic
from _taskManager.volumes_viewer_logic import volViewer_logic
from _taskManager.metric_logic import metric_logic
from _taskManager.augmentationViewer_logic import augViewer_logic
from _taskManager.resumeDialog_logic import resumeDialog_logic
from _taskManager.gradViewer_logic import gradView_logic

from util import print_nodes_name
from parser import string_to_hypers

import traceback, sys, os
from queue import Queue
from time import sleep
import subprocess
from threading import Thread
import re

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


def get_available_gpus_wrapper():
    """this threading wrapper can get rid of residus tensorflow in gpus"""

    proc = subprocess.Popen(['python', 'device.py'])
    proc.wait()
    with open('./device.txt', 'r') as f:
        gpu_list = [line.rstrip() for line in f.readlines()]
    return gpu_list


class queueManager(QThread):
    def __init__(self, gpu_queue: Queue):
        super().__init__()
        self.enqueueListener = gpu_queue
        self.signals = WorkerSignals()
        self.toggle = False

    @pyqtSlot(name='queue1')
    def run(self):
        self.toggle = True
        while self.toggle:
            if self.enqueueListener.empty():
                continue  # note: don't use continu here, or saturate the CPU
            else:
                _gpu = list(self.enqueueListener.queue)[0]
                self.signals.available_gpu.emit(_gpu)
            sleep(20)  # note: at least wait 2 min for thread security, unknown GPU/inputpipeline bug

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

    @pyqtSlot(name='predict')
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

    @pyqtSlot(name='train')
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
            '-cmt', self.params['comment'],
            '-trnd', self.params['trn repo. path'],
            '-vald', self.params['val repo. path']
        ]

        print(self.params)
        print('\n', terminal)

        terminal = ['python', 'test.py']  # todo: uncomment here for similation

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


class retraining_Worker(QRunnable):
    def __init__(self, *args, **kwargs):
        super(retraining_Worker, self).__init__()
        self.using_gpu = str(args[0])
        self.params = args[1]
        self.signals = WorkerSignals()

    @pyqtSlot(name='retrain')
    def run(self):
        thread_name = QThread.currentThread().objectName()
        thread_id = int(QThread.currentThreadId())
        print('On GPU: {}'.format(self.using_gpu))
        print('running name:{} on id:{}'.format(thread_name, thread_id))

        terminal = [
            'python', 'main_retrain.py',
            '-ckpt', self.params['ckpt path'],
            '-ep', self.params['nb epoch'],
            '-lr', self.params['lr type'],
            '-ilr', self.params['lr init'],
            '-klr', self.params['k param'],
            '-plr', self.params['period'],
            '-dv', self.using_gpu,
            '-cmt', self.params['comment'],
        ]



        print(self.params)
        print('\n', terminal)

        # terminal = ['python', 'test.py']  # todo: uncomment here for similation

        process = subprocess.Popen(
            terminal,
        )
        # set signal
        signal = ('retrain on {}: pid:{}'.format(self.using_gpu, process.pid), self.using_gpu, process, self.params)

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
        item = self.tableWidget.item(19, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "trn repo. path"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(20, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "val repo. path"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(21, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "ckpt path"))
        item.setBackground(QtGui.QColor(128, 128, 128))

        self.header = ['Parameters', 'nextTrain']
        self.setHeader()

        # set the the buttons
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.add_button.clicked.connect(self.addTrain)
        self.clean_button.clicked.connect(self.clean)
        self.loop_button.clicked.connect(self.loop_state)
        self.pushButton.clicked.connect(self.addResume)
        self.forward_button.clicked.connect(self.forward)
        self.dashboard_button.clicked.connect(self.openDashboard)
        self.predict_button.clicked.connect(self.predict)

        # menu bar
        self.Activations.triggered.connect(self.activation_plugin)
        self.Loss_Landscape.triggered.connect(self.loss_landscape)
        self.Random_Forest.triggered.connect(self.random_forest)
        self.Volumes_Viewer.triggered.connect(self.volViewer_plugin)
        self.Metrics.triggered.connect(self.metric_plugin)
        self.AugViewer.triggered.connect(self.augViewer_plugin)
        self.GradViewer.triggered.connect(self.gradViewer_plugin)

    ################# menubar methods

    def gradViewer_plugin(self):
        self.gradV = gradView_logic()
        self.gradV.exec_()
        if self.gradV.result() == 1:
            self.gradV.extract_gradient()

    def augViewer_plugin(self):
        self.augV = augViewer_logic()
        self.augV.exec_()

    def metric_plugin(self):
        self.metric = metric_logic()
        self.metric.exec_()

    def volViewer_plugin(self):
        self.volViewer = volViewer_logic()
        self.volViewer.exec_()

    def activation_plugin(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #note: here might have conflict if there's an ongoing training with GPU
        import tensorflow as tf
        dialog = file_dialog(title='select ckpts (*.meta) to retrieve activations', type='.meta')
        ckpt_paths = dialog.openFileNamesDialog()
        if len(ckpt_paths) != 0:
            # restore from ckpt the nodes
            tf.reset_default_graph()
            logger.debug(ckpt_paths[0])
            _ = tf.train.import_meta_graph(
                ckpt_paths[0],
                clear_devices=True,
            )

            # get arguments
            graph = tf.get_default_graph().as_graph_def()
            nodes = print_nodes_name(graph)
            steps = [re.search('step(\d+)', ck_pth).group(1) for ck_pth in ckpt_paths]

            # retrive nodes of activations
            options = []
            for node in nodes:
                tmp = re.search('(^[a-zA-Z]+\d*\/).*(leaky|relu|sigmoid|tanh|logits\/identity|up\d+\/Reshape\_4|concat)$', node)
                if tmp is not None:
                    tmp = tmp.string
                    if 'identity' in tmp:
                        iden = tmp
                    else:
                        options.append(tmp)

            # open nodes list dialog
            nodes_list = node_list_logic(options=options)
            nodes_list.exec()
            if nodes_list.result() == 1:
                acts = nodes_list.return_nodes()
                if iden:
                    acts.append(iden)
                types = nodes_list.return_analysis_types()
                if len(types) == 0:
                    types = ['activation']

                terminal = [
                    'python', 'main_analytic.py',
                    '-ckpt', *ckpt_paths,
                    '-step', *steps,
                    '-type', *types,
                    '-node', *acts,
                ]

                logger.debug(terminal)

                proc = subprocess.Popen(
                    terminal
                )
                proc.wait()

    def log_window(self, title: str, Msg: str):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(Msg)
        msg.setWindowTitle(title)
        msg.exec_()

    def loss_landscape(self):
        self.log_window(
            title='Ooppsss',
            Msg="Plug-in loss landscape of Goldstein et al. is coming in the next version. \nYou can try at terminal with main_loss_landscape.py"
        )

    def random_forest(self):
        self.log_window(
            title='Ooppsss',
            Msg="Plug-in randomForest of Arganda-Carreras et al. is coming in the next version. \nYou can try at terminal with randomForest.py"
        )

    ############ main button methods

    def addTrain(self):
        pivot_table = {
            'mdl': 'model',
            'bat_size': 'batch size',
            'win_size': 'window size',
            'conv_size': 'kernel size',
            'nb_conv': 'conv nb',
            'act_fn': 'act fn',
            'lss_fn': 'loss fn',
            'batch_norm': 'batch norm',
            'aug': 'augmentation',
            'dropout': 'dropout',
            'lr_type': 'lr type',
            'lr_init': 'lr init',
            'lr_k': 'k param',
            'lr_p': 'period',
            'cls_reg': 'cls/reg',
            'comment': 'comment',
            'nb_epoch': 'nb epoch',
            'sv_step': 'sv step',
            'tb_step': 'tb step',
            'train_dir': 'trn repo. path',
            'val_dir': 'val repo. path',
        }
        self.dialog = dialog_logic(None)
        self.dialog.exec()  #.show() won't return
        if self.dialog.result() == 1:  #cancel: 0, ok: 1
            output = self.dialog.return_params()
            nb_col = self.tableWidget.columnCount()
            self.tableWidget.setColumnCount(nb_col + 1)

            # write params
            for k, v in output.items():
                i = self.tableWidget.findItems(pivot_table[k], Qt.MatchFlag.MatchExactly)
                self.tableWidget.setItem(i[0].row(), nb_col - 1, QTableWidgetItem(v))

            if nb_col > 2:
                self.header.append('train')
            else:
                self.header[1] = 'nextTrain'
            # bold first column
            self.lock_params()
            self.bold(column=1)
            self.setHeader()

    def addResume(self):
        pivot_table = {
            'model': 'model',
            'batch_size': 'batch size',
            'window_size': 'window size',
            'kernel_size': 'kernel size',
            'nb_conv': 'conv nb',
            'act_fn': 'act fn',
            'BatchNorm': 'batch norm',
            'augmentation': 'augmentation',
            'dropout': 'dropout',
            'loss_fn': 'loss fn',
            'lr_decay_type': 'lr type',
            'lr_init': 'lr init',
            'lr_decay': 'k param',
            'lr_period': 'period',
            'comment': 'comment',
            'ckpt_path': 'ckpt path',
            'nb_epoch': 'nb epoch',
            'mode': 'cls/reg'
        }
        self.Rdialog = resumeDialog_logic(None)
        self.Rdialog.exec()
        if self.Rdialog.result() == 1:
            output = self.Rdialog.return_params()

            nb_col = self.tableWidget.columnCount()
            self.tableWidget.setColumnCount(nb_col + 1)

            # retrieve old params
            old_hypers = string_to_hypers(output['ckpt_path']).parse()
            old_hypers['ckpt_path'] = output['ckpt_path']
            old_hypers['nb_epoch'] = int(output['extra_ep'])
            old_hypers['comment'] = output['new_cmt']

            # write params
            for k, v in old_hypers.items():
                i = self.tableWidget.findItems(pivot_table[k], Qt.MatchFlag.MatchExactly)
                self.tableWidget.setItem(i[0].row(), nb_col - 1, QTableWidgetItem(str(v)))

            if nb_col > 2:
                self.header.append('resume')
            else:
                self.header[1] = 'nextResume'
            # bold first column
            self.lock_params()
            self.bold(column=1)
            self.setHeader()

    def predict(self):
        # define data folder path
        ckpt_dialog = file_dialog(title='select a checkpoint file .meta', type='.meta')
        ckpt_path = ckpt_dialog.openFileNameDialog()
        print(ckpt_path)

        if ckpt_path:
            # get to predict .tif
            predict_dialog = file_dialog(title='select folder of raw tomograms (*.tif) to predict', type='/')
            predict_dir = predict_dialog.openFolderDialog()
            print(predict_dir)

            # define predict folder path (can create new folder)
            if predict_dir:
                save_dialog = file_dialog(title='select folder to put prediction', type='/')
                save_dir = save_dialog.openFolderDialog()
                print(save_dir)

                if save_dir:
                    # spawn sub process
                    _Worker = predict_Worker(ckpt_path=ckpt_path, pred_dir=predict_dir, save_dir=save_dir)
                    self.threadpool.start(_Worker)
                    _Worker.signals.start_proc.connect(self.add_proc_surveillance)
                    _Worker.signals.released_proc.connect(self.remove_process_from_list)

    def start(self):
        if self.header[1] == 'nextTrain':
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

        elif self.header[1] == 'nextResume':
            gpu = self.gpu_queue.get()
            self.refresh_gpu_list()

            # start a thread
            _Worker = retraining_Worker(gpu, self.grab_params())
            self.threadpool.start(_Worker)
            _Worker.signals.start_proc.connect(self.add_proc_surveillance)

            # release gpu and process
            _Worker.signals.error.connect(self.print_in_log)
            _Worker.signals.released_gpu.connect(self.enqueue)
            _Worker.signals.released_proc.connect(self.remove_process_from_list)

    def print_in_log(self, content):
        # todo: in the log window
        print(content)

    def loop_state(self):
        if self.loop_button.isChecked():
            self.qManager.start()
        else:
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
        if column >= 1:
            self.tableWidget.removeColumn(column)
            if column == 1:
                self.bold(column=1)
                self.popHeader(column)

    def forward(self):
        column = self.tableWidget.currentColumn()
        if column > 1:
            self.tableWidget.insertColumn(column - 1)
            for i in range(self.tableWidget.rowCount()):
                self.tableWidget.setItem(i, column - 1, self.tableWidget.takeItem(i, column + 1))
                self.tableWidget.setCurrentCell(i, column - 1)
            self.tableWidget.removeColumn(column + 1)

            if column == 2:
                self.setHeader()
                self.bold(column=1)
                self.unbold(column=2)

            # swap header
            self.header[column - 1], self.header[column] = self.header[column], self.header[column - 1]

    def openDashboard(self):
        self.Dashboard = dashboard_logic(None)
        self.Dashboard.exec()

    ########### ongoing/available Qlist methods

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

    ########### Qtable method

    def grab_params(self):
        nb_row = self.tableWidget.rowCount()
        out = {}

        # get training params or resume params
        if self.header[1] == 'nextTrain':
            for row in range(nb_row):
                out[self.tableWidget.item(row, 0).text()] = self.tableWidget.item(row, 1).text()
            self.popHeader(1)

        elif self.header[1] == 'nextResume':
            for row in range(nb_row):
                if self.tableWidget.item(row, 1) is not None:
                    # use index name as dict's key
                    out[self.tableWidget.item(row, 0).text()] = self.tableWidget.item(row, 1).text()
            self.popHeader(1)

        else:
            raise NotImplementedError

        # refresh the table
        self.tableWidget.removeColumn(1)
        self.setHeader()
        self.bold(column=1)
        return out

    def verify_column_not_None(self, column=1):
        nb_row = self.tableWidget.rowCount()
        for row in range(nb_row):
            item = self.tableWidget.item(row, column)
            if item is None:
                return False
        return True

    def lock_params(self):
        for column, head in enumerate(self.header):
            if 'train' in head.lower():
                for row in range(self.tableWidget.rowCount()):
                    if self.tableWidget.item(row, column) is not None:
                        # self.tableWidget.item(row, column).setFlags(QtCore.Qt.ItemIsEditable)  # note: make it editable
                        pass
            elif 'resume' in head.lower():
                for row in range(self.tableWidget.rowCount()):
                    if self.tableWidget.item(row, 0).text() not in ['lr type', 'lr init', 'k param', 'period',
                                                                    'ckpt path', 'comment', 'nb epoch']:
                        if self.tableWidget.item(row, column) is not None:
                            self.tableWidget.item(row, column).setBackground(QtGui.QColor(230,230,250))
                            self.tableWidget.item(row, column).setFlags(QtCore.Qt.ItemIsEditable)  # note: make it read only

    def setHeader(self):
        self.tableWidget.setHorizontalHeaderLabels(self.header)

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

    def popHeader(self, column=1):
        if column == 1:
            try:
                if self.header[column + 1] == 'train':
                    self.header[column + 1] = 'nextTrain'
                elif self.header[column + 1] == 'resume':
                    self.header[column + 1] = 'nextResume'
                else:
                    raise ValueError('capture unknown header')
                self.header.pop(column)
            except IndexError as e:
                self.header.pop(column)
                self.header.append('nextTrain')

        self.setHeader()


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('./img/logo.png'))

    # set ui
    ui = mainwindow_logic()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

