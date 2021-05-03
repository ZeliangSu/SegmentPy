from PySide2.QtCore import Signal, QThreadPool, QThread, QObject, QRunnable, Slot, Qt
from PySide2.QtWidgets import QMainWindow,  QApplication, QTableWidgetItem, QMessageBox
from PySide2 import QtCore, QtGui

from segmentpy._taskManager.mainwindow_design import Ui_LRCSNet
from segmentpy._taskManager.dialog_logic import dialog_logic
from segmentpy._taskManager.file_dialog import file_dialog
from segmentpy._taskManager.dashboard_logic import dashboard_logic
from segmentpy._taskManager.nodes_list_logic import node_list_logic
from segmentpy._taskManager.volumes_viewer_logic import volViewer_logic
from segmentpy._taskManager.metric_logic import metric_logic
from segmentpy._taskManager.augmentationViewer_logic import augViewer_logic
from segmentpy._taskManager.ActViewer_logic import actViewer_logic
from segmentpy._taskManager.resumeDialog_logic import resumeDialog_logic
from segmentpy._taskManager.gradViewer_logic import gradView_logic
from segmentpy._taskManager.trainableParamsList_logic import resumeNodes_logic
from segmentpy._taskManager.predictDialog_logic import predictDialog_logic
from segmentpy._taskManager.gridSearch_dialog_logic import gS_dialog_logic
from segmentpy._taskManager.resultExtractor_logic import rltExtractor_logic

from segmentpy.tf114.util import print_nodes_name
from segmentpy.tf114.hypParser import string_to_hypers


import traceback, sys, os
from queue import Queue
from time import sleep
import subprocess
from itertools import product
import re

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level

loggerDir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log')
imgDir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'img')
segmentpyDir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tf114')
parentDir = os.path.dirname(__file__)

if not os.path.exists(loggerDir):
    os.makedirs(loggerDir)


def get_available_gpus_wrapper():
    """this threading wrapper can get rid of residus tensorflow in gpus"""

    logger.info('detecting GPUs with {}'.format(os.path.join(segmentpyDir, 'device.py')))
    proc = subprocess.Popen(['python', os.path.join(segmentpyDir, 'device.py')])
    proc.wait()
    with open(os.path.join(loggerDir, 'device.txt'), 'r') as f:
        gpu_list = [line.rstrip() for line in f.readlines()]
    return gpu_list


class queueManager(QThread):
    def __init__(self, gpu_queue: Queue):
        super().__init__()
        self.enqueueListener = gpu_queue
        self.signals = WorkerSignals()
        self.toggle = False

    @Slot(name='queue1')
    def run(self):
        self.toggle = True
        # todo: the following pythonic loop should be replaced by inner event loop of pyside!!
        while self.toggle:
            if self.enqueueListener.empty():
                continue  # note: don't use continue here, or saturate the CPU
            else:
                _gpu = list(self.enqueueListener.queue)[0]
                self.signals.available_gpu.emit(_gpu)
            sleep(20)  # note: at least wait 2 min for thread security, unknown GPU/inputpipeline bug

    def stop(self):
        self.toggle = False


class WorkerSignals(QObject):
    error = Signal(tuple)
    released_gpu = Signal(object)
    start_proc = Signal(tuple)
    released_proc = Signal(tuple)
    available_gpu = Signal(int)


class predict_Worker(QRunnable):
    def __init__(self, ckpt_path: str, pred_dir: list, save_dir: str, correction: float, *args, **kwargs):
        super(predict_Worker, self).__init__()
        self.ckpt_path = ckpt_path
        self.pred_dir = pred_dir
        self.save_dir = save_dir
        self.correction = correction
        self.device = 'cpu'
        self.signals = WorkerSignals()

    @Slot(name='predict')
    def run(self):
        thread_name = QThread.currentThread().objectName()
        thread_id = str(QThread.currentThread())
        print('On CPU')
        print('running name:{} on id:{}'.format(thread_name, thread_id))

        terminal = [
            'mpirun', '--use-hwthread-cpus',
            'python', os.path.join(segmentpyDir, 'inference.py'),
            '--ckpt', self.ckpt_path,
            '--raw', self.pred_dir,
            '--pred', self.save_dir,
            '--corr', str(self.correction),
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

    @Slot(name='train')
    def run(self):
        thread_name = QThread.currentThread().objectName()
        thread_id = str(QThread.currentThread())
        print('On GPU: {}'.format(self.using_gpu))
        print('running name:{} on id:{}'.format(thread_name, thread_id))

        terminal = [
            'python', os.path.join(segmentpyDir, 'main_train.py'),
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
            '-stride', self.params['sampl. gap'],
            '-cond', self.params['stop. crit.'],
            '-st', self.params['sv step'],
            '-tb', self.params['tb step'],
            '-cmt', self.params['comment'],
            '-corr', self.params['correction'],
            '-trnd', self.params['trn repo. path'],
            '-vald', self.params['val repo. path'],
            '-tstd', self.params['tst repo. path'],
            '-logd', self.params['mdl. saved path'],
        ]

        print(self.params)
        print('\n', terminal)

        # terminal = ['python', 'test.py']  # todo: uncomment here for similation

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

    @Slot(name='retrain')
    def run(self):
        thread_name = QThread.currentThread().objectName()
        thread_id = str(QThread.currentThread())
        print('On GPU: {}'.format(self.using_gpu))
        print('running name:{} on id:{}'.format(thread_name, thread_id))

        terminal = [
            'python', os.path.join(segmentpyDir, 'main_retrain.py'),
            '-ckpt', self.params['ckpt path'],
            '-ep', self.params['nb epoch'],
            '-lr', self.params['lr type'],
            '-ilr', self.params['lr init'],
            '-klr', self.params['k param'],
            '-plr', self.params['period'],
            '-dv', self.using_gpu,
            '-cmt', self.params['comment'],
            '-nodes', *self.params['nodes'].split(', '),  # params['nodes'] is a str so need split
            ######## misc
            '-st', self.params['sv step'],
            '-tb', self.params['tb step'],
            '-stride', self.params['sampl. gap'],
            '-cond', self.params['stop. crit.'],
            '-corr', self.params['correction'],
            ######## paths
            '-trnd', self.params['trn repo. path'],
            '-vald', self.params['val repo. path'],
            '-tstd', self.params['tst repo. path'],
            '-logd', self.params['mdl. saved path'],
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
        self.actVs = []
        self.gradVs = []

        self.threadpool = QThreadPool()
        self.qManager.signals.available_gpu.connect(self.start)

        _translate = QtCore.QCoreApplication.translate

        # set button icons
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "play-button.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.start_button.setIcon(icon)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "exchange_2.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "exchange.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.loop_button.setIcon(icon1)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "stop.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "stop.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.stop_button.setIcon(icon2)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "plus.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon3.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "plus.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.add_button.setIcon(icon3)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "minus.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon4.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "minus.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.clean_button.setIcon(icon4)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "reply.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon5.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "reply.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.forward_button.setIcon(icon5)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "resume.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon6.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "resume.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.pushButton.setIcon(icon6)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "rubik.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon7.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "rubik.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.predict_button.setIcon(icon7)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "speedometer.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon8.addPixmap(QtGui.QPixmap(os.path.join(imgDir, "speedometer.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.dashboard_button.setIcon(icon8)

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
        item.setText(_translate("LRCSNet", "sampl. gap"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(20, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "stop. crit."))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(21, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "correction"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(22, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "trn repo. path"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(23, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "val repo. path"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(24, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "tst repo. path"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(25, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "mdl. saved path"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(26, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "ckpt path"))
        item.setBackground(QtGui.QColor(128, 128, 128))
        item = self.tableWidget.item(27, 0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setText(_translate("LRCSNet", "nodes"))
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
        self.ActViewer.triggered.connect(self.actViewer_plugin)
        self.Loss_Landscape.triggered.connect(self.loss_landscape)
        self.Random_Forest.triggered.connect(self.random_forest)
        self.Volumes_Viewer.triggered.connect(self.volViewer_plugin)
        self.Metrics.triggered.connect(self.metric_plugin)
        self.AugViewer.triggered.connect(self.augViewer_plugin)
        self.GradViewer.triggered.connect(self.gradViewer_plugin)
        # self.actionGrid_Search.triggered.connect(self.gSearch)
        self.actionExtract_Result.triggered.connect(self.resultsExtractor)

    ################# menubar methods
    def resultsExtractor(self):
        self.rltE = rltExtractor_logic()
        try:
            self.rltE.show()
        except Exception as e:
            self.log_window('Unknown error', e.args[0])

    def gradViewer_plugin(self):
        i = self.gradVs.__len__()
        gradV = gradView_logic()
        self.gradVs.append(gradV)
        try:
            self.gradVs[i].show()
        except Exception as e:
            self.log_window('Unknown error', e.args[0])

    def actViewer_plugin(self):
        i = self.actVs.__len__()
        actV = actViewer_logic()
        self.actVs.append(actV)  #note: not using self.actV to create a new instance
        try:
            self.actVs[i].show()
        except Exception as e:
            self.log_window('Unknown error', e.args[0])

    def augViewer_plugin(self):
        self.augV = augViewer_logic()
        try:
            self.augV.show()
        except Exception as e:
            self.log_window('Unknown error', e.args[0])

    def metric_plugin(self):
        self.metric = metric_logic()
        try:
            self.metric.show()
        except Exception as e:
            self.log_window('Unknown error', e.args[0])

    def volViewer_plugin(self):
        self.volViewer = volViewer_logic()
        try:
            self.volViewer.exec_()
        except Exception as e:
            self.log_window('Unknown error', e.args[0])

    def activation_plugin(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #note: here might have conflict if there's an ongoing training with GPU
        import tensorflow as tf
        # get ckpt file
        dialog = file_dialog(title='select ckpts (*.meta) to retrieve activations', type='.meta')
        ckpt_paths = dialog.openFileNamesDialog()
        logger.debug('ckpt_paths: {}'.format(ckpt_paths))
        if ckpt_paths is not None and (not not ckpt_paths):
            # get input path
            dialog = file_dialog(title='select folder which contains datas for analyses')
            data_folder = dialog.openFolderDialog()
            logger.debug('data_folder: {}'.format(data_folder))
            if data_folder is not None and (not not data_folder):
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
                            'python', os.path.join(parentDir, 'main_analytic.py'),
                            '-ckpt', *ckpt_paths,
                            '-step', *steps,
                            '-type', *types,
                            '-node', *acts,
                            '-dir', data_folder,
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
        default = {
            # 'mdl': 'model',
            # 'bat_size': 'batch size',
            # 'win_size': '512',
            # 'conv_size': 'kernel size',
            # 'nb_conv': 'conv nb',
            'act_fn': 'leaky',
            'lss_fn': 'DSC',
            'batch_norm': 'True',
            'aug': 'True',
            # 'dropout': '0.0',
            'lr_type': 'ramp',
            # 'lr_init': 'lr init',
            # 'lr_k': 'k param',
            # 'lr_p': '50',
            'cls_reg': 'classification',
            # 'comment': 'comment',
            'nb_epoch': '500',
            'sv_step': '160',
            'tb_step': '50',
            # 'gap': '50',
            # 'condition': '0.001',
            # 'correction': '1e-2',
            # 'train_dir': 'trn repo. path',
            # 'val_dir': 'val repo. path',
            # 'test_dir': 'tst repo. path',
            # 'log_dir': 'mdl. saved path',
        }
        self.dialog = dialog_logic(None)
        self.dialog.exec()  #.show() won't return because of the Qdialog attribute
        if self.dialog.result() == 1:  #cancel: 0, ok: 1
            output = self.dialog.return_params()
            # convert strings to list then loop
            new = {}
            for k, v in output.items():
                new[k] = v.replace(';', ',').split(',')
            for ks, ws, nc, bs, ilr, decay, period, dp, af, lf in product(new['conv_size'],
                                                                          new['win_size'],
                                                                          new['nb_conv'],
                                                                          new['bat_size'],
                                                                          new['lr_init'],
                                                                          new['lr_k'],
                                                                          new['lr_p'],
                                                                          new['dropout'],
                                                                          new['act_fn'],
                                                                          new['lss_fn']):
                # HPs
                default['mdl'] = new['mdl'][0]
                default['conv_size'] = ks
                default['nb_conv'] = nc
                default['win_size'] = ws
                default['bat_size'] = bs
                default['lr_init'] = ilr
                default['lr_k'] = decay
                default['lr_p'] = period
                default['dropout'] = dp
                default['act_fn'] = af
                default['lss_fn'] = lf
                default['correction'] = new['correction'][0]
                default['gap'] = new['gap'][0]
                default['condition'] = new['condition'][0]
                # others
                default['train_dir'] = new['train_dir'][0]
                default['val_dir'] = new['val_dir'][0]
                default['test_dir'] = new['test_dir'][0]
                default['log_dir'] = new['log_dir'][0]
                default['comment'] = new['comment'][0]
                logger.debug(default)
                self.write_train_column(contents=default)

    def write_train_column(self, contents: dict):
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
            'gap': 'sampl. gap',
            'condition': 'stop. crit.',
            'correction': 'correction',
            'train_dir': 'trn repo. path',
            'val_dir': 'val repo. path',
            'test_dir': 'tst repo. path',
            'log_dir': 'mdl. saved path',
        }

        nb_col = self.tableWidget.columnCount()
        self.tableWidget.setColumnCount(nb_col + 1)

        # write params
        for k, v in contents.items():
            i = self.tableWidget.findItems(pivot_table[k], Qt.MatchFlag.MatchExactly)
            self.tableWidget.setItem(i[0].row(), nb_col - 1, QTableWidgetItem(v))

        # fill somethine on the empty cell
        i = self.tableWidget.findItems('ckpt path', Qt.MatchFlag.MatchExactly)
        self.tableWidget.setItem(i[0].row(), nb_col - 1, QTableWidgetItem('None'))

        i = self.tableWidget.findItems('nodes', Qt.MatchFlag.MatchExactly)
        self.tableWidget.setItem(i[0].row(), nb_col - 1, QTableWidgetItem('None'))

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
            # output dict key --> SegmentPy column name
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
            'sv_step': 'sv step',
            'tb_step': 'tb step',
            'sampl. gap': 'sampl. gap',
            'condition': 'stop. crit.',
            'mode': 'cls/reg',
            'correction': 'correction',
            'trn repo.path': 'trn repo. path',
            'val repo.path': 'val repo. path',
            'tst repo.path': 'tst repo. path',
            'mdl. saved path': 'mdl. saved path',
        }
        self.Rdialog = resumeDialog_logic(None)
        self.Rdialog.exec()
        if self.Rdialog.result() == 1:
            output = self.Rdialog.return_params()

            nb_col = self.tableWidget.columnCount()
            self.tableWidget.setColumnCount(nb_col + 1)

            # overwrite old params
            old_hypers = string_to_hypers(output['ckpt_path']).parse()
            old_hypers['ckpt_path'] = output['ckpt_path']
            old_hypers['nb_epoch'] = int(output['extra_ep'])
            old_hypers['trn repo.path'] = output['trn repo. path']
            old_hypers['val repo.path'] = output['val repo. path']
            old_hypers['tst repo.path'] = output['tst repo. path']
            old_hypers['mdl. saved path'] = output['mdl. saved path']

            # new lr
            old_hypers['lr_init'] = output['lr_init']
            old_hypers['lr_decay'] = output['lr_decay']
            old_hypers['lr_period'] = output['lr_period']

            # new sampling/stop condition
            old_hypers['sampl. gap'] = output['gap']
            old_hypers['condition'] = output['condition']

            old_hypers['sv_step'] = output['sv step']
            old_hypers['tb_step'] = output['tb step']

            # todo: add restricted node to restore
            self.Rnodes = resumeNodes_logic(ckpt_path=old_hypers['ckpt_path'])
            self.Rnodes.exec()
            if self.Rnodes.result() == 1:
                Rnodes = self.Rnodes.get_restore_nodes()
                i = self.tableWidget.findItems('nodes', Qt.MatchFlag.MatchExactly)
                self.tableWidget.setItem(i[0].row(), nb_col - 1, QTableWidgetItem(str(Rnodes)))

            # write other params
            for k, v in old_hypers.items():
                i = self.tableWidget.findItems(pivot_table[k], Qt.MatchFlag.MatchExactly)
                self.tableWidget.setItem(i[0].row(), nb_col - 1, QTableWidgetItem(str(v)))

            # fill the empty or might throw NoneType error elsewhere
            # i = self.tableWidget.findItems('sv step', Qt.MatchFlag.MatchExactly)
            # self.tableWidget.setItem(i[0].row(), nb_col - 1, QTableWidgetItem('None'))
            # i = self.tableWidget.findItems('tb step', Qt.MatchFlag.MatchExactly)
            # self.tableWidget.setItem(i[0].row(), nb_col - 1, QTableWidgetItem('None'))

            # handle the headers
            if nb_col > 2:
                self.header.append('resume')
            else:
                self.header[1] = 'nextResume'
            # bold first column
            self.lock_params()
            self.bold(column=1)
            self.setHeader()

    def predict(self):
        self.predDialog = predictDialog_logic(None)
        self.predDialog.exec()
        if self.predDialog.result() == 1:
            ckpt_path, raw_folder, pred_folder, correction = self.predDialog.get_params()
            logger.debug('ckpt path:{}\nraw dir:{}\npred dir:{}\ncorr:{}'.format(ckpt_path, raw_folder, pred_folder, correction))
            _Worker = predict_Worker(ckpt_path=ckpt_path,
                                     pred_dir=raw_folder,
                                     save_dir=pred_folder,
                                     correction=correction)
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
            self.qManager.stop()

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
        if column > 1:
            self.tableWidget.removeColumn(column)

        elif column == 1:
            self.tableWidget.removeColumn(column)
            nb_col = self.tableWidget.columnCount()
            if nb_col < 2:
                self.tableWidget.setColumnCount(nb_col + 1)

        self.bold(column=1)
        self.popHeader(column)
        self.tableWidget.repaint()

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

                # change the internal header list
                if self.header[column - 1] == 'nextTrain':
                    self.header[column - 1] = 'train'
                elif self.header[column - 1] == 'nextResume':
                    self.header[column - 1] = 'resume'
                if self.header[column] == 'train':
                    self.header[column] = 'nextTrain'
                elif self.header[column] == 'resume':
                    self.header[column] = 'nextResume'

            # swap header
            self.header[column - 1], self.header[column] = self.header[column], self.header[column - 1]

        # reset headers
        self.setHeader()
        self.tableWidget.repaint()

    def openDashboard(self):
        self.Dashboard = dashboard_logic(None)
        self.Dashboard.show()

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
                    # if self.tableWidget.item(row, column) is not None:
                    #     # self.tableWidget.item(row, column).setFlags(QtCore.Qt.ItemIsEnabled)  # note: make it editable
                    #     pass
                    if self.tableWidget.item(row, 0).text() in ['ckpt path', 'nodes']:
                        try:
                            self.tableWidget.item(row, column).setBackground(QtGui.QColor(230, 230, 250))
                            self.tableWidget.item(row, column).setFlags(QtCore.Qt.ItemIsEditable)
                        except Exception as e:
                            logger.debug(e)

            elif 'resume' in head.lower():
                for row in range(self.tableWidget.rowCount()):
                    if self.tableWidget.item(row, 0).text() not in ['lr type', 'lr init', 'k param', 'period',
                                                                    'ckpt path', 'comment', 'nb epoch', 'nodes',
                                                                    'mdl. saved path', 'sampl. gap', 'stop. crit.', 'correction']:
                        if self.tableWidget.item(row, column) is not None:
                            self.tableWidget.item(row, column).setBackground(QtGui.QColor(230, 230, 250))
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
    app.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), 'img', 'logo.png')))

    # set ui
    ui = mainwindow_logic()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

