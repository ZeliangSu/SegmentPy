from PySide2.QtWidgets import QApplication, QWidget, QMessageBox
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import Qt

from _taskManager.ActViewer_design import Ui_actViewer
from _taskManager.nodes_list_logic import node_list_logic
from _taskManager.file_dialog import file_dialog
from util import print_nodes_name
from analytic import partialRlt_and_diff, visualize_weights

from PIL import Image
import re
import sys
import os
import numpy as np
import subprocess
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)


class actViewer_logic(QWidget, Ui_actViewer):
    def __init__(self, *args, **kwargs):
        """order: set_ckpt() = set_input() > load_graph() > get_nodes() > load_activations()"""
        super().__init__()

        self.setupUi(self)

        self.ckptButton.clicked.connect(self.ckptFileDialog)
        self.inputButton.clicked.connect(self.inputFileDialog)
        self.load.clicked.connect(self.load_activations)
        self.saveButton.clicked.connect(self.save_selected_activations)
        self.cancelButton.clicked.connect(self.exit)
        self.ckptPathLine.returnPressed.connect(self.set_ckpt)
        self.inputPathLine.returnPressed.connect(self.set_input)
        self.actList.doubleClicked.connect(self.set_focused_layer)
        self.actSlider.valueChanged.connect(self.display)

        # variables
        self.ckpt = None
        self.input = None
        self.layer = None

    def log_window(self, title: str, Msg: str):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(Msg)
        msg.setWindowTitle(title)
        msg.exec_()

    def ckptFileDialog(self):
        tmp = file_dialog(title='choose .meta file').openFileNameDialog()
        if tmp:
            self.ckptPathLine.setText(tmp)
            self.set_ckpt()

    def inputFileDialog(self):
        tmp = file_dialog(title='choose .tif for input', type='.tif').openFileNameDialog()
        if tmp:
            self.inputPathLine.setText(tmp)
            self.set_input()

    def set_ckpt(self):
        self.ckpt = self.ckptPathLine.text()
        # hit Enter or close file dialog load automatically the model

        # prepare
        if self.ckpt:
            _re = re.search('(.+)ckpt/step(\d+)\.meta', self.ckpt)
            self.step = _re.group(2)
            self.graph_def_dir = _re.group(1)
            self.paths = {
                'step': self.step,
                'working_dir': self.graph_def_dir,
                'ckpt_dir': self.graph_def_dir + 'ckpt/',
                'ckpt_path': self.graph_def_dir + 'ckpt/step{}'.format(self.step),
                'save_pb_dir': self.graph_def_dir + 'pb/',
                'save_pb_path': self.graph_def_dir + 'pb/step{}.pb'.format(self.step),
                'data_dir': self.input,
            }

            model = re.search('mdl_([A-Za-z]*\d*)', self.ckpt).group(1)

            self.hyperparams = {
                'model': model,
                'window_size': int(re.search('ps(\d+)', self.ckpt).group(1)),
                'batch_size': int(re.search('bs(\d+)', self.ckpt).group(1)),
                # 'stride': args.stride,
                'device_option': 'cpu',
                'mode': 'classification',  # todo:
                'batch_normalization': False,
                'feature_map': True if model in ['LRCS8', 'LRCS9', 'LRCS10', 'Unet3'] else False,
            }

            # get node and set the listViewWidget
            self.get_nodes()

    def set_input(self):
        self.input = self.inputPathLine.text()
        self.paths['data_dir'] = self.input

    def get_nodes(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # restore from ckpt the nodes
        tf.reset_default_graph()
        self.actList.clear()
        _ = tf.train.import_meta_graph(
            self.ckpt,
            clear_devices=True,
        )

        # get_nodes
        graph = tf.get_default_graph().as_graph_def()
        nodes = print_nodes_name(graph)
        options = []
        for node in nodes:
            tmp = re.search('(^[a-zA-Z]+\d*\/).*(leaky|relu|sigmoid|tanh|logits\/identity|up\d+\/Reshape\_4|concat\d+\/concat)$', # concat\d+\/concat for uniquely Unet
                            node)
            if tmp is not None:
                tmp = tmp.string
                options.append(tmp)
        self.actList.addItems([n for n in options])

    def set_focused_layer(self, list_number=None):
        self.layer = self.actList.item(list_number.row()).text()
        self.display()

    def display(self, nth=0):
        logger.debug(self.layer)
        if not hasattr(self, 'activations'):
            self.get_nodes()
            self.load_activations()
        else:
            act = self.activations[self.layer][0]
            self.actSlider.setMaximum(act.shape[-1] - 1) # -1 as starts with 0

            # 1D dnn output
            if 'dnn' in self.layer:
                ceiling = int(np.ceil(np.sqrt(act.size)))
                tmp = np.zeros((ceiling ** 2), np.float32).ravel()
                tmp[:act.size] = act
                act = tmp.reshape(ceiling, ceiling)
            else:
                act = act[:, :, nth]
            act = (act - np.min(act)) / (np.max(act) - np.min(act)) * 255
            act = np.asarray(Image.fromarray(act).convert('RGB'))
            act = act.copy()

            self.q = QImage(act,
                                 act.shape[1],
                                 act.shape[0],
                                 act.shape[1] * 3, QImage.Format_RGB888)
            self.p = QPixmap(self.q)
            self.p.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.Images.setScaledContents(True)
            self.Images.setPixmap(self.p)
            self.Images.update()
            self.Images.repaint()

    def load_activations(self):
        if not self.input:
            self.log_window(title='Error!', Msg='Please indicate a input image')

        else:
            self.activations = partialRlt_and_diff(paths=self.paths, hyperparams=self.hyperparams,
                                              conserve_nodes=[self.actList.item(i).text() for i in range(self.actList.count())],
                                              write_rlt=False)
            logger.debug(self.activations)

            # todo: display the weight the input and output too
            # self.kern_name, self.kernels = visualize_weights(params=self.paths, write_rlt=False)
            # logger.debug(self.kern_name)

    def save_selected_activations(self):
        if self.input:
            self.log_window(title='Error!', Msg='Please indicate a input image')
        else:
            # retrive nodes of activations
            # open nodes list dialog
            nodes_list = node_list_logic(options=list(self.activations.keys()))
            nodes_list.exec()
            if nodes_list.result() == 1:
                acts = nodes_list.return_nodes()
                types = nodes_list.return_analysis_types()
                if len(types) == 0:
                    types = ['activation']
                step = int(re.search('.+ckpt/step(\d+)\.meta', self.ckpt).group(1))
                terminal = [
                    'python', 'main_analytic.py',
                    '-ckpt', self.ckpt,
                    '-step', step,
                    '-type', *types,
                    '-node', *acts,
                    '-dir', self.input,
                ]

                logger.debug(terminal)

                proc = subprocess.Popen(
                    terminal
                )
                proc.wait()

    def exit(self):
        self.close()


def test():
    app = QApplication(sys.argv)

    # set ui
    ui = actViewer_logic()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test()