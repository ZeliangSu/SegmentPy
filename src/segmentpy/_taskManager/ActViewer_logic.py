from PySide2.QtWidgets import QApplication, QWidget, QMessageBox, QListWidget
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import Qt

from segmentpy._taskManager.ActViewer_design import Ui_actViewer
from segmentpy._taskManager.nodes_list_logic import node_list_logic
from segmentpy._taskManager.file_dialog import file_dialog
from segmentpy.tf114.util import print_nodes_name, check_N_mkdir
from segmentpy.tf114.analytic import partialRlt_and_diff, visualize_weights

from PIL import Image
import re
import sys
import os
import numpy as np
import subprocess
import tensorflow as tf
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from operator import add

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)


class actViewer_logic(QWidget, Ui_actViewer):
    def __init__(self, *args, **kwargs):
        """order: set_ckpt() = set_input() > load_graph() > get_nodes() > load_activations()"""
        super().__init__()

        self.setupUi(self)
        self.actList.setSelectionMode(QListWidget.MultiSelection)

        self.ckptButton.clicked.connect(self.ckptFileDialog)
        self.inputButton.clicked.connect(self.inputFileDialog)
        self.load.clicked.connect(self.load_activations)
        self.saveButton.clicked.connect(self.save_selected_activations)
        self.cancelButton.clicked.connect(self.exit)
        self.ckptPathLine.editingFinished.connect(self.set_ckpt)
        self.inputPathLine.editingFinished.connect(self.set_input)
        self.corrector.editingFinished.connect(self.setCorrector)
        self.actList.doubleClicked.connect(self.set_focused_layer)
        self.actSlider.valueChanged.connect(self.display)
        self.weightSlider.valueChanged.connect(self.displayWeight)

        # variables
        self.ckpt = None
        self.input = None
        self.layer_name = None
        self.layer = None
        self.correction = None

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

    def setCorrector(self):
        self.correction = float(self.corrector.text())
        self.hyperparams['normalization'] = self.correction

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
                'correction': self.correction
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
        self.layer_name = self.actList.item(list_number.row()).text()
        self.layer = list_number.row()
        self.display(0)

    def display(self, nth=0):
        logger.debug(self.layer_name)
        logger.debug(self.layer)
        if not hasattr(self, 'activations'):
            self.get_nodes()
            self.load_activations()
        else:
            act = self.activations[self.layer_name][0]
            weight = self.kernels[self.layer]
            logger.debug('weight matrix shape: {}'.format(weight.shape))
            logger.debug('activations list len: {}'.format(len(self.activations[self.layer_name])))
            self.actSlider.setMaximum(act.shape[-1] - 1) # -1 as starts with 0

            # 1D dnn output
            if 'dnn' in self.layer_name:
                ceiling = int(np.ceil(np.sqrt(act.size)))
                tmp = np.zeros((ceiling ** 2), np.float32).ravel()
                tmp[:act.size] = act
                act = tmp.reshape(ceiling, ceiling)
            else:
                logger.debug('act shape: {}'.format(act.shape))
                logger.debug('weight shape: {}'.format(weight.shape))
                act = act[:, :, nth]
            act = (act - np.min(act)) / (np.max(act) - np.min(act)) * 255
            act = np.asarray(Image.fromarray(act).convert('RGB'))
            act = act.copy()

            # imshow
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

            # get weight
            weight = weight[:, :, :, nth]
            logger.debug('weightSlide maxi: {}'.format(weight.shape[2]))
            self.weightSlider.setMaximum(weight.shape[2] - 1)
            weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight)) * 255
            self.weight = weight.copy()
            self.displayWeight(0)

    def displayWeight(self, slide=None):
        # get weight
        fig_weight = plt.figure(figsize=(1.2, 1.2))
        fig_weight.clear()
        ax = fig_weight.add_subplot(111)
        img = np.squeeze(self.weight[:, :, slide])
        ax.imshow(img, interpolation='none', aspect='auto')
        for (y, x), z in np.ndenumerate(np.squeeze(img)):
            ax.text(x, y, '%.2f' % z, fontsize=5, ha='center', va='center',)
        ax.axis('off')
        fig_weight.canvas.draw()
        data = np.fromstring(fig_weight.canvas.tostring_rgb(), dtype=np.uint8)
        logger.debug('img shape: {}'.format(data.shape))
        logger.debug(fig_weight.canvas.get_width_height())
        logger.debug(fig_weight.canvas.get_width_height()[::-1])
        data = data.reshape(tuple(map(add, fig_weight.canvas.get_width_height()[::-1],
                                      fig_weight.canvas.get_width_height())[::-1]) + (3,))
        # plt.imshow(data)
        # plt.show()
        logger.debug('img shape: {}'.format(data.shape))
        del fig_weight


        logger.debug(slide)
        # plot weight
        self.wt = QImage(data,
                        data.shape[1],
                        data.shape[0],
                        data.shape[1] * 3, QImage.Format_RGB888)
        self.pw = QPixmap(self.wt)
        self.pw.scaled(self.width(), self.height(),
                       Qt.KeepAspectRatio,
                       Qt.SmoothTransformation
                       )
        self.weightLabel.setScaledContents(False)
        self.weightLabel.setPixmap(self.pw)
        self.weightLabel.update()
        self.weightLabel.repaint()

    def load_activations(self):
        if not self.input:
            self.log_window(title='Error!', Msg='Please indicate a input image')

        elif not self.correction:
            self.log_window(title='Error!', Msg='You forgot to put the corrector')

        else:
            self.activations = partialRlt_and_diff(paths=self.paths, hyperparams=self.hyperparams,
                                              conserve_nodes=[self.actList.item(i).text() for i in range(self.actList.count())],
                                              write_rlt=False)
            logger.debug(self.activations)

            # todo: display the weight the input and output too
            self.kern_name, self.kernels = visualize_weights(params=self.paths, write_rlt=False)
            logger.debug(self.kern_name)

    def save_selected_activations(self):
        if not self.input:
            self.log_window(title='Error!', Msg='Please indicate a input image')
        else:
            save_act_path = file_dialog(title='choose a folder to save the images', type='/').openFolderDialog()
            selected_idx = self.actList.selectionModel().selectedIndexes()
            for idx in selected_idx:
                layer_name = self.actList.item(idx.row()).text()
                rlt = np.squeeze(self.activations[layer_name])
                if rlt.ndim == 3:
                    for i in range(rlt.shape[-1]):
                        check_N_mkdir(save_act_path+layer_name.replace('/','_'))
                        Image.fromarray(rlt[:, :, i]).save(save_act_path+layer_name.replace('/','_')+'/{}.tif'.format(i))
                elif rlt.ndim == 2:
                    check_N_mkdir(save_act_path+layer_name.replace('/','_'))
                    Image.fromarray(rlt[:, :]).save(save_act_path + layer_name.replace('/','_') + '/act.tif')
                else:
                    logger.debug('got an unexpected ndim of the activations: {}'.format(rlt.ndim))

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