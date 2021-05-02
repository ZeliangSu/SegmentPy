from PySide2.QtWidgets import QApplication, QDialog, QMessageBox, QLabel, QWidget, QProgressDialog
from PySide2.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QImage
from PySide2.QtCore import Qt, QPoint, QThreadPool, QRunnable, Slot, Signal, QObject

from segmentpy._taskManager.volumes_viewer_design import Ui_volViewer
from segmentpy.tf114.metric import get_interface_3D, get_surface_3D

import sys
import os
import re
import numpy as np
import pandas as pd
import string
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm
from time import sleep
from itertools import combinations, repeat

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


def get_imgs(paths: list, norm=None):
    imgs = []
    for path in paths:
        img = np.asarray(Image.open(path))
        if norm:
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = np.asarray(Image.fromarray(img).convert('RGB'))  # work with RGB888 in QImage
        imgs.append(img)
    imgs = np.squeeze(np.stack(imgs, axis=0))  # squeeze when computing vol frac uses only 2D image, and interface 3D
    return imgs


class volViewer_logic(QDialog, Ui_volViewer):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        self.threadpool = QThreadPool()

        # backend variables
        self.vol1_fns = None
        self.vol2_fns = None
        self.area = None
        self.begin = QPoint()
        self.end = QPoint()

        # front end config
        self.setupUi(self)
        self.vol1.setAcceptDrops(True)  # fixme: this line has no effect
        self.vol2.setAcceptDrops(True)  # fixme: this line has no effect
        self.setAcceptDrops(True)
        self.Slider.setMinimum(1)
        self.Slider.setMaximum(1)
        self.Slider.valueChanged.connect(self.setLED)
        self.Slider.valueChanged.connect(self.Vol1show)
        self.Slider.valueChanged.connect(self.Vol2show)
        self.zoom_button.clicked.connect(self.zoom_state)
        self.dezoom_button.clicked.connect(self.dezoom)

    def zoom_state(self):
        if self.zoom_button.isChecked():
            self.setAcceptDrops(False)
        else:
            self.setAcceptDrops(True)

    def mousePressEvent(self, ev):
        if not self.acceptDrops():
            self.begin = ev.pos()
            self.end = ev.pos()
            # self.update()

    def mouseMoveEvent(self, ev):
        if not self.acceptDrops():
            self.end = ev.pos()
            # self.update()

    def mouseReleaseEvent(self, ev):
        if not self.acceptDrops():
            self.end = ev.pos()
            if self.end != self.begin:
                # self.update()
                if self.vol1.geometry().contains(self.begin) and self.vol1.geometry().contains(self.end):
                    if self.vol1.pixmap() is not None:
                        self.area = [self.begin.x(), self.end.x(), self.begin.y(), self.end.y()]
                        self.Vol1show(self.Slider.value())
                        self.Vol2show(self.Slider.value())

                elif self.vol2.geometry().contains(self.begin) and self.vol2.geometry().contains(self.end):
                    if self.vol2.pixmap() is not None:
                        self.area = [self.begin.x(), self.end.x(), self.begin.y(), self.end.y()]
                        self.Vol1show(self.Slider.value())
                        self.Vol2show(self.Slider.value())

            else:
                pass

    def dragEnterEvent(self, event: QDragEnterEvent):
        logger.debug('detected: {}'.format(event.mimeData().text()))
        event.accept()

    def dropEvent(self, event: QDropEvent):
        dir_or_path = event.mimeData().text().replace('file://', '').replace('\r','').replace('\n','')

        fns = []
        if os.path.isdir(dir_or_path):
            if not dir_or_path.endswith('/'):
                dir_or_path += '/'
            for f in os.listdir(dir_or_path):
                fns.append(dir_or_path + f)
        else:
            fns.append(dir_or_path)

        # sorted by the last interger preceding the .tif
        fns = sorted(fns, key=lambda x: int(re.search('(\d+)\.tif', x).group(1)))

        if self.vol1.geometry().contains(event.pos()):
            self.vol1_fns = fns.copy()
            self.Vol1show(1)  # start with 1 not 0
            self.refresh_plot_and_label(self.vol1_fns, self.accumulated_value1, self.accum_plot1)

        elif self.vol2.geometry().contains(event.pos()):
            self.vol2_fns = fns.copy()
            self.Vol2show(1)  # start with 1 not 0
            self.refresh_plot_and_label(self.vol2_fns, self.accumulated_value2, self.accum_plot2)

        else:
            # ignore if been dropped at elsewhere
            pass

        if self.vol1_fns and self.vol2_fns is not None:
            if self.vol1_fns.__len__ != self.vol2_fns.__len__:
                self.log_window(Msg='vol1 and vol2 do not have same shape')

        # set slider and LCD
        max_slide = len(self.vol1_fns if self.vol1_fns is not None else self.vol2_fns)
        self.Slider.setMaximum(max_slide)
        self.setLED(actual=1, total=max_slide)

    def log_window(self, Msg: str, title='Error'):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(Msg)
        msg.setWindowTitle(title)
        msg.exec_()

    def setLED(self, actual=1, total=None):
        self.actual_slide.display(actual)
        if total:
            self.total_slides.display(total)

    def refresh(self):
        self.vol1_fns = self.vol2_fns = None

    def Vol1show(self, fns_idx):
        # note: fns_idx start with 1 not 0 like pythonic indexing
        if self.vol1_fns is not None:
            img = get_imgs([self.vol1_fns[fns_idx - 1]], norm=True)

            # crop if zoom
            img = img.copy()
            if self.area is not None:
                img = img[self.area[0]:self.area[1], self.area[2]:self.area[3]].copy()

            self.qimg1 = QImage(img,
                          img.shape[1], img.shape[0],  # shape
                          img.shape[1] * 3,  # byte per pixel * line width; 1byte = 8bits
                          QImage.Format_RGB888)  # 256 per channel so 3bytes or 24bits
            self.pmap1 = QPixmap(self.qimg1)

            # Fit the window
            self.pmap1.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.vol1.setScaledContents(True)
            self.vol1.setPixmap(self.pmap1)
            self.vol1.update()

    def Vol2show(self, fns_idx):
        # note: fns_idx start with 1 not 0 like pythonic indexing
        if self.vol2_fns is not None:
            img = get_imgs([self.vol2_fns[fns_idx - 1]], norm=True)

            # crop if zoom
            img = img.copy()
            if self.area is not None:
                img = img[self.area[0]:self.area[1], self.area[2]:self.area[3]].copy()

            self.qimg2 = QImage(img,
                          img.shape[1], img.shape[0],
                          img.shape[1] * 3,
                          QImage.Format_RGB888)
            self.pmap2 = QPixmap(self.qimg2)

            # Fit the window
            self.pmap2.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.vol2.setScaledContents(True)
            self.vol2.setPixmap(self.pmap2)
            self.vol2.update()

    def dezoom(self):
        self.area = None
        self.Vol1show(self.Slider.value())
        self.Vol2show(self.Slider.value())

    def volFrac_interf_wrapper(self, return_dict: dict, z: int, fns: list):
        # convention: vol.shape (Z, H, W)
        len_fns = len(fns)
        # compute
        if (z > 0) and (z < len_fns):
            imgs = get_imgs(fns[z - 1: z + 2], norm=False)
        elif z == 0:
            imgs = get_imgs(fns[z: z + 2], norm=False)
            imgs = np.concatenate([-np.ones((1, *imgs.shape[1:])), imgs], axis=0)
        elif z == len_fns:
            imgs = get_imgs(fns[z - 1: z + 1], norm=False)
            imgs = np.concatenate([imgs, -np.ones((1, *imgs.shape[1:]))], axis=0)
        else:
            raise ValueError()

        _accum_nb_vx = {}
        _accum_interf = {}

        imgs = imgs.astype(np.int)

        # vol fracs
        for cls in np.unique(imgs[1]):
            nb_vx = np.where(imgs[1] == cls)[0].size
            _accum_nb_vx[cls] = nb_vx

        # note: this is a on-disk interface extractor
        # interfaces (expensive)
        for ph1, ph2 in combinations(np.unique(imgs[1]), 2):
            interf_name = '{}-{}'.format(ph1, ph2)
            # give % per slides
            interf = get_interface_3D(imgs, ph1, ph2)
            per = len(np.where(interf == 1)[0]) / interf.size
            _accum_interf[interf_name] = per

        return_dict[z] = [z, _accum_nb_vx, _accum_interf]

    def get_volFracs_interf(self, fns: list):
        '''on disk get volFracs of each slides'''

        length = len(fns)
        pbar = QProgressDialog('Scanning slides...', None, 0, length, self)
        pbar.setAutoClose(True)
        pbar.setFixedSize(300, 100)
        pbar.setWindowModality(Qt.WindowModal)  # or it won't show up

        # compute
        accum_nb_vx = pd.DataFrame({'index': np.arange(len(fns))})
        accum_interf = pd.DataFrame({'index': np.arange(len(fns))})
        total_volFrac = {}
        total_interf = {}

        pbar.show()
        # todo: the following loop can be parallelized
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []

        # multi-proc or too slow
        for z in range(len(fns)):
            p = mp.Process(target=self.volFrac_interf_wrapper,
                           args=(return_dict, z, fns))
            jobs.append(p)
            p.start()

        for z, proc in enumerate(jobs):
            proc.join()

            # update pbar
            if z % 20 == 0:
                pbar.setValue(z)
                pbar.setLabelText('Scanning slides...[%d/%d]' % (z, length))

        for results in return_dict.values():
            (z, nb_vx, interf) = results

            # write data in DataFrame
            # vol frac
            for cls, vx in nb_vx.items():
                if str(cls) not in accum_nb_vx.columns:
                    # col_name = string.ascii_lowercase[cls]  # fixme: only supported maxi 26 cls, but its enough though
                    accum_nb_vx[str(cls)] = 0
                accum_nb_vx[str(cls)].iloc[z] = vx
            # interf
            for interf_name, nb in interf.items():
                if interf_name not in accum_interf.columns:
                    accum_interf[interf_name] = 0
                accum_interf[interf_name].iloc[z] = nb

            # update pbar
            if z % 20 == 0:
                pbar.setValue(z)
                pbar.setLabelText('Plotting...[%d/%d]' % (z, length))

        total_vox = accum_nb_vx.iloc[:, 1:].values.sum()

        # quit
        pbar.setValue(length)

        # compute total volume fractions
        for col in accum_nb_vx.columns[1:]:
            total_volFrac[col] = accum_nb_vx[col].sum() / total_vox

        for col in accum_interf.columns[1:]:
            total_interf[col] = accum_interf[col].sum() / total_vox

        return total_vox, accum_nb_vx, total_volFrac, accum_interf, total_interf

    def refresh_plot_and_label(self, which_vol: list, which_label: QLabel, which_plot: QWidget):
        tt_vs, acc_vf, tvf, acc_interf, tinterf = self.get_volFracs_interf(which_vol)

        self.set_titles(which_label, tt_vs, tvf, tinterf)

        which_plot.accum_nb_vx = acc_vf

        which_plot.accum_interf = acc_interf

        which_plot.plot()

    def set_titles(self, title: QLabel, total_vs: int, total_vol_frac: dict, total_interf: dict):
        content = '\nTotal vol.frac.:'
        for cls, v in total_vol_frac.items():
            content += '\n{}: {:.4f}'.format(cls, v)
        content += '\nTotal interf.:'
        for cls, v in total_interf.items():
            content += ' \n{}: {:.4f}'.format(cls, v)

        title.setText(
            '\nTotal voxels:{}{}'.format(total_vs, content))


def test():
    app = QApplication(sys.argv)

    # set ui
    ui = volViewer_logic()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test()