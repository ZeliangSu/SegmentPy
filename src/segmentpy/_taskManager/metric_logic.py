from PySide2.QtWidgets import QWidget, QApplication
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import Qt

from segmentpy._taskManager.metric_design import Ui_metricViewer
from segmentpy._taskManager.pooling_dialog_logic import dialog_logic
from segmentpy._taskManager.file_dialog import file_dialog
from segmentpy.tf114.metric import *
from segmentpy.tf114.util import dimension_regulator
from PIL import Image

import sys
from itertools import combinations

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


def load_img(path: str):
    try:
        img = np.asarray(Image.open(path))
        return img
    except Exception as e:
        logger.debug(e)


def handle_error(method):
    def wrapper(*args):
        try:
            method(*args)

        except Exception as e:
            logger.debug(e)
    return wrapper


class metric_logic(QWidget, Ui_metricViewer):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.current_page = 0
        self.max_page = 0
        self.pageSlider.setMinimum(0)
        self.pageSlider.setMaximum(0)
        self.accept_img_format = ['tiff', 'tif', 'jpeg', 'png']

        # backup values
        self.raw = {}
        self.diff = {}
        self.gt1 = {}
        self.gt2 = {}

        self.vol_frac1 = {}  # {0: {0: 0.123, 1:0.234...}, 1: {0: 0.123, 1:0.234...},...}
        self.vol_frac2 = {}  # {0: {0: 0.123, 1:0.234...}, 1: {0: 0.123, 1:0.234...},...}
        self.surf1 = {}  # {0: {0: 1234, 1:2345...}, 1: {0: 1234, 1:2345...},...}
        self.surf2 = {}  # {0: {0: 1234, 1:2345...}, 1: {0: 1234, 1:2345...},...}
        self.interf1 = {}  # {0: {'0-1': 1234, '1-2':2345...}, 1: {'0-1': 1234, '1-2':2345...},...}
        self.interf2 = {}  # {0: {'0-1': 1234, '1-2':2345...}, 1: {'0-1': 1234, '1-2':2345...},...}

        # buttons and slots
        self.pageSlider.valueChanged.connect(self.go2page)
        self.add_button.clicked.connect(self.add_page)
        self.clean_button.clicked.connect(self.clean)
        self.next_button.clicked.connect(self.next_page)
        self.previous_button.clicked.connect(self.previous_page)
        self.save_Button.clicked.connect(self.save_diff)

    def dragEnterEvent(self, ev):
        path = ev.mimeData().text().replace('\r', '').replace('\n', '')
        format = path.split('.')[-1]
        logger.debug('detected: {}'.format(path))
        if format in self.accept_img_format:
            ev.accept()

    def dropEvent(self, ev):
        dir_or_path = ev.mimeData().text().replace('file://', '').replace('\r', '').replace('\n', '')
        img = load_img(dir_or_path)

        if self.raw_frame.geometry().contains(ev.pos()):
            # read & show img
            self.raw[self.current_page] = img
            self.show_raw()

        elif self.gt1_frame.geometry().contains(ev.pos()):
            # read & show img
            self.gt1[self.current_page] = img
            self.gt1[self.current_page] = dimension_regulator(self.gt1[self.current_page])
            self.show_gt1()

            if self.current_page in self.gt2.keys():
                if self.gt2[self.current_page] is not None:
                    # regularize the dimensions of gt1 and gt2
                    if self.gt1[self.current_page].shape != self.gt2[self.current_page].shape:
                        d = dialog_logic()
                        d.exec_()
                        if d.result() == 1:
                            nb = int(d.return_nb_max())
                        else:
                            nb = 3
                        self.gt1[self.current_page] = dimension_regulator(self.gt1[self.current_page], maxp_times=nb)
                        self.gt2[self.current_page] = dimension_regulator(self.gt2[self.current_page], maxp_times=nb)
                        logger.debug('gt1: {}, gt2: {}'.format(
                            self.gt1[self.current_page].shape, self.gt2[self.current_page].shape)
                        )

                    # show diff
                    if self.current_page in self.diff.keys():
                        if self.diff[self.current_page] is None:
                            self.diff[self.current_page] = self.get_diff()
                    else:
                        self.diff[self.current_page] = self.get_diff()
                    self.show_diff()

            # calculate properties
            self.vol_frac1[self.current_page] = self.get_vol_frac(img)
            self.surf1[self.current_page] = self.get_surf(img)
            self.interf1[self.current_page] = self.get_interf(img)

            # show lists
            self.list_volFrac()
            self.list_surf()
            self.list_interf()

            # header
            self.change_Header()

        elif self.gt2_frame.geometry().contains(ev.pos()):
            # read & show img
            self.gt2[self.current_page] = img
            self.show_gt2()

            if self.current_page in self.gt1.keys():
                if self.gt1[self.current_page] is not None:
                    # regularize the dimensions of gt1 and gt2
                    if self.gt1[self.current_page].shape != self.gt2[self.current_page].shape:
                        d = dialog_logic()
                        d.exec_()
                        if d.result() == 1:
                            nb = int(d.return_nb_max())
                        else:
                            nb = 3
                        self.gt1[self.current_page] = dimension_regulator(self.gt1[self.current_page], maxp_times=nb)
                        self.gt2[self.current_page] = dimension_regulator(self.gt2[self.current_page], maxp_times=nb)

                    # show diff
                    if self.current_page in self.diff.keys():
                        if self.diff[self.current_page] is None:
                            self.diff[self.current_page] = self.get_diff()
                    else:
                        self.diff[self.current_page] = self.get_diff()
                    self.show_diff()

            # calculate properties
            self.vol_frac2[self.current_page] = self.get_vol_frac(img)
            self.surf2[self.current_page] = self.get_surf(img)
            self.interf2[self.current_page] = self.get_interf(img)

            # show lists
            self.list_volFrac()
            self.list_surf()
            self.list_interf()

            # header
            self.change_Header()

    def show_raw(self):
        if self.current_page in self.raw.keys():
            if self.raw[self.current_page] is not None:
                # work with RGB888 in QImage
                img = Image.fromarray(self.raw[self.current_page])
                img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                img = np.asarray(Image.fromarray(img).convert('RGB')).copy()
                Qimg = QImage(
                    img,
                    img.shape[1], img.shape[0],  # shape
                    img.shape[1] * 3,  # byte per pixel * line width; 1byte = 8bits
                    QImage.Format_RGB888
                )
                Pmap = QPixmap(Qimg)

                Pmap.scaled(self.raw_frame.width(), self.raw_frame.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.raw_frame.setScaledContents(True)
                self.raw_frame.setPixmap(Pmap)
                self.raw_frame.update()
            else:
                self.raw_frame.setText('Drop a raw tomogram here')
        else:
            self.raw_frame.setText('Drop a raw tomogram here')

    def show_diff(self):
        if self.current_page in self.diff.keys():
            if self.diff[self.current_page] is not None:
                # work with RGB888 in QImage
                img = Image.fromarray(self.diff[self.current_page])
                img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                img = np.asarray(Image.fromarray(img).convert('RGB')).copy()
                Qimg = QImage(
                    img,
                    img.shape[1], img.shape[0],  # shape
                    img.shape[1] * 3,  # byte per pixel * line width; 1byte = 8bits
                    QImage.Format_RGB888
                )
                Pmap = QPixmap(Qimg)

                Pmap.scaled(self.diff_frame.width(), self.diff_frame.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.diff_frame.setScaledContents(True)
                self.diff_frame.setPixmap(Pmap)
                self.diff_frame.update()
            else:
                self.diff_frame.setText('Difference will be shown here')
        else:
            self.diff_frame.setText('Difference will be shown here')

    def show_gt1(self):
        if self.current_page in self.gt1.keys():
            if self.gt1[self.current_page] is not None:
                # work with RGB888 in QImage
                img = Image.fromarray(self.gt1[self.current_page])
                img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                img = np.asarray(Image.fromarray(img).convert('RGB')).copy()
                Qimg = QImage(
                    img,
                    img.shape[1], img.shape[0],  # shape
                    img.shape[1] * 3,  # byte per pixel * line width; 1byte = 8bits
                    QImage.Format_RGB888
                )
                Pmap = QPixmap(Qimg)

                Pmap.scaled(self.gt1_frame.width(), self.gt1_frame.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.gt1_frame.setScaledContents(True)
                self.gt1_frame.setPixmap(Pmap)
                self.gt1_frame.update()
            else:
                self.gt1_frame.setText('Drop a (reference) segmentation here')
        else:
            self.gt1_frame.setText('Drop a (reference) segmentation here')

    def show_gt2(self):
        if self.current_page in self.gt2.keys():
            if self.gt2[self.current_page] is not None:
                # work with RGB888 in QImage
                img = Image.fromarray(self.gt2[self.current_page])
                img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                img = np.asarray(Image.fromarray(img).convert('RGB')).copy()
                Qimg = QImage(
                    img,
                    img.shape[1], img.shape[0],  # shape
                    img.shape[1] * 3,  # byte per pixel * line width; 1byte = 8bits
                    QImage.Format_RGB888
                )
                Pmap = QPixmap(Qimg)

                Pmap.scaled(self.gt2_frame.width(), self.gt2_frame.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.gt2_frame.setScaledContents(True)
                self.gt2_frame.setPixmap(Pmap)
                self.gt2_frame.update()
            else:
                self.gt2_frame.setText('Drop a segmentation here')
        else:
            self.gt2_frame.setText('Drop a segmentation here')

    def change_Header(self):
        if (self.current_page in self.gt1.keys()) and (self.current_page in self.gt2.keys())\
                and (self.gt1[self.current_page] is not None) and (self.gt2[self.current_page] is not None):
            total_pix = self.gt1[self.current_page].size
            total_acc = self.get_acc(self.gt1[self.current_page], self.gt2[self.current_page])
            total_iou = self.get_IoU(self.gt1[self.current_page], self.gt2[self.current_page])
            self.total_pix.setText('Total pixels: {}'.format(total_pix))
            self.total_acc.setText('Accuracy: {}'.format(total_acc))
            self.total_IoU.setText('IoU:\n{}'.format(str(total_iou).replace(',', '\n').replace('{','').replace('}','')))
        else:
            self.total_pix.setText('Total pixels:')
            self.total_acc.setText('Accuracy:')
            self.total_IoU.setText('IoU:')

    def list_volFrac(self):
        self.gt1_vol_frac.clear()
        if (self.current_page in self.vol_frac1.keys()) and (self.vol_frac1[self.current_page] is not None):
            for k, v in self.vol_frac1[self.current_page].items():
                self.gt1_vol_frac.addItems(['{}: {:.4f}'.format(k, v)])

        self.gt2_vol_frac.clear()
        if self.current_page in self.vol_frac2.keys():
            if self.vol_frac2[self.current_page] is not None:
                for k, v in self.vol_frac2[self.current_page].items():
                    self.gt2_vol_frac.addItems(['{}: {:.4f}'.format(k, v)])

    def list_surf(self):
        self.gt1_surf.clear()
        if self.current_page in self.surf1.keys():
            if self.surf1[self.current_page] is not None:
                if self.gt1[self.current_page] is not None:
                    for k, v in self.surf1[self.current_page].items():
                        self.gt1_surf.addItems(['{}: {} / {:.4f}'.format(k, v, v/self.gt1[self.current_page].size * 100)])

        self.gt2_surf.clear()
        if self.current_page in self.surf2.keys():
            if self.surf2[self.current_page] is not None:
                if self.gt2[self.current_page] is not None:
                    for k, v in self.surf2[self.current_page].items():
                        self.gt2_surf.addItems(['{}: {} / {:.4f}'.format(k, v, v/self.gt2[self.current_page].size * 100)])

    def list_interf(self):
        self.gt1_interf.clear()
        if self.current_page in self.interf1.keys():
            if self.interf1[self.current_page] is not None:
                if self.gt1[self.current_page] is not None:
                    for k, v in self.interf1[self.current_page].items():
                        self.gt1_interf.addItems(['{}: {} / {:.4f}'.format(k, v, v/self.gt1[self.current_page].size * 100)])

        self.gt2_interf.clear()
        if self.current_page in self.interf2.keys():
            if self.interf2[self.current_page] is not None:
                if self.gt2[self.current_page] is not None:
                    for k, v in self.interf2[self.current_page].items():
                        self.gt2_interf.addItems(['{}: {} / {:.4f}'.format(k, v, v/self.gt2[self.current_page].size * 100)])

    def refresh_page(self):
        self.show_raw()
        self.show_diff()
        self.show_gt1()
        self.show_gt2()

        self.list_volFrac()
        self.list_surf()
        self.list_interf()

        self.change_Header()

    def get_diff(self):
        diff = get_diff_map(self.gt1[self.current_page], self.gt2[self.current_page])
        return diff

    def get_vol_frac(self, vol: np.ndarray):
        vol_fracs = volume_fractions(vol)
        # vol_fracs: {0: 0.234, 1: 0.234,...}
        return vol_fracs

    def get_surf(self, vol):
        if len(vol.shape) < 3:
            vol = vol.reshape((*vol.shape, 1))
        surf = {}
        for ph in np.unique(vol):
            surf[ph] = len(np.where(get_surface(vol, ph) != 0)[0])
        # surf: {0: 1234, 1: 2345, ...}
        return surf

    def get_interf(self, vol):
        if len(vol.shape) < 3:
            vol = vol.reshape((*vol.shape, 1))
        interf = {}
        for ph1, ph2 in combinations(list(np.unique(vol)), r=2):
            interf['{}-{}'.format(ph1, ph2)] = len(np.where(get_interface(vol, ph1, ph2))[0])
        # interf: {'0-1': 1234, '1-2': 2345,...}
        return interf

    def get_IoU(self, vol: np.ndarray, gt: np.ndarray):
        iou = IoU(vol, gt)  # return a dict
        return iou

    def get_acc(self, vol: np.ndarray, gt: np.ndarray):
        acc = ACC(vol, gt)  # return a float
        return acc

    def add_page(self):
        self.max_page += 1
        self.pageSlider.setMaximum(self.max_page)

    def clean(self):
        self.raw[self.current_page] = None
        self.diff[self.current_page] = None
        self.gt1[self.current_page] = None
        self.gt2[self.current_page] = None

        self.vol_frac1[self.current_page] = None
        self.vol_frac2[self.current_page] = None
        self.interf1[self.current_page] = None
        self.interf2[self.current_page] = None

        self.refresh_page()

    def previous_page(self):
        if self.current_page >= 1:
            self.current_page -= 1
            self.go2page(self.current_page)

    def next_page(self):
        if self.current_page <= self.max_page - 1:
            self.current_page += 1
            self.go2page(self.current_page)

    def go2page(self, page=None):
        self.current_page = page
        self.pageSlider.setValue(self.current_page)
        self.refresh_page()

    def save_diff(self):
        if self.diff[self.current_page] is not None:
            save_path = file_dialog(title='select a place to save the difference image', type='/').openFolderDialog()
            if save_path is not None:
                Image.fromarray(self.diff[self.current_page]).save(save_path+'/diff.tif')



def test():
    app = QApplication(sys.argv)

    # set ui
    ui = metric_logic()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test()