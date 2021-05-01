from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import Qt

from segmentpy._taskManager.augmentationViewer_design import Ui_augViewer
from segmentpy.tf114.input import coords_gen, stretching
from segmentpy.tf114.augmentation import random_aug
from segmentpy.tf114.util import load_img

from PIL import Image
import sys
import numpy as np

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


class augViewer_logic(QWidget, Ui_augViewer):
    def __init__(self, tn_dir='./train/', batch_size=1, window_size=512, stride=5, norm=1e-3, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)

        self.setupUi(self)

        # backend variables
        self.bs = batch_size
        self.ws = window_size
        self.stride = stride
        self.normalization = norm

        self.coords_gen = coords_gen(
            train_dir=tn_dir,
            window_size=self.ws,
            train_test_ratio=0.9,
            stride=self.stride,
            batch_size=self.bs
        )

        if np.greater(self.ws, self.coords_gen.get_min_dim()):
            logger.warning('detected window size {} larger than the smallest training dimension {}'.format(self.ws, self.coords_gen.get_min_dim()))
            self.ws = self.coords_gen.get_min_dim()
            # todo: the following should be override
            self.coords_gen = coords_gen(
                train_dir=tn_dir,
                window_size=self.ws,
                train_test_ratio=0.9,
                stride=self.stride,
                batch_size=self.bs
            )

        self.img_paths, self.window_sizes, self.xs, self.ys = self.coords_gen.get_train_args()
        self.next.clicked.connect(self.show_imgs)

    def show_imgs(self):
        # load img
        random = np.random.randint(len(self.img_paths))
        tomogram = load_img(self.img_paths[random]) / self.normalization
        tomogram = tomogram[self.xs[random]:self.xs[random] + self.window_sizes[random],
                   self.ys[random]:self.ys[random] + self.window_sizes[random]]
        # stretch
        aug, _ = stretching(tomogram.copy(), label=None,
                            x_coord=self.xs[random], y_coord=self.ys[random],
                            window_size=self.ws, stretch_max=3)
        # + augmentation
        aug, _ = random_aug(aug, tomogram.copy().reshape(*tomogram.shape, 1))

        # to RGB
        tomogram = (tomogram - np.min(tomogram)) / (np.max(tomogram) - np.min(tomogram)) * 255
        tomogram = np.asarray(Image.fromarray(tomogram).convert('RGB'))
        aug = (aug - np.min(aug)) / (np.max(aug) - np.min(aug)) * 255
        aug = np.asarray(Image.fromarray(aug).convert('RGB'))

        tomogram = tomogram.copy()
        aug = aug.copy()

        # convert to qt objects
        self.q1 = QImage(tomogram, tomogram.shape[1], tomogram.shape[0], tomogram.shape[1] * 3, QImage.Format_RGB888)
        self.q2 = QImage(aug, aug.shape[1], aug.shape[0], aug.shape[1] * 3, QImage.Format_RGB888)

        self.p1 = QPixmap(self.q1)
        self.p2 = QPixmap(self.q2)

        # Fit the window
        self.p1.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.p2.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.raw.setScaledContents(True)
        self.aug.setScaledContents(True)
        self.raw.setPixmap(self.p1)
        self.aug.setPixmap(self.p2)

        self.raw.update()
        self.raw.repaint()
        self.aug.update()
        self.aug.repaint()


def test():
    app = QApplication(sys.argv)

    # set ui
    ui = augViewer_logic(tn_dir='../train/')
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test()