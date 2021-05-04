from PySide2.QtWidgets import QApplication, QDialog, QColorDialog, QVBoxLayout, QHBoxLayout

from segmentpy._taskManager.blanketColorPalette_design import Ui_Blanket
from segmentpy._taskManager.file_dialog import file_dialog

import pandas as pd
import os
# from itertools import product


import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


class clrPalette_logic(QDialog, Ui_Blanket):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
