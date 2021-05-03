from PySide2.QtWidgets import QApplication, QWidget, QColorDialog, QVBoxLayout, QHBoxLayout

from segmentpy._taskManager.blanketColorPalette_design import Ui_Blanket
from segmentpy._taskManager.file_dialog import file_dialog

import pandas as pd
import os
# from itertools import product


import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


class clrPalette_logic(QWidget, Ui_Blanket):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setupUi(self)
