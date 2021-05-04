from PySide2.QtWidgets import QApplication
from PySide2 import QtGui

from segmentpy._taskManager.mainwindow_logic import mainwindow_logic
import sys, os, signal

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), 'img', 'logo.png')))

    # set ui
    ui = mainwindow_logic()
    try:
        ui.show()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(app.exec_())
    except Exception as e:
        logger.warn(e)
        for p in ui.proc_list:
            p[2].kill()


if __name__ == '__main__':
    main()

