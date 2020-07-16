from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui

from _taskManager.mainwindow_logic import mainwindow_logic
import sys

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('./img/logo.png'))

    # set ui
    ui = mainwindow_logic()
    try:
        ui.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)
        for p in ui.proc_list:
            p[2].kill()


if __name__ == '__main__':
    main()

