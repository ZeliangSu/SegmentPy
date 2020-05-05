from PyQt5.QtCore import pyqtSignal, QThreadPool, QThread, QObject, QRunnable, pyqtSlot
from PyQt5.QtWidgets import QMainWindow,  QApplication, QTableWidgetItem, QErrorMessage, QMessageBox, QGraphicsView
from PyQt5 import QtCore, QtGui

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tensorboard_extractor import lr_curve_extractor


class Canvas(QGraphicsView):
    def __init__(self):
        QtGui.QGraphicsView.__init__(self)

        scene = QtGui.QGraphicsScene(self)
        self.scene = scene

        figure = Figure()
        axes = figure.gca()
        axes.set_title("title")
        axes.plot(plt.contourf(xx, yy, Z,cmap=plt.cm.autumn, alpha=0.8))
        canvas = FigureCanvas(figure)
        canvas.setGeometry(0, 0, 500, 500)
        scene.addWidget(canvas)

        self.setScene(scene)