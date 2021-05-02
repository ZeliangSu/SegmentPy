from PySide2.QtWidgets import QDialog
from PySide2.QtWidgets import QListWidget

from segmentpy._taskManager.nodes_list_design import Ui_nodes_list


class node_list_logic(QDialog, Ui_nodes_list):
    def __init__(self, options: list, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        # dialog UI
        self.setupUi(self)
        self.listWidget.setSelectionMode(QListWidget.MultiSelection)

        # backend variables
        self.nodes = options
        self.selected = []
        self.types = []
        self.listWidget.addItems([n for n in self.nodes])

    def return_nodes(self):
        selected_idx = self.listWidget.selectionModel().selectedIndexes()
        for idx in selected_idx:
            self.selected.append(self.listWidget.item(idx.row()).text())
        return self.selected

    def return_analysis_types(self):
        if self.act.isChecked():
            self.types.append('activation')
        if self.wt.isChecked():
            self.types.append('weight')
        if self.tsne.isChecked():
            self.types.append('tsne')
        if self.l2norm.isChecked():
            self.types.append('l2')
        if self.ang.isChecked():
            self.types.append('ang')
        if self.hist.isChecked():
            self.types.append('hist')
        return self.types
