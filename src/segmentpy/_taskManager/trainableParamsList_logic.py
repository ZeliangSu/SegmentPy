from segmentpy._taskManager.trainableParamsList_design import Ui_chooseParams

from PySide2.QtWidgets import QDialog

from segmentpy.tf114.analytic import get_all_trainable_variables


class resumeNodes_logic(QDialog, Ui_chooseParams):
    def __init__(self, ckpt_path=None, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.meta_path = ckpt_path
        self.wn, self.bn, _, _, self.dnn_wn, self.dnn_bn, _, _ = get_all_trainable_variables(self.meta_path)
        self.paramList.addItems(sorted(self.wn + self.bn + self.dnn_wn + self.dnn_bn))
        self.selected = []

    def get_restore_nodes(self):
        selected_idx = self.paramList.selectionModel().selectedIndexes()
        for idx in selected_idx:
            self.selected.append(self.paramList.item(idx.row()).text())
        return str(self.selected)[1:-1].replace('\'', '')  # special treatment to feed the args.parser()

