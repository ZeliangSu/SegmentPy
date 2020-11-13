import sys
from PySide2.QtWidgets import QApplication, QWidget, QFileDialog, QDesktopWidget


class file_dialog(QWidget):

    def __init__(self, title='select a checkpoint file .meta', type='.meta', previous_dir=None):
        super().__init__()
        self.title = title
        self.type = type
        self.filter = filter
        self.initUI()
        self.pdir = previous_dir

    def initUI(self):
        frame = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        frame.moveCenter(centerPoint)
        self.setGeometry(frame)

    def openFolderDialog(self):
        options = QFileDialog.Options(QFileDialog.ShowDirsOnly)
        options |= QFileDialog.DontUseNativeDialog  # don't use Mac/Win/Linux dialog
        folder = QFileDialog.getExistingDirectory(self, self.title, options=options)
        return folder + '/'

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # don't use Mac/Win/Linux dialog
        file, _ = QFileDialog.getOpenFileName(self, self.title, "",
                                                "{} Files (*{});;All Files (*)".format(self.type, self.type), options=options)
        return file

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # don't use Mac/Win/Linux dialog
        files, _ = QFileDialog.getOpenFileNames(self, self.title, "",
                                                "{} Files (*{});;All Files (*)".format(self.type, self.type), options=options)
        return files


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = file_dialog()
    ex.openFileNamesDialog()
    ex.openFileNameDialog()
    sys.exit(app.exec_())