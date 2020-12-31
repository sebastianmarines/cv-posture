from time import sleep

from PyQt5.QtCore import pyqtSignal, QObject


class Counter(QObject):
    second = pyqtSignal(int)

    def __init__(self, time: int):
        super(Counter, self).__init__()
        self.time = time

    def run(self) -> None:
        for i in range(self.time, -1, -1):
            self.second.emit(i)
            sleep(1)
