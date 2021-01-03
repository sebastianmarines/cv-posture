from typing import Any

from PyQt5.QtCore import QThread, QObject

from threads import MPThread, DataSave, Counter


class ThreadsMixin:
    data: Any
    mp_thread: QThread
    data_thread: QThread
    counter_thread: QThread
    mpworker: QObject
    data_worker: QObject
    counter_worker: QObject

    def start_mediapipe(self) -> None:
        self.mp_thread = QThread(parent=self)
        # Step 3: Create a mpworker object
        self.mpworker = MPThread()
        # Step 4: Move mpworker to the thread
        self.mpworker.moveToThread(self.mp_thread)
        # Step 5: Connect signals and slots
        # noinspection PyUnresolvedReferences
        self.mp_thread.started.connect(self.mpworker.run)
        self.mpworker.new_frame.connect(self.show_image)
        self.mpworker.finished.connect(self.finish_thread)
        self.mpworker.data.connect(self.store_data)
        # Step 6: Start the thread
        self.mp_thread.start()

    def send_data(self) -> None:
        self.data_thread = QThread(parent=self)
        self.data_worker = DataSave(self.data, extras=[])
        self.data_worker.moveToThread(self.data_thread)
        # noinspection PyUnresolvedReferences
        self.data_thread.started.connect(self.data_worker.run)
        self.data_worker.finished.connect(self.next)
        self.data_thread.start()

    def counter(self, time: int = 5):
        self.counter_thread = QThread(parent=self)
        self.counter_worker = Counter(time=time)
        self.counter_worker.moveToThread(self.counter_thread)
        # noinspection PyUnresolvedReferences
        self.counter_thread.started.connect(self.counter_worker.run)
        self.counter_worker.second.connect(self.update_counter)
        self.counter_worker.finished.connect(self.send_data)
        self.counter_thread.start()

    def print_msg(self):
        pass

    def toggle_button(self):
        pass

    def update_counter(self):
        pass

    def show_image(self):
        pass

    def finish_thread(self):
        pass

    def store_data(self):
        pass

    def next(self):
        pass
