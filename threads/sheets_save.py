from typing import List

import gspread
from PyQt5.QtCore import QObject, pyqtSignal
from oauth2client.service_account import ServiceAccountCredentials


class DataSave(QObject):
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, data, extras: List):
        QObject.__init__(self)
        self.data = data
        self.extras = extras

        scope = ['https://spreadsheets.google.com/feeds']
        credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        client = gspread.authorize(credentials)
        sheet = client.open_by_key('1tHJcEPP03dWHxb7JTOw2xt9tE7mTRWeSdF2ywv3lRCo')
        self.worksheet = sheet.sheet1

    def run(self) -> None:
        flat_list = [item.item() for array in self.data for item in array.flatten()]
        self.worksheet.append_row(flat_list)
        self.message.emit("Finishing")
        self.finished.emit()
