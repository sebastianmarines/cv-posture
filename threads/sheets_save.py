from typing import List

import gspread
from PyQt5.QtCore import QObject, pyqtSignal
from oauth2client.service_account import ServiceAccountCredentials

from utils import resource_path


class DataSave(QObject):
    finished = pyqtSignal()

    def __init__(self, data, extras: List):
        QObject.__init__(self)
        self.data = data
        self.extras = extras

    def run(self) -> None:
        scope = ['https://spreadsheets.google.com/feeds']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            resource_path('credentials.json'),
            scope
        )
        client = gspread.authorize(credentials)
        sheet = client.open_by_key('1tHJcEPP03dWHxb7JTOw2xt9tE7mTRWeSdF2ywv3lRCo')
        worksheet = sheet.sheet1
        flat_list = self.extras
        flat_list += [item.item() for array in self.data for item in array.flatten()]
        worksheet.append_row(flat_list)
        self.finished.emit()
