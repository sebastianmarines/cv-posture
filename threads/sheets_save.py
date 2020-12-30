import gspread
from PyQt5.QtCore import QObject, pyqtSignal
from oauth2client.service_account import ServiceAccountCredentials


class DataSave(QObject):
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, data):
        QObject.__init__(self)
        self.data = data

        scope = ['https://spreadsheets.google.com/feeds']
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key('1tHJcEPP03dWHxb7JTOw2xt9tE7mTRWeSdF2ywv3lRCo')
        self.worksheet = sheet.sheet1

        # array = np.array([[1, 2, 3], [4, 5, 6]])
        # values_list = worksheet.col_values(1)
        # index = len(values_list) + 1
        # worksheet.update(f"A{index]", [array.flatten().tolist()])

    def run(self) -> None:
        flat_list = [item.item() for array in self.data for item in array.flatten()]
        self.worksheet.append_row(flat_list)
        self.message.emit("Finishing")
        self.finished.emit()
