from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage


def cv2_to_qimage(img) -> QImage:
    h, w, ch = img.shape
    bytes_per_line = ch * w
    convert_to_qt_format = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
    return p
