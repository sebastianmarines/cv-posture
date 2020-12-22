@echo off
pyinstaller --noconfirm --log-level=WARN ^
--onefile --windowed ^
--add-data=.\venv\Lib\site-packages\mediapipe\;mediapipe ^
%1
