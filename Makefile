my_ui:
	python compile_ui.py

pyinstaller:
ifdef onefile
	pyinstaller --noconfirm --log-level=WARN --onefile --windowed \
	--add-data=.\venv\Lib\site-packages\mediapipe\;mediapipe \
	--add-data=.\posture-images\;.\posture-images\ \
	--add-data=.\credentials.json;. $(file)
else
	pyinstaller --noconfirm --log-level=WARN \
	--add-data=.\venv\Lib\site-packages\mediapipe\;mediapipe \
	--add-data=.\posture-images\;.\posture-images\ \
	--add-data=.\credentials.json;. $(file)
endif