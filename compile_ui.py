import glob
import os
import subprocess

for ui_file in glob.glob('ui\\*.ui'):
    py_name = os.path.splitext(ui_file)[0] + '.py'
    subprocess.run(['pyuic5', ui_file, '-o', py_name])
