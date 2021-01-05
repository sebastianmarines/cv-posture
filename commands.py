import argparse
import glob
import os
import subprocess

import PyInstaller.__main__


def build_ui():
    """ Build al files in ui directory """
    for ui_file in glob.glob('ui\\*.ui'):
        py_name = os.path.splitext(ui_file)[0] + '.py'
        subprocess.run(['pyuic5', ui_file, '-o', py_name])


def compile_file(file: str, onefile: bool, name):
    params = f"{file} --noconfirm --log-level=WARN " \
             "--add-data=.\\venv\\Lib\\site-packages\\mediapipe\\;mediapipe " \
             "--add-data=.\\posture-images\\;.\\posture-images\\ " \
             "--add-data=.\\credentials.json;. " \
             "--add-data=.\\logo.png;. " \
             "--icon=icon.ico"
    params = params.split()

    if onefile:
        params += ["--onefile", "--windowed"]

    if name:
        params.append(f"--name={name}")

    PyInstaller.__main__.run(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["buildui", "compile"],
                        help="Command to execute (buildui or compile)")
    parser.add_argument("--file", "-file", type=str, required=False,
                        help="File to build. Just needed if command is compile")
    parser.add_argument("--onefile", "-onefile", action="store_true", required=False,
                        help="Compress app in one file")
    parser.add_argument("--name", "-name", type=str, required=False,
                        help="Output file for compiling")
    args = parser.parse_args()

    if args.command == "compile" and args.file is None:
        parser.error("--file is required")

    if args.command == "buildui" and (args.file is not None or args.onefile is not False or args.name is not None):
        parser.error("Unexpected arguments for command buildui")

    if args.command == "buildui":
        build_ui()
    elif args.command == "compile":
        compile_file(file=args.file, onefile=args.onefile, name=args.name)
