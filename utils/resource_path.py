import os
import sys


def resource_path(relative_path) -> str:
    """
    Get path to resource for development and bundled app
    :param relative_path: The path of the file trying to access
    :return: Path to resource
    """
    try:
        # noinspection PyUnresolvedReferences
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = ""

    return os.path.join(base_path, relative_path)
