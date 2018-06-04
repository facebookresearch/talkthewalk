import logging
from logging.handlers import RotatingFileHandler

def create_logger(save_path):
    logger = logging.getLogger()
    # Debug = write everything
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(save_path, 'a', 1000000, 1)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.INFO)
    logger.addHandler(steam_handler)

    return logger