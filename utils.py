import logging
from logging.handlers import RotatingFileHandler
import time

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

class ProgressLogger(object):
    """
    Throttles and display progress in human readable form.
    Default throttle speed is 1 sec
    """
    def __init__(self, throttle=1, should_humanize=True):
        self.latest = time.time()
        self.throttle_speed = throttle
        self.should_humanize = should_humanize

    def humanize(self, num, suffix='B'):
        if num < 0:
            return num
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    def log(self, curr, total, width=40, force=False):
        """Displays a bar showing the current progress."""
        if curr == 0 and total == -1:
            print('[ no data received for this file ]', end='\r')
            return
        curr_time = time.time()
        if not force and curr_time - self.latest < self.throttle_speed:
            return
        else:
            self.latest = curr_time

        self.latest = curr_time
        done = min(curr * width // total, width)
        remain = width - done

        if self.should_humanize:
            curr = self.humanize(curr)
            total = self.humanize(total)

        progress = '[{}{}] {} / {}'.format(
            ''.join(['|'] * done),
            ''.join(['.'] * remain),
            curr,
            total
        )
        print(progress, end='\r')
