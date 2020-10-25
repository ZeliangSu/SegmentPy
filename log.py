import logging
import sys


def setup_custom_logger(name, level=logging.WARNING):
    formatter = logging.Formatter(fmt='%(asctime)s, %(levelname)s \n[%(filename)s:%(lineno)d] %(message)s')
    logging.basicConfig(filename='./debug.log',
                        level=level,
                        format='%(asctime)s, %(levelname)s \n[%(filename)s:%(lineno)d] %(message)s')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

