import logging
import os

logpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))), 'log', 'debug.log')
if not os.path.exists(logpath):
    os.makedirs(os.path.dirname(logpath))


def setup_custom_logger(name, level=logging.WARNING):
    formatter = logging.Formatter(fmt='%(asctime)s, %(levelname)s \n[%(filename)s:%(lineno)d] %(message)s')
    logging.basicConfig(filename=logpath,
                        level=level,
                        format='%(asctime)s, %(levelname)s \n[%(filename)s:%(lineno)d] %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

