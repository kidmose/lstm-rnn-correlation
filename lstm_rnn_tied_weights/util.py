from .__init__ import logger as modules_logger

import datetime
import socket
import logging
import os

_runid = None

def get_runid():
    global _runid
    if _runid is None:
        _runid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + socket.gethostname()
    return _runid

def get_logger(fileprefix='lstm_rnn_tied_weights'):
    """ For use in scripts/notebooks """
    logging.getLogger().handlers = []
    logger = modules_logger
    out_dir = os.path.join('log', fileprefix + '-' + get_runid())
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_prefix = os.path.join(out_dir, get_runid() + '-')

    # info log file
    infofh = logging.FileHandler(out_prefix + 'info.log')
    infofh.setLevel(logging.INFO)
    infofh.setFormatter(logging.Formatter(
        fmt='%(message)s',
    ))
    logger.addHandler(infofh)
    # verbose log file
    vfh = logging.FileHandler(out_prefix + 'verbose.log')
    vfh.setLevel(logging.DEBUG)
    vfh.setFormatter(logging.Formatter(
        fmt='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
    ))
    logger.addHandler(vfh)
    logger.info('Output prefix: '+ out_prefix)

    return logger

