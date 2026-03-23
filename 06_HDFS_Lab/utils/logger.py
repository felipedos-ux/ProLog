"""
Logger utility for HDFS Lab experiments.
"""
import logging
import sys

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
