"""Report events"""
import logging
import sys

logger = logging.getLogger("ID3")
logger.setLevel(logging.DEBUG)
ch_stdout = logging.StreamHandler(sys.stdout)
ch_stdout.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
ch_stdout.setFormatter(formatter)
logger.addHandler(ch_stdout)
