from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path

import os
import logging
import time
from loguru import logger
import sys
from .comm import comm


def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{phase}_{time_str}_rank{rank}.txt'
    final_log_file = os.path.join(final_output_dir, log_file)
    head = "%(asctime)-15s:[P:%(process)d]:" + comm.head + ' %(message)s'
    logging.basicConfig(
        filename=str(final_log_file), format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)

def setup_loguru_logger(final_output_dir, rank, phase, log_level):
    # format="{time:YYYY-MM-DD at HH:mm:ss}
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{phase}_{time_str}_rank{rank}.txt'
    final_log_file = os.path.join(final_output_dir, log_file)
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> | <cyan>{name}</cyan><cyan>:line {line}</cyan>: <level>{message}</level>", level=log_level)
    logger.add(final_log_file, colorize=True, format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> | <cyan>{name}</cyan><cyan>:line {line}</cyan>: <level>{message}</level>", level=log_level)

def create_logger(cfg, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    dataset = cfg.DATASET.DATASET
    cfg_name = cfg.NAME
    log_level = cfg.LOG_LEVEL
    final_output_dir = root_output_dir / dataset / cfg_name

    logger.info('=> creating {} ...'.format(root_output_dir))
    root_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('=> creating {} ...'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('=> setup loguru logger ...')
    setup_loguru_logger(final_output_dir, cfg.RANK, phase, log_level)

    return str(final_output_dir)

