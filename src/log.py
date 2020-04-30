import logging
import time
import os
import sys


def set_up_log(args):
    sys_argv = sys.argv
    log_dir = args.log_dir
    dataset_log_dir = os.path.join(log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(dataset_log_dir):
        os.mkdir(dataset_log_dir)
    file_path = os.path.join(dataset_log_dir, '{}{}.log'.format(str(time.time()), args.dataset))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    return logger

