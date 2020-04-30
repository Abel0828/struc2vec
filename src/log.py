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
    logging.basicConfig(filename=file_path, filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info('Create log file at {}'.format(file_path))
    logging.info('Command line executed: python ' + ' '.join(sys_argv))
    logging.info('Full args parsed:')
    logging.info(args)

