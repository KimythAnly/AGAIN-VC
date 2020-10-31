import os
import copy
import logging

import numpy as np
import pyworld as pw
import resemblyzer
from glob import glob
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool

from util.parser import get_parser
from util.config import Config
from util.dsp import Dsp
from preprocessor import get_preprocessor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = get_parser(description='Preprocess')

    # config
    parser.add_argument('--config', '-c', default='./config/preprocess.yaml', help='config yaml file')
    # multi thread
    parser.add_argument('--njobs', '-p', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    # config
    args = get_args()
    config = Config(args.config)

    # build preprocessor
    preprocessor = get_preprocessor(config)
    
    # process
    for feat in config.feat_to_preprocess:
        preprocessor.preprocess(input_path=config.input_path, output_path=config.output_path, feat=feat, njobs=args.njobs)
    
    # # For debugging, only use one file.
    # input_path = config.input_path
    # output_path = config.output_path
    # feat = 's3prl_mockingjay'
    # fin = os.path.join(input_path, 'p317/p317_424.wav')
    # fout = os.path.join(output_path, feat)
    # print(fin, fout)
    # process_one(fin, processor.dsp_modules[feat], fout)
    