import os
import glob
import copy
import logging

import numpy as np
import pyworld as pw
import resemblyzer
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
    parser.add_argument('--config', '-c', default='./config/preprocess.yaml', help='Configure file path.')
    # multi process
    parser.add_argument('--njobs', '-p', type=int, default=4, help='')
    # dryrun
    parser.add_argument('--dry', action='store_true', help='whether to dry run')
    # debugging mode
    parser.add_argument('--debug', action='store_true', help='debugging mode')
    # seed
    parser.add_argument('--seed', type=int, help='random seed', default=961998)

    return parser.parse_args()

def process_one(wav_path, module, output_path=''):
    y = module.load_wav(wav_path)
    basename = os.path.basename(wav_path).split('.')[0]
    if module.hparams.dtype == 'wav':
        ret = y
    elif module.hparams.dtype == 'melspectrogram':
        ret = module.wav2mel(y)
    elif module.hparams.dtype == 'f0':
        f0, sp, ap = pw.wav2world(y.astype(np.float64), module.hparams.sample_rate)
        ret = f0
        if (f0 == 0).all():
            logger.warn(f'f0 returns all zeros: {wav_path}')
    elif module.hparams.dtype == 's3prl_spec':
        ret = module.wav2s3prl_spec(y)
        if ret is None:
            logger.warn(f'S3PRL spectrogram returns NoneType: {wav_path}')
    elif module.hparams.dtype == 'resemblyzer':
        y = resemblyzer.preprocess_wav(wav_path)
        ret = module.wav2resemblyzer(y)
    else:
        logger.warn(f'Not implement feature type {module.params.dtype}')
    if output_path == '':
        return ret
    else:
        if type(ret) is np.ndarray:
            np.save(os.path.join(output_path, basename+'.npy'), ret)
        else:
            logger.warn(f'Feature {module.hparams.dtype} is not saved: {wav_path}.')
        return 1

class Processor:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.dsp_modules = {}
        for feat in config.feat_to_process:
            self.dsp_modules[feat] = Dsp(config.feat[feat])

    def process(self):
        input_path = self.config.input_path
        output_path = self.config.output_path
        for feat in self.dsp_modules:
            os.makedirs(os.path.join(output_path, feat), exist_ok=True)
            speaker_dir_list = glob.glob(os.path.join(input_path, '*'))

        for speaker_dir in tqdm(speaker_dir_list, desc=f'[Main] {input_path}'):
            self.process_speaker(speaker_dir, output_path)

    def process_speaker(self, speaker_dir, output_path):
        wav_list = glob.glob(os.path.join(speaker_dir, '*.wav'))
        for feat, module in self.dsp_modules.items():
            process = partial(process_one, module=module, output_path=os.path.join(output_path, feat))
            with ThreadPool(self.args.njobs) as pool:
                _ = list(tqdm(pool.imap(process, wav_list), total=len(wav_list), desc=f'[Sub] {speaker_dir}'))


if __name__ == '__main__':
    # arguments and config
    args = get_args()
    config = Config(args.config)

    # init preprocessor
    preprocessor = get_preprocessor(module_name=config.preprocessor_name, 
        feat_to_preprocess=config.feat_to_preprocess, feat_config=config.feat_config)
    
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
    