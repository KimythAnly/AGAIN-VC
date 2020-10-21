import os
import logging
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool

from util.dsp import Dsp


logger = logging.getLogger(__name__)

def preprocess_one(input_items, module, output_path=''):
    input_path, basename = input_items
    y = module.load_wav(input_path)
    if module.config.dtype == 'wav':
        ret = y
    elif module.config.dtype == 'melspectrogram':
        ret = module.wav2mel(y)
    elif module.config.dtype == 'f0':
        f0, sp, ap = pw.wav2world(y.astype(np.float64), module.config.sample_rate)
        ret = f0
        if (f0 == 0).all():
            logger.warn(f'f0 returns all zeros: {input_path}')
    elif module.config.dtype == 's3prl_spec':
        ret = module.wav2s3prl_spec(y)
        if ret is None:
            logger.warn(f'S3PRL spectrogram returns NoneType: {input_path}')
    elif module.config.dtype == 'resemblyzer':
        y = resemblyzer.preprocess_wav(input_path)
        ret = module.wav2resemblyzer(y)
    else:
        logger.warn(f'Not implement feature type {module.config.dtype}')
    if output_path == '':
        return ret
    else:
        if type(ret) is np.ndarray:
            np.save(os.path.join(output_path, f'{basename}.npy'), ret)
        else:
            logger.warn(f'Feature {module.config.dtype} is not saved: {input_path}.')
        return 1


class BasePreproceccor():
    def __init__(self, config):
        self.dsp_modules = {}
        for feat in config.feat_to_preprocess:
            self.dsp_modules[feat] = Dsp(config.feat[feat])

    def preprocess(self, input_path, output_path, feat, njobs):
        file_dict = self.gen_file_dict(input_path)
        logger.info(f'Starting to preprocess from {input_path}.')
        self.preprocess_from_file_dict(file_dict=file_dict, output_path=output_path, feat=feat, njobs=njobs)
        logger.info(f'Saving processed file to {output_path}.')
        return

    def preprocess_from_file_dict(self, file_dict, output_path, feat, njobs):
        os.makedirs(os.path.join(output_path, feat), exist_ok=True)
        module = self.dsp_modules[feat]
        task = partial(preprocess_one, module=module, output_path=os.path.join(output_path, feat))
        with ThreadPool(njobs) as pool:
            _ = list(tqdm(pool.imap(task, file_dict.items()), total=len(file_dict), desc=f'Preprocessing '))

    def gen_file_dict(self, input_path):
        raise NotImplementedError
