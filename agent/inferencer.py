import os
import torch
import logging
from glob import glob
from tqdm import tqdm
from .base import BaseAgent

logger = logging.getLogger(__name__)

def gen_wav_list(path):
    if os.path.isdir(path):
        wav_list = glob(os.path.join(path, '*.wav'))
    elif os.path.isfile(path):
        wav_list = [path]
    return wav_list

class WaveData():
    def __init__(self, path):
        self.path = path
        self.processed = False
        self.data = {}

    def set_processed(self):
        self.processed = True

    def is_processed(self):
        return self.processed
    
    def __getitem__(self, key):
        if type(key) is str:
            return self.data[key]
        else:
            raise NotImplementedError
    
    def __setitem__(self, key, value):
        if type(key) is str:
            self.data[key] = value
        else:
            raise NotImplementedError

class Inferencer(BaseAgent):
    def __init__(self, config, args):
        super().__init__(config, args)

    def build_model(self, build_config):
        return super()._build_model(build_config, mode='inference')

    def load_wav_data(self, source_path, target_path, out_path):
        # load wavefiles
        sources = gen_wav_list(source_path)
        assert(len(sources) > 0), f'Source path "{source_path}"" should be a wavefile or a directory which contains wavefiles.'
        targets = gen_wav_list(target_path)
        assert(len(targets) > 0), f'Target path "{target_path}" should be a wavefile or a directory which contains wavefiles.'
        if os.path.exists(out_path):
            assert(os.path.isdir(out_path)), f'Output path "{out_path}" should be a directory.'
        else:
            os.makedirs(out_path)
            logger.info(f'Output directory "{out_path}" is created.')
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(os.path.join(out_path, 'wav'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'plt'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'mel'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'npy'), exist_ok=True)

        for i, source in enumerate(sources):
            sources[i] = WaveData(source)
        for i, target in enumerate(targets):
            targets[i] = WaveData(target)

        return sources, targets, out_path

    # ====================================================
    #  inference
    # ====================================================
    def inference(self):
        try:
            data = next(self.dev_iter)
        except :
            self.dev_iter = iter(self.dev_loader)
            data = next(self.dev_iter)
        with torch.no_grad():

            meta = self.step_fn(self.model_state, data, train=False)

            mels = meta['mels']

            _data = {}
            for k, v in mels.items():
                if v.shape[1] != 80:
                    v = torch.nn.functional.interpolate(v.transpose(1,2), 80).transpose(1,2)
                _data[k] = (v.cpu().numpy()/5+1, self.mel2wav(v))
            self.writer.mels_summary(
                tag='dev/unseen',
                data=_data,
                sample_rate=22050,
                step=self.model_state['steps']
            )