import os
import numpy as np
import logging
from matplotlib import cm

logger = logging.getLogger(__name__)

viridis = cm.get_cmap('viridis')

def get_writer(config, args, ckpt_dir_flag):
    mapping = {
        'wandb': WandbWriter
    }
    if config.writer not in mapping.keys():
        Writer = BaseWriter
    else:
        Writer = mapping[config.writer]
    return Writer(config, args, ckpt_dir_flag)

class BaseWriter():
    dummy = lambda *arg, **kwargs: None
    def __init__(self, config, args, **kwargs):
        self.config = config
        self.args = args
        self.config['_args'] = args
        self.log = BaseWriter.dummy
        self.mels_summary = BaseWriter.dummy
    
    

import wandb
class WandbWriter(BaseWriter):
    def __init__(self, config, args, ckpt_dir_flag):
        super().__init__(config, args)
        if args.load != '':
            with open(os.path.join(ckpt_dir_flag, 'wid')) as f:
                wid = f.read().strip()
        else:
            wid = wandb.util.generate_id()
            if not args.dry:
                with open(os.path.join(ckpt_dir_flag, 'wid'), 'w') as f:
                    f.write(wid)

        self.wid = wid

        if not args.dry:
            if config.flag != '':
                init_name = f'{config._name}-{config.flag}-{self.wid}'
            else:
                init_name = f'{config._name}-{self.wid}'
            wandb.init(id=self.wid, name=init_name,
                 resume='allow',
                 config=self.config)
            self.log = wandb.log
            self.mels_summary = self._mels_summary

    def _mels_summary(self, tag, data, step, sample_rate=22050):
        self.log({
            f'{tag}/Melspectrogram': [wandb.Image(viridis(v[0][:,::-1]), caption=k) for k, v in data.items()],
            f'{tag}/Audio': [wandb.Audio(v[1], caption=k, sample_rate=sample_rate) for k, v in data.items()]
        }, step=step)
