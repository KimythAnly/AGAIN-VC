import os
import torch
from tqdm import tqdm
from .base import BaseAgent

class Trainer(BaseAgent):
    def __init__(self, config, args):
        super().__init__(config, args)

    def build_model(self, build_config):
        return super()._build_model(build_config, mode='train')
        
    def train(self, total_steps, verbose_steps, log_steps, save_steps, eval_steps):
        steps = self.model_state['steps']
        while steps <= total_steps:
            train_bar = tqdm(self.train_loader)
            for data in train_bar:
                steps += 1
                meta = self.step_fn(self.model_state, data, steps=steps)
                message = meta['message']
                #if steps % verbose_steps == 0:
                    # train_bar.set_description(f'Train [{steps}] {message}')
                train_bar.set_postfix(meta['log'])
                if steps % log_steps == 0:
                    if self.writer is None:
                        print('* self.writer is not implemented.')
                    else:
                        self.writer.log(meta['log'], step=steps)
                        mels = meta['mels']
                        _data = {}
                        for k, v in mels.items():
                            if v.shape[1] != 80:
                                v = torch.nn.functional.interpolate(v.transpose(1,2), 80).transpose(1,2)
                            _data[k] = (v.cpu().numpy()/5+1, self.mel2wav(v))
                        self.writer.mels_summary(
                            tag='train/seen',
                            data=_data,
                            sample_rate=22050,
                            step=steps
                        )
                if steps % save_steps == 0:
                    self.save_model(self.model_state, steps, os.path.join(self.ckpt_dir_flag, f'steps_{steps}.pth'))
                if steps % eval_steps == 0 and steps != 0:
                    self.evaluate(steps)

