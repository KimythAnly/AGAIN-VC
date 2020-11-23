import os
import torch
from tqdm import tqdm
from .base import BaseAgent
from util.mylogger import get_writer

class Trainer(BaseAgent):
    def __init__(self, config, args):
        super().__init__(config, args)
        if args.load != '':
            self.ckpt_dir_flag, self.train_set, self.dev_set, self.train_loader, self.dev_loader = \
                self.load_data(ckpt_path=args.load, 
                    dataset_config=config.dataset, 
                    dataloader_config=config.dataloader,
                    njobs=args.njobs)
            self.model_state, self.step_fn = self.build_model(config.build)
            self.model_state = self.load_model(self.model_state, args.load)
        else:
            self.ckpt_dir_flag, self.train_set, self.dev_set, self.train_loader, self.dev_loader = \
                self.gen_data(ckpt_path=config.ckpt_dir, flag=config.flag,
                    dataset_config=config.dataset, 
                    dataloader_config=config.dataloader,
                    njobs=args.njobs)
            self.model_state, self.step_fn = self.build_model(config.build)
        # use customed logger
        # eg. wandb, tensorboard
        self.writer = get_writer(config, args, self.ckpt_dir_flag)

    def build_model(self, build_config):
        return super().build_model(build_config, mode='train', device=self.device)

    # ====================================================
    #  train
    # ====================================================
    def train(self, total_steps, verbose_steps, log_steps, save_steps, eval_steps):
        while self.model_state['steps'] <= total_steps:
            train_bar = tqdm(self.train_loader)
            for data in train_bar:
                self.model_state['steps'] += 1
                meta = self.step_fn(self.model_state, data)

                if self.model_state['steps'] % log_steps == 0:
                    if self.writer is None:
                        print('* self.writer is not implemented.')
                    else:
                        self.writer.log(meta['log'], step=self.model_state['steps'])
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
                            step=self.model_state['steps']
                        )
                        
                if self.model_state['steps'] % verbose_steps == 0:
                    meta['log']['steps'] = self.model_state['steps']
                    train_bar.set_postfix(meta['log'])

                if self.model_state['steps'] % save_steps == 0:
                    self.save_model(self.model_state, \
                        os.path.join(self.ckpt_dir_flag, f'steps_{self.model_state["steps"]}.pth'))
                if self.model_state['steps'] % eval_steps == 0 and self.model_state['steps'] != 0:
                    self.evaluate()

    # ====================================================
    #  evaluate
    # ====================================================
    def evaluate(self):
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