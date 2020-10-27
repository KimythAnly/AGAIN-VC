import os
import logging
import pickle as pk
import importlib
import torch

from dataloader import get_dataset, get_dataloader
from util.vocoder import get_vocoder
from util.mytorch import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

class BaseAgent():
    def __init__(self, config, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocoder = get_vocoder(device=self.device)
        self.mel2wav = self.vocoder.mel2wav

    @staticmethod
    def build_model(build_config, mode, device):
        _build_model = importlib.import_module(f'model.{build_config.model_name}').build_model
        model_state, step_fn = _build_model(build_config, device=device, mode=mode)
        return model_state, step_fn

    @staticmethod
    def gen_data(ckpt_path, flag, dataset_config, dataloader_config, njobs):
        ckpt_dir = os.path.join(ckpt_path, '')
        ckpt_dir_flag = os.path.join(ckpt_dir, flag)
        prefix = os.path.basename(os.path.dirname(dataset_config.indexes_path))
        train_pkl = os.path.join(ckpt_dir, f'{prefix}_train.pkl')
        dev_pkl = os.path.join(ckpt_dir, f'{prefix}_dev.pkl')
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, 'dirtype'), 'w') as f:
            f.write('ckpt_dir')
        os.makedirs(ckpt_dir_flag, exist_ok=True)
        with open(os.path.join(ckpt_dir_flag, 'dirtype'), 'w') as f:
            f.write('ckpt_dir_flag')
        file_handler = logging.FileHandler(os.path.join(ckpt_dir, 'train.log'))
        logging.getLogger().addHandler(file_handler)
        if os.path.exists(train_pkl):
            logger.info(f'=> load train, dev data from {ckpt_dir}')
            train_data = pk.load(open(train_pkl, 'rb'))
            dev_data = pk.load(open(dev_pkl, 'rb'))
            train_set = get_dataset(dset='train', dataset_config=dataset_config, njobs=njobs, metadata=train_data)
            dev_set = get_dataset(dset='dev', dataset_config=dataset_config, njobs=njobs, metadata=dev_data)
        else:
            train_set = get_dataset(dset='train', dataset_config=dataset_config, njobs=njobs)
            dev_set = get_dataset(dset='dev', dataset_config=dataset_config, njobs=njobs)
            pk.dump(train_set.data, open(train_pkl, 'wb'))
            pk.dump(dev_set.data, open(dev_pkl, 'wb'))
        train_loader = get_dataloader(dset='train', dataloader_config=dataloader_config, dataset=train_set)
        dev_loader = get_dataloader(dset='dev', dataloader_config=dataloader_config, dataset=dev_set)
        
        return ckpt_dir_flag, train_set, dev_set, train_loader, dev_loader

    @staticmethod
    def load_data(ckpt_path, dataset_config, dataloader_config, njobs):
        if os.path.isdir(ckpt_path):
            d = os.path.join(ckpt_path, '')
            with open(os.path.join(d, 'dirtype'), 'r') as f:
                dirtype = f.read().strip()
            if dirtype == 'ckpt_dir':
                logger.warn(f'The ckpt_path is {ckpt_path}, the flag is not specified.')
                logger.warn(f'Use "default" flag.')
                ckpt_dir = d
                flag = 'default'
                ckpt_dir_flag = os.path.join(ckpt_dir, flag)
                ckpt_path = ckpt_dir_flag
            elif dirtype == 'ckpt_dir_flag':
                ckpt_dir_flag = os.path.dirname(d)
                ckpt_dir = os.path.dirname(ckpt_dir_flag)
                flag = os.path.basename(ckpt_dir_flag)
            else:
                raise NotImplementedError(f'Wrong dirtype: {dirtype} from {d}.')
        else:
            ckpt_dir_flag = os.path.dirname(ckpt_path)
            ckpt_dir = os.path.dirname(ckpt_dir_flag)
            flag = os.path.basename(ckpt_dir_flag)

        file_handler = logging.FileHandler(os.path.join(ckpt_dir, 'train.log'))
        logging.getLogger().addHandler(file_handler)
        logger.info(f'=> load train, dev data from {os.path.join(ckpt_dir)}')
        prefix = os.path.basename(os.path.dirname(dataset_config.indexes_path))
        train_data = pk.load(open(os.path.join(ckpt_dir, f'{prefix}_train.pkl'), 'rb'))
        dev_data = pk.load(open(os.path.join(ckpt_dir, f'{prefix}_dev.pkl'), 'rb'))

        train_set = get_dataset(dset='train', dataset_config=dataset_config, njobs=njobs, metadata=train_data)
        dev_set = get_dataset(dset='dev', dataset_config=dataset_config, njobs=njobs, metadata=dev_data)
        train_loader = get_dataloader(dset='train', dataloader_config=dataloader_config, dataset=train_set)
        dev_loader = get_dataloader(dset='dev', dataloader_config=dataloader_config, dataset=dev_set)
        
        return ckpt_dir_flag, train_set, dev_set, train_loader, dev_loader


    @staticmethod
    def save_model(model_state, save_path):
        dynamic_state = model_state['_dynamic_state']
        state = {}
        for key in dynamic_state:
            if hasattr(model_state[key], 'state_dict'):
                state[key] = model_state[key].state_dict()
            else:
                state[key] = model_state[key]
        save_checkpoint(state, save_path)

    @staticmethod
    def load_model(model_state, load_path):
        dynamic_state = model_state['_dynamic_state']
        state = load_checkpoint(load_path)
        for key in dynamic_state:
            if hasattr(model_state[key], 'state_dict'):
                try:
                    model_state[key].load_state_dict(state[key])
                except:
                    print(f'Load Model Error: key {key} not found')
            elif key in state.keys():
                try:
                    model_state[key] = state[key]
                except:
                    print('Load Other State Error')
            else:
                print(f'{key} is not in state_dict.')
        return model_state
