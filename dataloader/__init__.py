from torch.utils.data import DataLoader
import logging
import importlib

logger = logging.getLogger(__name__)

def get_dataset(dset, dataset_config, njobs, metadata=None):
    # dataset = 'Dataset'
    # l = locals()
    # exec(f'from .{name} import Dataset', globals(), l)
    # exec(f'ret = {dataset}(config, args, dset, metadata)', globals(), l)
    # ret = l['ret']
    # return ret
    Dataset = importlib.import_module(f'.{dataset_config.dataset_name}', package=__package__).Dataset
    return Dataset(dset, 
        dataset_config.indexes_path, 
        dataset_config.feat,
        dataset_config.feat_path,
        dataset_config.seglen,
        njobs,
        metadata)

def get_dataloader(dset, dataloader_config, dataset):
    ret = DataLoader(dataset, **dataloader_config[dset])
    return ret

