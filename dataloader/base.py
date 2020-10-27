import pickle as pk
import logging
from functools import partial
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from torch.utils.data import Dataset
from util.transform import segment 

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, dset, indexes_path, feat, feat_path, seglen, njobs, metadata):
        super().__init__()
        if metadata is None:
            data = self.gen_metadata(dset, indexes_path, feat, feat_path, njobs)
        else:
            data = metadata
        self.seglen = seglen
        self.data = BaseDataset.drop_invalid_data(data, verbose=True)
        self.speaker2data = BaseDataset.gen_speaker2data(self.data)

    def gen_metadata(self, dset, indexes_path, feat, feat_path, njobs):
        logger.info(f'Start to generate metadata({dset}) using {indexes_path}')
        indexes = pk.load(open(indexes_path, 'rb'))
        sdata = indexes[dset]
        metadata = [(spk, d) for spk, data in sdata.items() for d in data]

        task = partial(self.sub_process, feat=feat, feat_path=feat_path)
        with ThreadPool(njobs) as pool:
            results = list(tqdm(pool.imap(task, metadata), total=len(metadata)))
        print(len(results))
        return results

    @staticmethod
    def gen_speaker2data(data):
        speaker2data = {}
        for i, d in enumerate(data):
            if d['speaker'] in speaker2data.keys():
                speaker2data[d['speaker']].append(i)
            else:
                speaker2data[d['speaker']] = [i]
        return speaker2data

    @staticmethod
    def drop_invalid_data(data, verbose=False):
        return data

    def sub_process(self):
        raise NotImplementedError

    
    def __getitem__(self, index):
        y = segment(self.data[index], seglen=self.seglen)
        return y

    def __len__(self):
        return len(self.data)