import os
import logging
import numpy as np
import re
from .base import BaseDataset
from util.transform import segment, random_scale

logger = logging.getLogger(__name__)

class Dataset(BaseDataset):
    def __init__(self, dset, indexes_path, feat, feat_path, seglen, njobs, metadata):
        super().__init__(dset, indexes_path, feat, feat_path, seglen, njobs, metadata)

    def sub_process(self, each_data, feat, feat_path):
        speaker = each_data[0]
        basename = each_data[1]
        # print(each_data, feat, feat_path)
        ret = {}
        for f in feat:
            path = os.path.join(feat_path, f, basename)
            if os.path.isfile(path):
                ret[f] = np.load(path)
            else:
                logger.info(f'Skip {path} {f}: invalid file.')
                return
        ret['speaker'] = speaker
        return ret
    
    def __getitem__(self, index):
        speaker = self.data[index]['speaker']
        mel = self.data[index]['mel']

        mel, r = segment(mel, return_r=True, seglen=self.seglen)
        pos = random_scale(mel)

        candidates = list(self.speaker2data.keys())

        neg_i = np.random.choice(len(candidates) - 1)
        if candidates[neg_i] == speaker:
            neg_speaker = candidates[-1]
        else:
            neg_speaker = candidates[neg_i]
        
        candidates = self.speaker2data[neg_speaker]
        if len(candidates) == 1:
            import pdb; pdb.set_trace()
        
        neg_i = np.random.choice(len(candidates) - 1)
        # print(candidates[neg_i], len(self.data))
        neg = self.data[candidates[neg_i]]['mel']
        neg = segment(neg, seglen=self.seglen)
        neg = random_scale(neg)


        meta = {
            'speaker': speaker,
            'mel': mel,
            'pos': pos,
            'neg': neg
        }
        return meta