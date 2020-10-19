import os
from .base import BaseIndexer

class Indexer(BaseIndexer):
    def __init__(self):
        super().__init__()

    def split(self, file_list, split_all, split_train):

        train = {}
        dev = {}

        for d in file_list:
            basename = os.path.basename(d)
            speaker = basename.split('_')[0]
            if speaker in train.keys():
                if len(train[speaker]) < split_train.train:
                    train[speaker].append(basename)
                elif speaker in dev.keys():
                    dev[speaker].append(basename)
                else:
                    dev[speaker] = [basename]
            else:
                if len(train.keys()) < split_all.train:
                    train[speaker] = [basename]
                elif len(dev.keys()) <= split_all.train + split_all.dev:
                    if speaker in dev.keys():
                        dev[speaker].append(basename)
                    elif len(dev.keys()) < split_all.train + split_all.dev:
                        dev[speaker] = [basename]
        indexes = {
                'train': train,
                'dev': dev
                }

        return indexes