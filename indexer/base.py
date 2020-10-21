import os
import pickle as pk
import logging
from  glob import glob

logger = logging.getLogger(__file__)

class BaseIndexer():
    def __init__(self):
        pass

    def make_indexes(self, input_path, output_path, split_all, split_train):
        logger.info(f'Starting to make indexes from {input_path}.')
        file_list = self.gen_file_list(input_path)
        indexes = self.split(file_list, split_all, split_train)

        assert len(indexes['train'].keys()) <= split_all.train

        os.makedirs(os.path.join(output_path), exist_ok=True)
        out_path = os.path.join(output_path, 'indexes.pkl')
        pk.dump(indexes, open(out_path, 'wb'))
        logger.info(f'The output file is saved to {out_path}')

    def gen_file_list(self, input_path):
        return sorted(glob(os.path.join(input_path, '*')))

    def split(self, file_list, split_all, split_train):
        raise NotImplementedError
