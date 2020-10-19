import os
import glob
from .base import BasePreproceccor

class Preprocessor(BasePreproceccor):
    def __init__(self, feat_to_preprocess, feat_config):
        super().__init__(feat_to_preprocess, feat_config)
    
    def gen_file_dict(self, input_path):
        speaker_dir_list = glob.glob(os.path.join(input_path, '*'))
        file_list = []
        basename_list = []
        for speaker_dir in speaker_dir_list:
            lst = glob.glob(os.path.join(speaker_dir, '*.wav'))
            basename_list += [os.path.basename(f) for f in lst]
            file_list += lst
        file_dict = dict(zip(file_list, basename_list))
        return file_dict

    def process_one_wav(self, file_name):
        raise NotImplementedError

