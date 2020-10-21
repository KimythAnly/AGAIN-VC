import os
from glob import glob
from .base import BasePreproceccor

class Preprocessor(BasePreproceccor):
    def __init__(self, config):
        super().__init__(config)
    
    def gen_file_dict(self, input_path):
        speaker_dir_list = glob(os.path.join(input_path, '*'))
        file_list = []
        basename_list = []
        for speaker_dir in speaker_dir_list:
            lst = glob(os.path.join(speaker_dir, '*.wav'))
            basename_list += [os.path.basename(f) for f in lst]
            file_list += lst
        file_dict = dict(zip(file_list, basename_list))
        return file_dict
