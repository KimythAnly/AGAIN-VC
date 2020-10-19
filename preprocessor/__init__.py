import importlib

def get_preprocessor(module_name, feat_to_preprocess, feat_config):
    Preprocessor = importlib.import_module(f'.{module_name}', package=__package__).Preprocessor
    return Preprocessor(feat_to_preprocess, feat_config)