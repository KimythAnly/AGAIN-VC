import importlib

def get_preprocessor(config):
    Preprocessor = importlib.import_module(f'.{config.preprocessor_name}', package=__package__).Preprocessor
    return Preprocessor(config)