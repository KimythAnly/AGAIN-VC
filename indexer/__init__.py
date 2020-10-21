import importlib

def get_indexer(config):
    Indexer = importlib.import_module(f'.{config.indexer_name}', package=__package__).Indexer
    return Indexer()