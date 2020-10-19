import importlib

def get_indexer(config):
    module_name = config.indexer_name
    indexer = importlib.import_module(f'.{module_name}', package=__package__).Indexer
    return indexer(config)