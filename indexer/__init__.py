import importlib

def get_indexer(module_name):
    Indexer = importlib.import_module(f'.{module_name}', package=__package__).Indexer
    return Indexer()