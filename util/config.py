import os
import yaml

class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct={}):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, dct):
        self.__dict__ = dct
    def todict(self):
        dct = {}
        for k, v in self.items():
            if issubclass(type(v), DotDict):
                v = v.todict()
            dct[k] = v
        return dct

class Config(DotDict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def yaml_load(path):
        ret = yaml.safe_load(open(path, 'r', encoding='utf8'))
        assert ret is not None, f'Config file {path} is empty.'
        return Config(ret)

    def __init__(self, dct):
        if type(dct) is str:
            dct = Config.yaml_load(dct)
        super().__init__(dct)
        try:
            self._name = dct['_name']
        except:
            self._name = 'Config'
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        ret = f'[{self._name}]'
        for k, v in self.items():
            if k[0] != '_':
                ret += f'\n    {k:16s}: {v}'
        return ret

    def _apply_config(self, config, replace=False):
        for k, v in config.items():
            self[k] = v

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
