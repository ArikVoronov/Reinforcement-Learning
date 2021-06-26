import yaml


class Config:
    def __init__(self):
        pass

    def to_dict(self):
        out_dict = vars(self)
        for k, v in out_dict.items():
            if isinstance(v, Config):
                out_dict[k] = v.to_dict()
        return out_dict

    @staticmethod
    def _make_config(dictionary):
        config = Config()
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(config, k, Config._make_config(v))
            else:
                setattr(config, k, v)
        return config

    @staticmethod
    def load_from_yaml(path_to_yaml):

        with open(path_to_yaml, 'rb') as file:
            config_dict = yaml.load(file, Loader=yaml.Loader)
        config = Config._make_config(config_dict)
        return config

