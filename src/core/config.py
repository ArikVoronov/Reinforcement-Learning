import yaml


class Config:

    def __init__(self, path_to_config=None, replace_dict=None):
        if path_to_config is not None:
            self.path_to_config = path_to_config
            self._replace_dict = replace_dict
            self._input_dict = dict()
            self._load_config_file(path_to_config)
            self._path_to_config = path_to_config

    def _load_config_file(self, file_path):
        with open(file_path) as file:
            data = yaml.load(file, Loader=yaml.UnsafeLoader)
        if self._replace_dict:
            for config_keys_str, value in self._replace_dict.items():
                config_keys = config_keys_str.split('.')
                latest_dict = data
                for key in config_keys[:-1]:
                    latest_dict = latest_dict[key]
                latest_dict[config_keys[-1]] = value

        return self.process_dict_data(data)

    def process_dict_data(self, input_dict):
        self._input_dict = input_dict
        for k, v in input_dict.items():
            if isinstance(v, dict):
                sub_config = Config()
                config_from_dict = sub_config.process_dict_data(v)
                setattr(self, k, config_from_dict)
            else:
                if isinstance(v, str):
                    v = self._convert_string_value(v)
                setattr(self, k, v)
                self._input_dict[k] = v
        return self

    def to_dict(self):
        return self._input_dict

    def __len__(self):
        return len(self._input_dict.keys())

    def __iter__(self):
        self._key_list = list(self._input_dict.keys())
        return self

    def __next__(self):
        if len(self._key_list) > 0:
            sub_config_name = self._key_list.pop(0)

            sub_config = getattr(self, sub_config_name)

            return sub_config_name, sub_config
        else:
            raise StopIteration

    @staticmethod
    def _convert_string_value(string_value):

        if string_value.lower() == 'true':
            converted_string_value = True
        elif string_value.lower() == 'false':
            converted_string_value = False
        elif string_value.lower() == 'none':
            converted_string_value = None
        elif string_value.isdigit():
            converted_string_value = int(string_value)
        elif string_value.replace('.', '', 1).isdigit():
            # in case there's a decimal point
            converted_string_value = float(string_value)
        elif string_value.lower().replace('e', '', 1).replace('.', '', 1).isdigit() or \
                string_value.lower().replace('e-', '', 1).replace('.', '', 1).isdigit():
            # in case there's a scientific notation, e.g 1e5
            converted_string_value = float(string_value)
        else:
            converted_string_value = string_value
        return converted_string_value
