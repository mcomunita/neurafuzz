import os
import json
import torch
from src.gcntfilm import GCNTF


# check directory exists, make ond if it doesn't
def dir_check(dir_name):
    dir_name = [dir_name] if not type(dir_name) == list else dir_name
    dir_path = os.path.join(*dir_name)
    if os.path.isdir(dir_path):
        pass
    else:
        os.mkdir(dir_path)


# check file exists
def file_check(file_name, dir_name=''):
    assert type(file_name) == str
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    full_path = os.path.join(*dir_name, file_name)
    return os.path.isfile(full_path)


def json_save(data, file_name, dir_name='', indent=0):
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    assert type(file_name) == str
    file_name = file_name + '.json' if not file_name.endswith('.json') else file_name
    full_path = os.path.join(*dir_name, file_name)
    with open(full_path, 'w') as fp:
        json.dump(data, fp, indent=indent)


def json_load(file_name, dir_name=''):
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    file_name = file_name + '.json' if not file_name.endswith('.json') else file_name
    full_path = os.path.join(*dir_name, file_name)
    with open(full_path) as fp:
        return json.load(fp)


def load_model(model_data, device):
    model_meta = model_data.pop('model_data')

    if model_meta["model_type"] == "gcntf":
        model = GCNTF(**model_meta, device=device)
    else:
        raise NotImplementedError

    if 'state_dict' in model_data:
        state_dict = model.state_dict()
        for each in model_data['state_dict']:
            state_dict[each] = torch.tensor(model_data['state_dict'][each])
        model.load_state_dict(state_dict)

    return model
