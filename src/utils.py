import argparse
import os
import json
import torch
from src.dataset import FuzzDataset


# GENERAL ARGS
def add_general_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--save_location", default="results", help="trained models directory")
    # data
    parser.add_argument('--num_workers', type=int, default=16)
    # training
    parser.add_argument("--epochs", type=int, default=1000, help="max epochs")
    parser.add_argument("--val_freq", type=int, default=2, help="validation frequency (in epochs)")
    parser.add_argument("--val_patience", type=int, default=25, help="validation patience or None")
    parser.add_argument("--batch_size", type=int, default=6, help="mini-batch size")
    parser.add_argument("--learn_rate", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--sched_patience", type=int, default=20, help="scheduler patience")
    parser.add_argument("--sched_factor", type=float, default=0.5, help="scheduler factor")
    # gpu
    parser.add_argument("--cuda", default=1, help="use GPU if available")
    # loss functions
    parser.add_argument("--loss_fcns", default={"L1": 0.5, "MSTFT": 0.5}, help="loss functions and weights")
    parser.add_argument("--prefilt", default="None", help="pre-emphasis filter coefficients, can also read in a csv file")
    return parser


# PREFILT ARGS
def add_prefilt_args(dict_args):
    if dict_args["prefilt"] == "a-weighting":
        # as reported in in https://ieeexplore.ieee.org/abstract/document/9052944
        dict_args["prefilt"] = [0.85, 1]
    elif dict_args["prefilt"] == "high_pass":
        # args.prefilt = [-0.85, 1] # as reported in https://ieeexplore.ieee.org/abstract/document/9052944
        # as reported in (https://www.mdpi.com/2076-3417/10/3/766/htm)
        dict_args["prefilt"] = [-0.95, 1]
    else:
        dict_args["prefilt"] = None
    return dict_args


# CUDA
def cuda_device(dict_args):
    if not torch.cuda.is_available() or dict_args["cuda"] == 0:
        torch.set_default_tensor_type("torch.FloatTensor")
        device = torch.device("cpu")
        print("\ncuda device not available/not selected")
    else:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print("\ncuda device available")
    return device


# OPTIMISER and SCHEDULER
def optim_and_sched(dict_args, model):
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=dict_args["learn_rate"],
                                 weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                           "min",
                                                           factor=0.5,
                                                           patience=dict_args["sched_patience"],
                                                           verbose=True)
    return optimiser, scheduler


# DATASET
def dataset(dict_args, device):
    dataset = FuzzDataset(root_dir=dict_args["data_rootdir"],
                          sample_length=dict_args["sample_length"],
                          preload=dict_args["preload"])
    return dataset


def dir_check(dir_name):
    dir_name = [dir_name] if not type(dir_name) == list else dir_name
    dir_path = os.path.join(*dir_name)
    if os.path.isdir(dir_path):
        pass
    else:
        os.mkdir(dir_path)


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


# def load_model(model_data, device):
#     model_meta = model_data.pop('model_data')

#     if model_meta["model_type"] == "gcntf":
#         model = GCNTF(**model_meta, device=device)
#     else:
#         raise NotImplementedError

#     if 'state_dict' in model_data:
#         state_dict = model.state_dict()
#         for each in model_data['state_dict']:
#             state_dict[each] = torch.tensor(model_data['state_dict'][each])
#         model.load_state_dict(state_dict)

#     return model
