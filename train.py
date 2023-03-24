import argparse
import os
import time
import torch
import torchinfo
import torch.utils.tensorboard as tensorboard
import src.utils as utils

from src.dataloader import split, dataloaders
from src.trainer import TrainTrack, trainer, tester
from src.loss import LossWrapper
from src.load_model import load_model
from src.dataset import FuzzDataset
from src.gcntfilm import GCNTF

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

train_configs = [
    {
        "name": "GCNTF3",
        "model_type": "gcntf",
        "nblocks": 1,
        "nlayers": 10,
        "nchannels": 16,
        "kernel_size": 3,
        "dilation_growth": 2,
        "tfilm_block_size": 128,

        "loss_fcns": {"L1": 0.5, "MSTFT": 0.5},
        "prefilt": None,

        "label": "fuzz",
        "data_rootdir": "data/fuzz",
        "nparams": 4,
        "sample_length": 48000*5,
        "preload": False,
        "data_split": [.8, .1, .1],
        "num_workers": 0,

        "epochs": 2,
        "val_freq": 2,
        "sched_patience": 20,
        "val_patience": 40,
        "batch_size": 6,
        "learn_rate": 5e-3,

        "cuda": 1
    },
]


n_configs = len(train_configs)


for idx, tconf in enumerate(train_configs):

    parser = argparse.ArgumentParser()

    # general args
    parser = utils.add_general_args(parser)
    args = parser.parse_args()

    # add model specific args
    if tconf["model_type"] == "gcntf":
        parser = GCNTF.add_model_specific_args(parser)
    else:
        raise RuntimeError("error: model type")

    # parse general + model args
    args = parser.parse_args()

    # create dictionary with args
    dict_args = vars(args)

    # overwrite with train configuration
    dict_args.update(tconf)

    # set filter args
    dict_args.update(utils.add_prefilt_args(dict_args))

    # directory where results will be saved
    if dict_args["model_type"] == "gcntf":
        specifier = f"{idx+1}-{dict_args['name']}-{dict_args['label']}"
        specifier += f"__{dict_args['nblocks']}-{dict_args['nlayers']}-{dict_args['nchannels']}"
        specifier += f"-{dict_args['kernel_size']}-{dict_args['dilation_growth']}-{dict_args['tfilm_block_size']}"
        specifier += f"__prefilt-{dict_args['prefilt']}-bs{dict_args['batch_size']}"
    else:
        raise RuntimeError("error: model type")

    # results directory
    save_path = os.path.join(dict_args["save_location"], specifier)
    valbest_out_path = os.path.join(save_path, "val_best_out")
    test_final_out_path = os.path.join(save_path, "test_final_out")
    test_valbest_out_path = os.path.join(save_path, "test_valbest_out")
    utils.dir_check(dict_args["save_location"])
    utils.dir_check(save_path)
    utils.dir_check(valbest_out_path)
    utils.dir_check(test_final_out_path)
    utils.dir_check(test_valbest_out_path)

    print()
    print(save_path)
    print(valbest_out_path)
    print(test_final_out_path)
    print(test_valbest_out_path)

    # set the seed
    # TODO

    # cuda
    device = utils.cuda_device(dict_args)

    # init model
    if dict_args["model_type"] == "gcntf":
        model = GCNTF(**dict_args, device=device)
    else:
        raise RuntimeError("error: model type")

    # compute rf
    dict_args["rf"] = model.compute_receptive_field()

    # save model params
    model.save_model("model", save_path)

    # save settings
    utils.json_save(dict_args, "config", save_path, indent=4)
    print(f"\n* Training config {idx+1}/{n_configs}")
    print(dict_args)

    # optimiser + scheduler + loss fcns
    optimiser, scheduler = utils.optim_and_sched(dict_args, model)
    loss_functions = LossWrapper(dict_args["loss_fcns"], dict_args["prefilt"])

    # training tracker
    train_track = TrainTrack()
    writer = tensorboard.SummaryWriter(os.path.join("results", specifier))

    # dataset
    print("\n== DATASET ==")
    dataset = utils.dataset(dict_args=dict_args, device=device)

    # dataloaders
    train_idxs, val_idxs, test_idxs = split(dataset)
    train_dataloader, val_dataloader, test_dataloader = dataloaders(dataset,
                                                                    train_idxs,
                                                                    val_idxs,
                                                                    test_idxs,
                                                                    batch_size=dict_args["batch_size"],
                                                                    num_workers=dict_args["num_workers"])
    print("")
    print("full dataset: ", len(dataset))
    print("train dataset: ", len(train_idxs))
    print("val dataset: ", len(val_idxs))
    print("test dataset: ", len(test_idxs))

    # summary
    print()
    if dict_args["model_type"] == "gcntf":
        torchinfo.summary(model,
                          input_size=[(dict_args["batch_size"], 1, dict_args["rf"]),
                                      (dict_args["batch_size"], dict_args["nparams"])],
                          col_names=["input_size", "output_size", "num_params"],
                          device=device)
    else:
        raise RuntimeError("error: model type")
    
    # training tracker
    train_tracker = TrainTrack()
    writer = tensorboard.SummaryWriter(os.path.join("results", specifier))

    # ===== TRAIN ===== #
    print("\n== TRAIN ==")
    model = model.to(device)
    start_time = time.time()

    trainer(model=model,
            dataset=dataset,
            train_tracker=train_tracker,
            writer=writer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            val_idxs=val_idxs,
            optimiser=optimiser,
            scheduler=scheduler,
            loss_functions=loss_functions,
            epochs=dict_args["epochs"],
            val_freq=dict_args["val_freq"],
            val_patience=dict_args["val_patience"],
            save_path=save_path,
            out_path=valbest_out_path)
    
    # ===== TEST (last model) ===== #
    print("\n== TEST (last) ==")
    tester(model=model,
           dataset=dataset,
           train_tracker=train_tracker,
           writer=writer,
           test_dataloader=test_dataloader,
           test_idxs=test_idxs,
           loss_functions=loss_functions,
           last_best="last",
           out_path=test_final_out_path)

    # ===== TEST (best model) ===== #
    print("\n== TEST (best) ==")
    model_best = utils.json_load('model_best', save_path)
    model = load_model(model_best, device=device)
    model.device = device
    model = model.to(device)
    tester(model=model,
           dataset=dataset,
           train_tracker=train_tracker,
           writer=writer,
           test_dataloader=test_dataloader,
           test_idxs=test_idxs,
           loss_functions=loss_functions,
           last_best="best",
           out_path=test_valbest_out_path)

    # log training stats to json
    utils.json_save(train_track, 'training_stats', save_path, indent=4)

    stop_time = time.time()
    print(f"\ntraining time: {(stop_time-start_time)/60:0.2f} min")
