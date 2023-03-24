import numpy as np
import os
import scipy
import time
import torch
from re import split as resplit

import src.utils as utils
# from src.dataset import PluginDataset, PedalboardReverbDataset
from src.dataset import FuzzDataset


class TrainTrack(dict):
    def __init__(self):
        self.update({'current_epoch': 0,

                     'tot_train_losses': [],
                     'train_losses': [],

                     'tot_val_losses': [],
                     'val_losses': [],

                     'train_av_time': 0.0,
                     'val_av_time': 0.0,
                     'total_time': 0.0,

                     'val_loss_best': 1e12,
                     'val_losses_best': 1e12,

                     'test_loss_final': 0,
                     'test_losses_final': {},

                     'test_loss_best': 0,
                     'test_losses_best': {}})

    def restore_data(self, training_info):
        self.update(training_info)

    def train_epoch_update(self, epoch_loss, epoch_losses, ep_st_time, ep_end_time, init_time, current_ep):
        self['current_epoch'] = current_ep
        self['tot_train_losses'].append(epoch_loss)
        self['train_losses'].append(epoch_losses)

        if self['train_av_time']:
            self['train_av_time'] = (self['train_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['train_av_time'] = ep_end_time - ep_st_time

        self['total_time'] += ((init_time + ep_end_time - ep_st_time)/3600)

    def val_epoch_update(self, val_loss, val_losses, ep_st_time, ep_end_time):
        self['tot_val_losses'].append(val_loss)
        self['val_losses'].append(val_losses)

        if self['val_av_time']:
            self['val_av_time'] = (self['val_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['val_av_time'] = ep_end_time - ep_st_time

        if val_loss < self['val_loss_best']:
            self['val_loss_best'] = val_loss
            self['val_losses_best'] = val_losses


def trainer(model,
            dataset,
            train_tracker,
            writer,
            train_dataloader,
            val_dataloader,
            val_idxs,
            optimiser,
            scheduler,
            loss_functions,
            epochs,
            val_freq,
            val_patience,
            save_path,
            out_path):

    patience_counter = 0
    init_time = 0

    for epoch in range(train_tracker['current_epoch'] + 1, epochs + 1):
        ep_st_time = time.time()

        epoch_losses = model.train_epoch(train_dataloader,
                                         loss_functions,
                                         optimiser)
        epoch_loss = 0
        for loss in epoch_losses:
            epoch_loss += epoch_losses[loss]
        print(f"epoch {epoch} | \ttrain loss: \t{epoch_loss:0.4f}", end="")
        for loss in epoch_losses:
            print(f" | \t{loss}: \t{epoch_losses[loss]:0.4f}", end="")
        print()

        # ===== VALIDATION ===== #
        if epoch % val_freq == 0:
            val_ep_st_time = time.time()
            val_losses = model.val_epoch(val_dataloader,
                                         loss_functions)

            # val losses
            val_loss = 0
            for loss in val_losses:
                val_loss += val_losses[loss]
            print(f"\t\tval loss: \t{val_loss:0.4f}", end="")
            for loss in val_losses:
                print(f" | \t{loss}: \t{val_losses[loss]:0.4f}", end="")
            print()

            # update lr
            scheduler.step(val_loss)

            # save best model
            if val_loss < train_tracker['val_loss_best']:
                patience_counter = 0

                model.save_model('model_best', save_path, save_statedict=True)

                # save data
                os.system(f"rm -rf {out_path}/*")  # delete previous files
                for idx in val_idxs:
                    sample = dataset.getsample(idx)
                    # process
                    val_output = model.process_data(input=sample["input_audio"].unsqueeze(0),
                                                    params=sample["params"].unsqueeze(0))

                    # filenames
                    offset_in_seconds = sample["offset"] // sample["sr"]
                    ifile = os.path.basename(sample["input_file"])[:-4]
                    ifile = f"{ifile}_{offset_in_seconds}.wav"
                    ofile = resplit("_", os.path.basename(sample["target_file"])[:-4])
                    tfile = f"{ofile[0]}_{ofile[1]}_target_{ofile[3]}_{offset_in_seconds}.wav"
                    pfile = f"{ofile[0]}_{ofile[1]}_pred_{ofile[3]}_{offset_in_seconds}.wav"

                    # save
                    scipy.io.wavfile.write(os.path.join(out_path, ifile),
                                           sample["sr"],
                                           sample["input_audio"].numpy()[0, :])
                    scipy.io.wavfile.write(os.path.join(out_path, tfile),
                                           sample["sr"],
                                           sample["target_audio"].numpy()[0, :])
                    scipy.io.wavfile.write(os.path.join(out_path, pfile),
                                           sample["sr"],
                                           val_output.cpu().numpy()[0, 0, :])
            else:
                patience_counter += 1

            # log validation losses
            for loss in val_losses:
                val_losses[loss] = val_losses[loss].item()

            train_tracker.val_epoch_update(val_loss=val_loss.item(),
                                           val_losses=val_losses,
                                           ep_st_time=val_ep_st_time,
                                           ep_end_time=time.time())

            writer.add_scalar('Loss/Val (Tot)', val_loss, epoch)
            for loss in val_losses:
                writer.add_scalar(
                    f"Loss/Val ({loss})", val_losses[loss], epoch)

        # log training losses
        for loss in epoch_losses:
            epoch_losses[loss] = epoch_losses[loss].item()

        train_tracker.train_epoch_update(epoch_loss=epoch_loss.item(),
                                         epoch_losses=epoch_losses,
                                         ep_st_time=ep_st_time,
                                         ep_end_time=time.time(),
                                         init_time=init_time,
                                         current_ep=epoch)

        writer.add_scalar('Loss/Train (Tot)', epoch_loss, epoch)
        for loss in epoch_losses:
            writer.add_scalar(
                f"Loss/Train ({loss})", epoch_losses[loss], epoch)

        # log learning rate
        writer.add_scalar('LR/current', optimiser.param_groups[0]['lr'])

        # save model
        model.save_model('model', save_path, save_statedict=True)

        # log training stats to json
        utils.json_save(train_tracker, 'training_stats', save_path, indent=4)

        # check early stopping
        if val_patience is not None and patience_counter > val_patience:
            print("Early Stop: ", patience_counter)
            print('\nvalidation patience limit reached at epoch ' + str(epoch))
            break


def tester(model,
           dataset,
           train_tracker,
           writer,
           test_dataloader,
           test_idxs,
           loss_functions,
           last_best,
           out_path):
    assert last_best in ["last", "best"]
    test_losses = model.test_epoch(test_dataloader,
                                   loss_functions)

    test_loss = 0
    for loss in test_losses:
        test_loss += test_losses[loss]
    print(f"\ttest loss {last_best}: \t{test_loss:0.4f}", end="")
    for loss in test_losses:
        print(f" | \t{loss}: \t{test_losses[loss]:0.4f}", end="")
    print()

    # save data
    for idx in test_idxs:
        sample = dataset.getsample(idx)
        # process
        test_output = model.process_data(input=sample["input_audio"].unsqueeze(0),
                                         params=sample["params"].unsqueeze(0))

        # filenames
        offset_in_seconds = sample["offset"] // sample["sr"]
        ifile = os.path.basename(sample["input_file"])[:-4]
        ifile = f"{ifile}_{offset_in_seconds}.wav"
        ofile = resplit("_", os.path.basename(sample["target_file"])[:-4])
        tfile = f"{ofile[0]}_{ofile[1]}_target_{ofile[3]}_{offset_in_seconds}.wav"
        pfile = f"{ofile[0]}_{ofile[1]}_pred_{ofile[3]}_{offset_in_seconds}.wav"

        # save
        scipy.io.wavfile.write(os.path.join(out_path, ifile),
                               sample["sr"],
                               sample["input_audio"].numpy()[0, :])
        scipy.io.wavfile.write(os.path.join(out_path, tfile),
                               sample["sr"],
                               sample["target_audio"].numpy()[0, :])
        scipy.io.wavfile.write(os.path.join(out_path, pfile),
                               sample["sr"],
                               test_output.cpu().numpy()[0, 0, :])

    # log test losses
    for loss in test_losses:
        test_losses[loss] = test_losses[loss].item()

    if last_best == "last":
        train_tracker['test_loss_final'] = test_loss.item()
        train_tracker['test_losses_final'] = test_losses
        writer.add_scalar('Loss/Test/Last (Tot)', test_loss, 0)
        for loss in test_losses:
            writer.add_scalar(f"Loss/Test/Last ({loss})", test_losses[loss], 0)
    elif last_best == "best":
        train_tracker['test_loss_best'] = test_loss.item()
        train_tracker['test_losses_best'] = test_losses
        writer.add_scalar('Loss/Test/Best (Tot)', test_loss, 0)
        for loss in test_losses:
            writer.add_scalar(f"Loss/Test/Best ({loss})", test_losses[loss], 0)
