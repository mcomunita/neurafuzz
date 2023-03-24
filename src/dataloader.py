import torch
import numpy as np


def split(dataset,
          data_split=[0.8, 0.1, 0.1]):
    split = np.array(data_split)
    split_lengths = np.int_(split * len(dataset))
    split_idxs = [split_lengths[0], sum(split_lengths[:2]), sum(split_lengths)]

    idxs = np.arange(len(dataset))
    np.random.shuffle(idxs)

    train_idxs = idxs[:split_idxs[0]]
    val_idxs = idxs[split_idxs[0]:split_idxs[1]]
    test_idxs = idxs[split_idxs[1]:split_idxs[2]]

    return train_idxs, val_idxs, test_idxs


def dataloaders(dataset,
                train_idxs,
                val_idxs,
                test_idxs,
                batch_size,
                num_workers):

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idxs)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idxs)

    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=False,
                                                   num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size,
                                                 sampler=val_sampler,
                                                 shuffle=False,
                                                 num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size,
                                                  sampler=test_sampler,
                                                  shuffle=False,
                                                  num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader
