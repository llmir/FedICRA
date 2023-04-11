import itertools
import os
import random
import re
from glob import glob
import pandas as pd
import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

def pseudo_label_generator_acdc(data, seed, beta=50.00, mode='bf', img_class='odoc'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    # print("data.shape=",data.shape)
    # print("seed.unique=",np.unique(seed))
    # print("seed.shape=",seed.shape)
    # if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
    #     pseudo_label = np.zeros_like(seed)
    # else:
    #     markers = np.ones_like(seed)
    #     markers[seed == 4] = 0
    #     markers[seed == 0] = 1
    #     markers[seed == 1] = 2
    #     markers[seed == 2] = 3
    #     markers[seed == 3] = 4
    if img_class=='odoc':
        if 1 not in np.unique(seed) or 2 not in np.unique(seed):
            pseudo_label = np.zeros_like(seed)
        else:
            markers = np.ones_like(seed)
            markers[seed == 3] = 0
            markers[seed == 0] = 1
            markers[seed == 1] = 2
            markers[seed == 2] = 3
            sigma = 0.35
            data = np.array(data)
            data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                    out_range=(-1, 1))
            segmentation = random_walker(data, markers, beta, mode = 'bf', channel_axis=0)
            pseudo_label = segmentation - 1
    if img_class=='faz' or img_class == 'polyp':
        if 1 not in np.unique(seed):
            pseudo_label = np.zeros_like(seed)
        else:
            markers = np.ones_like(seed)
            markers[seed == 2] = 0
            markers[seed == 0] = 1
            markers[seed == 1] = 2
            sigma = 0.35
            data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                    out_range=(-1, 1))
            segmentation = random_walker(data, markers, beta, mode)
            pseudo_label = segmentation - 1
        # print("mask=",np.unique(pseudo_label))
    return pseudo_label


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, client="client1", sup_type="label",img_class='odoc'):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.img_class=img_class
        self.sup_type = sup_type
        self.transform = transform
        if self.img_class == 'odoc' or self.img_class == 'faz':
            train_ids, val_ids = self._get_client_ids(client)
        elif self.img_class =='polyp':
            train_ids, val_ids = self._get_client_ids_polyp(client)

        if self.split == 'train':
            self.sample_list = train_ids
        elif self.split == 'val':
            self.sample_list=val_ids
        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

        self.data_list = []
        for case in self.sample_list:
            h5f = h5py.File(self._base_dir + "/{}".format(case), 'r')
            if self.split == "train":
                image = h5f['image'][:]
                if self.sup_type == "random_walker":
                    label = pseudo_label_generator_acdc(image, h5f[self.sup_type][:], self.img_class)
                else:
                    label = h5f[self.sup_type][:]
            else:
                image = h5f['image'][:]
                label = h5f['mask'][:]
            self.data_list.append({'image': image, 'label': label})

    def _get_client_ids(self, client):
        client1_test_set = 'Domain1/test/'+pd.Series(os.listdir( self._base_dir+"/Domain1/test"))
        client1_training_set = 'Domain1/train/'+pd.Series(os.listdir( self._base_dir+"/Domain1/train"))
        client2_test_set = 'Domain2/test/'+pd.Series(os.listdir( self._base_dir+"/Domain2/test"))
        client2_training_set = 'Domain2/train/'+pd.Series(os.listdir( self._base_dir+"/Domain2/train"))
        client3_test_set = 'Domain3/test/'+pd.Series(os.listdir( self._base_dir+"/Domain3/test"))
        client3_training_set = 'Domain3/train/'+pd.Series(os.listdir( self._base_dir+"/Domain3/train"))
        client4_test_set = 'Domain4/test/'+pd.Series(os.listdir( self._base_dir+"/Domain4/test"))
        client4_training_set = 'Domain4/train/'+pd.Series(os.listdir( self._base_dir+"/Domain4/train"))
        client5_test_set = 'Domain5/test/'+pd.Series(os.listdir( self._base_dir+"/Domain5/test"))
        client5_training_set = 'Domain5/train/'+pd.Series(os.listdir( self._base_dir+"/Domain5/train"))
        client1_test_set = client1_test_set.tolist()
        client1_training_set = client1_training_set.tolist()
        client2_test_set = client2_test_set.tolist()
        client2_training_set = client2_training_set.tolist()
        client3_test_set = client3_test_set.tolist()
        client3_training_set = client3_training_set.tolist()
        client4_test_set = client4_test_set.tolist()
        client4_training_set = client4_training_set.tolist()
        client5_test_set = client5_test_set.tolist()
        client5_training_set = client5_training_set.tolist()
        
        if client == "client1":
            return [client1_training_set, client1_test_set]
        elif client == "client2":
            return [client2_training_set, client2_test_set]
        elif client == "client3":
            return [client3_training_set, client3_test_set]
        elif client == "client4":
            return [client4_training_set, client4_test_set]
        elif client == "client5":
            return [client5_training_set, client5_test_set]
        elif client == "client_all":
            client_train_all = client1_training_set + client2_training_set + client3_training_set + \
                            client4_training_set + client5_training_set
            client_test_all = client1_test_set + client2_test_set + client3_test_set + \
                            client4_test_set + client5_test_set
            return [client_train_all, client_test_all]
        else:
            return "ERROR KEY"
    def _get_client_ids_polyp(self, client):
        client1_test_set = 'Domain1/test/'+pd.Series(os.listdir( self._base_dir+"/Domain1/test"))
        client1_training_set = 'Domain1/train/'+pd.Series(os.listdir( self._base_dir+"/Domain1/train"))
        client2_test_set = 'Domain2/test/'+pd.Series(os.listdir( self._base_dir+"/Domain2/test"))
        client2_training_set = 'Domain2/train/'+pd.Series(os.listdir( self._base_dir+"/Domain2/train"))
        client3_test_set = 'Domain3/test/'+pd.Series(os.listdir( self._base_dir+"/Domain3/test"))
        client3_training_set = 'Domain3/train/'+pd.Series(os.listdir( self._base_dir+"/Domain3/train"))
        client4_test_set = 'Domain4/test/'+pd.Series(os.listdir( self._base_dir+"/Domain4/test"))
        client4_training_set = 'Domain4/train/'+pd.Series(os.listdir( self._base_dir+"/Domain4/train"))
        client1_test_set = client1_test_set.tolist()
        client1_training_set = client1_training_set.tolist()
        client2_test_set = client2_test_set.tolist()
        client2_training_set = client2_training_set.tolist()
        client3_test_set = client3_test_set.tolist()
        client3_training_set = client3_training_set.tolist()
        client4_test_set = client4_test_set.tolist()
        client4_training_set = client4_training_set.tolist()
        
        if client == "client1":
            return [client1_training_set, client1_test_set]
        elif client == "client2":
            return [client2_training_set, client2_test_set]
        elif client == "client3":
            return [client3_training_set, client3_test_set]
        elif client == "client4":
            return [client4_training_set, client4_test_set]
        elif client == "client_all":
            client_train_all = client1_training_set + client2_training_set + client3_training_set + \
                            client4_training_set
            client_test_all = client1_test_set + client2_test_set + client3_test_set + \
                            client4_test_set
            return [client_train_all, client_test_all]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        if self.split == "train":
            if self.transform:
                sample = self.transform(sample)
        sample["idx"] = idx
        # print('idx=',sample)
        return sample


def random_rot_flip(image, label, img_class):
    if img_class == 'odoc' or img_class == 'polyp':
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2))
        label = np.rot90(label, k, axes=(0, 1))
        axis = np.random.randint(1, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis-1).copy()
        return image, label
    if img_class == 'faz':
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label


def random_rotate(image, label,img_class='odoc'):
    if img_class=='faz':
        angle = np.random.randint(-45, 45)
        image = ndimage.rotate(image, angle, order=0, reshape=False, cval=0.8)
        label = ndimage.rotate(label, angle, order=0,
                            reshape=False, mode="constant", cval=2)
        return image, label

    if img_class=='odoc':
        angle = np.random.randint(-45, 45)
        image = ndimage.rotate(image, angle, axes=(1,2), order=0, reshape=False)
        label = ndimage.rotate(label, angle, axes=(0,1), order=0,reshape=False, mode="constant", cval=3)
        return image, label

    if img_class=='polyp':
        angle = np.random.randint(-45, 45)
        image = ndimage.rotate(image, angle, axes=(1,2), order=0, reshape=False)
        label = ndimage.rotate(label, angle, axes=(0,1), order=0,reshape=False, mode="constant", cval=2)
        return image, label


class RandomGenerator(object):
    def __init__(self, output_size,img_class='odoc'):
        self.output_size = output_size
        self.img_class=img_class
        

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label, img_class=self.img_class)
        if random.random() > 0.5:
            image, label = random_rotate(image, label,img_class=self.img_class)
        # x, y, z = image.shape
        # image = zoom(
        #     image, 1, order=3)
        # label = zoom(
        #     label, 1, order=3)
        image = torch.from_numpy(
            image.astype(np.float32))
        # print(image.shape)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """ 

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
