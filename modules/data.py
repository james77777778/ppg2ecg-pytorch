import os
import glob
import random
import numpy as np
from scipy import stats as st
import torch
from torch.utils import data

from modules.util import (get_bidmc_num, make_filter_tbme_num, rpeak_detection)


class PPGDataset(data.Dataset):
    def __init__(self, data_dir: str):
        if not os.path.exists(data_dir):
            raise FileNotFoundError
        self.filepaths = glob.glob(os.path.join(data_dir, '*.npy'))
        self.filepaths = sorted(self.filepaths)
        self.num_file = len(self.filepaths)

    def __getitem__(self, idx: int):
        pair = np.load(self.filepaths[idx])
        ppg = torch.from_numpy(pair[0].reshape(1, -1)).type(torch.FloatTensor)
        ecg = torch.from_numpy(pair[1].reshape(1, -1)).type(torch.FloatTensor)

        return {'ppg': ppg, 'ecg': ecg, 'path': self.filepaths[idx]}

    def __len__(self):
        return self.num_file


class BIDMCDataset(PPGDataset):
    def __init__(self, sigma=1., is_train=True, use_aug=True, scaled_exp=True):
        if is_train:  # hard code data_dir
            data_dir = os.path.join(
                'data', 'bidmc', 'bidmc-filtered-train')
        else:
            data_dir = os.path.join(
                'data', 'bidmc', 'bidmc-filtered-test')
        if not os.path.exists(data_dir):
            raise FileNotFoundError
        if not os.path.exists(data_dir):
            raise FileNotFoundError
        self.filepaths = glob.glob(os.path.join(data_dir, '*.npy'))
        self.filepaths = sorted(self.filepaths)
        self.num_file = len(self.filepaths)
        self.max_ins = max(
            [get_bidmc_num(os.path.basename(path))for path in self.filepaths])
        self.ws = 256  # training window size
        self.all_ws = 512  # full window size
        self.is_train = is_train
        self.use_aug = use_aug  # data augmentation
        self.rand_range = [-64, 64]
        self.sampling_rate = 125.
        self.sigma = sigma
        self.scaled_exp = scaled_exp

    def expand_rpeaks(self, rpeaks):
        if len(rpeaks) == 0:
            return np.zeros(self.all_ws), np.zeros(self.all_ws)
        normal_rpeaks = np.zeros(self.all_ws)
        onehot_rpeaks = np.zeros(self.all_ws)
        for rpeak in rpeaks:
            nor_rpeaks = st.norm.pdf(
                np.arange(0, self.all_ws), loc=rpeak, scale=self.sigma)
            normal_rpeaks = normal_rpeaks + nor_rpeaks
            # onthot expands (-50 ~ +70)
            st_ix = np.clip(
                rpeak-50//(1000/self.sampling_rate), 0, self.all_ws)
            ed_ix = np.clip(
                rpeak+70//(1000/self.sampling_rate), 0, self.all_ws)
            st_ix = np.long(st_ix)
            ed_ix = np.long(ed_ix)
            onehot_rpeaks[st_ix:ed_ix] = 1
        # scale to [0, 1]
        if self.scaled_exp:
            normal_rpeaks = (normal_rpeaks - np.min(normal_rpeaks))
            normal_rpeaks = normal_rpeaks/np.ptp(normal_rpeaks)
        return np.array(normal_rpeaks), np.array(onehot_rpeaks)

    def __getitem__(self, idx: int):
        pair = np.load(self.filepaths[idx])
        ppg = torch.from_numpy(pair[0].reshape(1, -1)).type(torch.FloatTensor)
        ecg = torch.from_numpy(pair[1].reshape(1, -1)).type(torch.FloatTensor)
        # rpeak detection
        rpeaks = rpeak_detection(pair[1], self.sampling_rate)
        normal_rpeaks, onehot_rpeaks = self.expand_rpeaks(rpeaks)
        exp_rpeaks = torch.from_numpy(
            normal_rpeaks.reshape(1, -1)).type(torch.FloatTensor)
        rpeaks_arr = torch.from_numpy(
            onehot_rpeaks.reshape(1, -1)).type(torch.LongTensor)
        # random index [-64, 64]
        if self.use_aug:
            rand = random.randint(self.rand_range[0], self.rand_range[1])
        else:
            rand = 0
        st_ix = int((self.all_ws-self.ws)*0.5+rand)
        ed_ix = st_ix + self.ws
        return {'ppg': ppg[:, st_ix:ed_ix],
                'ecg': ecg[:, st_ix:ed_ix],
                'path': self.filepaths[idx],
                'rpeaks_arr': rpeaks_arr[:, st_ix:ed_ix],
                'exp_rpeaks': exp_rpeaks[:, st_ix:ed_ix]}


class UQVITDataset(PPGDataset):
    def __init__(self, sigma=1., is_train=True, use_aug=True, scaled_exp=True):
        if is_train:  # hard code data_dir
            data_dir = os.path.join(
                'data', 'uqvitalsigns', 'uqvitalsignsdata-train')
        else:
            data_dir = os.path.join(
                'data', 'uqvitalsigns', 'uqvitalsignsdata-test')
        if not os.path.exists(data_dir):
            raise FileNotFoundError
        self.filepaths = glob.glob(os.path.join(data_dir, '*.npy'))
        self.filepaths = sorted(self.filepaths)
        self.num_file = len(self.filepaths)
        self.max_ins = max(
            [get_bidmc_num(os.path.basename(path))for path in self.filepaths])
        self.ws = 200  # training window size
        self.all_ws = 400  # full window size
        self.is_train = is_train
        self.use_aug = use_aug  # data augmentation
        self.rand_range = [-50, 50]
        self.sampling_rate = 100.
        self.sigma = sigma
        self.scaled_exp = scaled_exp

    def expand_rpeaks(self, rpeaks):
        if len(rpeaks) == 0:
            return np.zeros(self.all_ws), np.zeros(self.all_ws)
        normal_rpeaks = np.zeros(self.all_ws)
        onehot_rpeaks = np.zeros(self.all_ws)
        for rpeak in rpeaks:
            nor_rpeaks = st.norm.pdf(
                np.arange(0, self.all_ws), loc=rpeak, scale=self.sigma)
            normal_rpeaks = normal_rpeaks + nor_rpeaks
            # onthot expands (-50 ~ +70)
            st_ix = np.clip(
                rpeak-50//(1000/self.sampling_rate), 0, self.all_ws)
            ed_ix = np.clip(
                rpeak+70//(1000/self.sampling_rate), 0, self.all_ws)
            st_ix = np.long(st_ix)
            ed_ix = np.long(ed_ix)
            onehot_rpeaks[st_ix:ed_ix] = 1
        # scale to [0, 1]
        if self.scaled_exp:
            normal_rpeaks = (normal_rpeaks - np.min(normal_rpeaks))
            normal_rpeaks = normal_rpeaks/np.ptp(normal_rpeaks)
        return np.array(normal_rpeaks), np.array(onehot_rpeaks)

    def __getitem__(self, idx: int):
        pair = np.load(self.filepaths[idx])
        ppg = torch.from_numpy(pair[0].reshape(1, -1)).type(torch.FloatTensor)
        ecg = torch.from_numpy(pair[1].reshape(1, -1)).type(torch.FloatTensor)
        # rpeak detection
        rpeaks = rpeak_detection(pair[1], self.sampling_rate)
        normal_rpeaks, onehot_rpeaks = self.expand_rpeaks(rpeaks)
        exp_rpeaks = torch.from_numpy(
            normal_rpeaks.reshape(1, -1)).type(torch.FloatTensor)
        rpeaks_arr = torch.from_numpy(
            onehot_rpeaks.reshape(1, -1)).type(torch.LongTensor)
        # random index [-50, 50]
        if self.use_aug:
            rand = random.randint(self.rand_range[0], self.rand_range[1])
        else:
            rand = 0
        st_ix = int((self.all_ws-self.ws)*0.5+rand)
        ed_ix = st_ix + self.ws
        return {'ppg': ppg[:, st_ix:ed_ix],
                'ecg': ecg[:, st_ix:ed_ix],
                'path': self.filepaths[idx],
                'rpeaks_arr': rpeaks_arr[:, st_ix:ed_ix],
                'exp_rpeaks': exp_rpeaks[:, st_ix:ed_ix]}


class TBMEDataset(UQVITDataset):
    def __init__(self, data_dir: str, case_id, is_train=True):
        if not os.path.exists(data_dir):
            raise FileNotFoundError
        self.filepaths = glob.glob(os.path.join(data_dir, '*.npy'))
        self.filepaths = sorted(self.filepaths)
        if case_id is not None:
            tbme_filter = make_filter_tbme_num(case_id)
            self.filepaths = list(
                filter(tbme_filter, self.filepaths))
        self.num_file = len(self.filepaths)
        self.max_ins = max(
            [get_bidmc_num(os.path.basename(path))for path in self.filepaths])
        # print([get_bidmc_num(path)for path in self.filepaths])
        self.ws = 600
        self.all_ws = 1200
        self.is_train = is_train
        self.rand_range = [-150, 150]
        self.sampling_rate = 300.


datasets = {
    'UQVIT': UQVITDataset,
    'BIDMC': BIDMCDataset,
}
