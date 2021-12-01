import os.path as osp
import pdb
import pickle

import ipdb
import torch
import torch.utils.data as data
import numpy as np


class HaierDataset(data.Dataset):
    def __init__(self, cfg, phase):
        self.data_root = cfg.dataset.data_root
        self.phase = phase
        self.pos = cfg.dataset.position
        self.enc_steps = cfg.model.enc_layers
        self.num_class = cfg.model.num_class
        self.dec_steps = cfg.model.query_num
        self.training = phase == 'train'
        self.inputs = []

        target_all, features_all = self.deal_dataset()
        self.sessions = target_all.keys()
        self.features_all = features_all
        for session in self.sessions:
            target = target_all[session]['anno']
            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                    range(seed, target.shape[0], 1),
                    range(seed + self.enc_steps, target.shape[0]-self.dec_steps, 1)):
                enc_target = target[start:end]
                dec_target = target[end:end + self.dec_steps]
                distance_target, class_h_target = self.get_distance_target(target[start:end])
                self.inputs.append([
                    session, start, end, enc_target, distance_target, class_h_target, dec_target
                ])

    def deal_dataset(self):
        prex = self.pos + '_' + self.phase
        pkl_file = pickle.load(open(osp.join(self.data_root, prex + '.pkl'), 'rb'))
        target_all = dict()
        features_all = dict()
        for session in pkl_file.keys():
            label_chunk = pkl_file[session]['label']
            target_all[session] = {'anno': np.eye(self.numclass)[label_chunk],
                                   'label': label_chunk,
                                   'feature_length': label_chunk.size
                                   }
            features_all[session] = pkl_file[session]['embedding']
        return target_all, features_all

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps, target_vector.shape[-1]))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                target_matrix[i, j] = target_vector[i + j, :]
        return target_matrix

    def get_distance_target(self, target_vector):
        target_matrix = np.zeros(self.enc_steps - 1)
        target_argmax = target_vector[self.enc_steps-1].argmax()
        for i in range(self.enc_steps - 1):
            if target_vector[i].argmax() == target_argmax:
                target_matrix[i] = 1.
        return target_matrix, target_vector[self.enc_steps-1]

    def __getitem__(self, index):
        """self.inputs.append([
                    session, start, end, enc_target, distance_target, class_h_target
                ])"""
        session, start, end, enc_target, distance_target, class_h_target, dec_target = self.inputs[index]
        _inputs = self.features_all[session][start:end]
        _inputs = torch.tensor(_inputs)

        distance_target = torch.tensor(distance_target)
        class_h_target = torch.tensor(class_h_target)
        dec_target = torch.tensor(dec_target)
        return _inputs, (class_h_target, dec_target)

    def __len__(self):
        return len(self.inputs)
