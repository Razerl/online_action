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
        anno_file = open(osp.join(self.data_root, 'annotations', 'anno0529_' + prex + '_online_0918.txt'), 'rb')
        labels_file = pickle.load(open(osp.join(self.data_root, 'result', prex + '.pkl'), 'rb'))
        features_file = pickle.load(open(osp.join(self.data_root, 'result', prex + '_embedding.pkl'), 'rb'))
        ipdb.set_trace()
        annos = anno_file.readlines()
        target_all = dict()
        features_all = dict()
        for i in range(len(annos)):
            session = annos[i].strip().split()[0]
            _, _, label_video, label_chunk = labels_file[i]
            bg, fg = np.ones_like(label_chunk) * (self.numclass - 1), np.ones_like(label_chunk) * label_video
            label_chunk = np.where(label_chunk == 0, bg, fg).astype(np.int)
            if session not in target_all.keys():
                target_all[session] = [np.eye(self.numclass)[label_chunk]]
                features_all[session] = [features_file[i]]
            else:
                target_all[session] += [np.eye(self.numclass)[label_chunk]]
                features_all[session] += [features_file[i]]
        for session in target_all.keys():
            anno = np.array(target_all[session]).reshape((-1, self.numclass))
            feature_length = anno.shape[0]
            target_all[session] = {'anno': anno, 'feature_length': feature_length}
            features_all[session] = np.array(features_all[session]).reshape((-1, 2048))
            assert feature_length == features_all[session].shape[0], 'Error: target and features are not aligned !!!'
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
        return _inputs, enc_target, distance_target, class_h_target, dec_target

    def __len__(self):
        return len(self.inputs)
