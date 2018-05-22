import os.path
import random
import numpy as np
import json
import torchvision.transforms as transforms
import torch
from PIL import Image
import cv2
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.compute_metrics import compute_metrics

class MultiDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.center_crop = opt.center_crop
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        if self.opt.metrics_condition:
            self.facade_color = [c.color for c in self.opt.lbl_classes if c.name == 'facade']

        assert opt.resize_or_crop == 'resize_and_crop'

    def __getitem__(self, index):

        # read images/data
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize(
            (self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = transforms.ToTensor()(AB)
        if self.opt.window_condition:
            window_path = os.path.join(self.opt.window_dataroot, os.path.relpath(self.AB_paths[index], self.root))
            window_labels = Image.open(window_path).convert('RGB')
            # C = Image.open(self.opt.window_filepath).convert('RGB')
            window_labels = window_labels.resize(
                (self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            window_labels = transforms.ToTensor()(window_labels)
        if self.opt.metrics_condition:
            # load empty facade mask
            metrics_path = os.path.join(self.opt.empty_dataroot, os.path.relpath(self.AB_paths[index], self.root))
            metrics_mask = cv2.imread(metrics_path, cv2.IMREAD_COLOR)
            # metrics_mask = cv2.imread(self.opt.empty_filepath, cv2.IMREAD_COLOR)
            metrics_mask = metrics_mask[:, :, [2, 1, 0]] # convert from BGR to RGB
            metrics_mask = (metrics_mask == self.facade_color).all(axis=2).astype(np.uint8)
            metrics_mask = cv2.resize(metrics_mask, (self.opt.loadSize, self.opt.loadSize), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

            # load unit size
            # info_path = os.path.join(self.opt.metrics_condition[1], os.path.relpath(self.AB_paths[index], self.root))
            # with open(info_path, 'r') as f:
            # # with open(self.opt.info_filepath, 'r') as f:
            #     unit_size = json.load(f)['unit_size'] # in fraction of the image height
            #     unit_size *= metrics_mask.shape[0] # in pixels

            unit_size = float(os.path.basename(os.path.splitext(self.AB_paths[index])[0]).split('@')[1].split('_')[0]) # in fraction of the image height
            unit_size *= metrics_mask.shape[0] # in pixels

            # compute metrics
            metrics = compute_metrics(mask=metrics_mask, scale=1/unit_size)
            metrics[5, :, :] *= self.opt.loadSize # convert floor height from fraction of image height to pixels

            metrics = torch.from_numpy(metrics)
        if self.opt.empty_facade_condition:
            empty_path = os.path.relpath(self.AB_paths[index], self.root)
            empty_path = os.path.join(self.opt.empty_dataroot, empty_path)
            empty_facade = Image.open(empty_path).convert('RGB')
            # empty_facade = Image.open(self.opt.empty_filepath).convert('RGB')
            empty_facade = empty_facade.resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST)
            empty_facade = transforms.ToTensor()(empty_facade)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        if self.center_crop:
            w_offset = int(round((w - self.opt.fineSize) / 2.0))
            h_offset = int(round((h - self.opt.fineSize) / 2.0))
        else:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.window_condition:
            if window_labels.size(2) != w or window_labels.size(1) != h:
                raise ValueError('the additional input does not have the right size.')
            window_labels = window_labels[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            window_labels = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(window_labels)
        if self.opt.metrics_condition:
            metrics = metrics[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            # no need to adjust floor height, floor height in pixels remains the same
        if self.opt.empty_facade_condition:
            empty_facade = empty_facade[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            if self.opt.window_condition:
                window_labels = window_labels.index_select(2, idx)
            if self.opt.metrics_condition:
                metrics = metrics.index_select(2, idx)
            if self.opt.empty_facade_condition:
                empty_facade = empty_facade.index_select(2, idx)

        if input_nc == 1:
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        data = {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

        if self.opt.window_condition:
            data['window'] = window_labels
            data['window_paths'] = window_path
        if self.opt.metrics_condition:
            data['metrics'] = metrics
            data['metrics_paths'] = metrics_path
        if self.opt.empty_facade_condition:
            data['empty'] = empty_facade
            data['empty_paths'] = empty_path

        return data

    def __len__(self):
        if self.opt.phase == 'val':
            return len(self.AB_paths)
        else:
            return len(self.AB_paths) // 2 * 2

    def name(self):
        return 'MultiDataset'
