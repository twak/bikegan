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

        assert opt.resize_or_crop == 'resize_and_crop'

    def __getitem__(self, index):

        # read images/data
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize(
            (self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = transforms.ToTensor()(AB)

        if self.opt.mlabel_condition:
            mlabel_path = os.path.join(self.opt.mlabel_dataroot, os.path.relpath(self.AB_paths[index], self.root))
            mlabels = Image.open(mlabel_path).convert('RGB')
            mlabels = mlabels.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            mlabels = transforms.ToTensor()(mlabels)

        if self.opt.metrics_condition:
            # load empty mask
            metrics_path = ''
            empty_path = os.path.join(self.opt.empty_dataroot, os.path.relpath(self.AB_paths[index], self.root))
            metrics_mask = cv2.imread(empty_path, cv2.IMREAD_COLOR)
            metrics_mask = metrics_mask[:, :, [2, 1, 0]] # convert from BGR to RGB
            metrics_mask = (metrics_mask == self.opt.metrics_mask_color).all(axis=2).astype(np.uint8)
            metrics_mask = cv2.resize(metrics_mask, (self.opt.loadSize, self.opt.loadSize), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

            if '@' in os.path.basename(os.path.splitext(self.AB_paths[index])[0]):
                unit_size = float(os.path.basename(os.path.splitext(self.AB_paths[index])[0]).split('@')[1].split('_')[0]) # in fraction of the image height
                unit_size *= metrics_mask.shape[0] # in pixels
            else:
                unit_size = metrics_mask.shape[0]

            # compute metrics
            metrics = compute_metrics(mask=metrics_mask, scale=1/unit_size)

            if not (self.opt.normalize_metrics or self.opt.normalize_metrics2):
                metrics[5, :, :] *= self.opt.loadSize # convert floor height from fraction of image height to pixels
            if self.opt.normalize_metrics2:
                metrics[:5, :, :] *= 0.1

            metrics = torch.from_numpy(metrics)
        if self.opt.empty_condition:
            empty_path = os.path.relpath(self.AB_paths[index], self.root)
            empty_path = os.path.join(self.opt.empty_dataroot, empty_path)
            empty_mask = Image.open(empty_path).convert('RGB')
            empty_mask = empty_mask.resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST)
            empty_mask = transforms.ToTensor()(empty_mask)

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

        if self.opt.mlabel_condition:
            if mlabels.size(2) != w or mlabels.size(1) != h:
                raise ValueError('the additional input does not have the right size.')
            mlabels = mlabels[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            mlabels = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(mlabels)
        if self.opt.metrics_condition:
            metrics = metrics[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            # no need to adjust floor height, floor height in pixels remains the same
        if self.opt.empty_condition:
            empty_mask = empty_mask[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

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
            if self.opt.mlabel_condition:
                mlabels = mlabels.index_select(2, idx)
            if self.opt.metrics_condition:
                metrics = metrics.index_select(2, idx)
            if self.opt.empty_condition:
                empty_mask = empty_mask.index_select(2, idx)

        if input_nc == 1:
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        data = {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

        if self.opt.mlabel_condition:
            data['mlabel'] = mlabels
            data['mlabel_paths'] = mlabel_path
        if self.opt.metrics_condition:
            data['metrics'] = metrics
            data['metrics_paths'] = metrics_path
        if self.opt.empty_condition:
            data['empty'] = empty_mask
            data['empty_paths'] = empty_path

        return data

    def __len__(self):
        if self.opt.phase == 'val':
            return len(self.AB_paths)
        else:
            return len(self.AB_paths) // 2 * 2

    def name(self):
        return 'MultiDataset'
