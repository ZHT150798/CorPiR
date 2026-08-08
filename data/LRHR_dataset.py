from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import random
import data.util as Util
from torchvision.transforms import functional as trans_fn
import torch
import cv2
import numpy as np
import core.imgproc as imgproc
from core.utils import make_coord


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1,
                 need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            # self.sr_path = Util.get_paths_from_images(
            #    '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(dataroot)  # '{}/hr_{}'.format(dataroot, r_resolution))
            # if self.need_LR:
            #    self.lr_path = Util.get_paths_from_images('{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len - 1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            if self.split == 'train':
                img_HR = transforms.RandomCrop(self.r_res)(img_HR)
            elif self.split == 'val':
                img_HR = transforms.CenterCrop(self.r_res)(img_HR)
            # img_LR = trans_fn.resize(img_HR, self.l_res, InterpolationMode.BICUBIC)
            # img_SR = trans_fn.resize(img_LR, self.r_res, InterpolationMode.BICUBIC)
            # [img_LR, img_SR, img_HR] = Util.transform_augment_v1([img_LR, img_SR, img_HR], split=self.split,
            #                                                 min_max=(-1, 1))
            [img_HR] = Util.transform_augment([img_HR], split=self.split, min_max=(-1, 1))
            if self.need_LR:
                # img_LR = Image.open(self.lr_path[index]).convert("RGB")
                return {'HR': img_HR, 'Index': index}
                # return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
            return {'HR': img_HR, 'Index': index}
            # return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        '''if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}'''


class MyDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1,
                 need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.upscale_factor = self.r_res//self.l_res # random.uniform(1.01, scale)


        if datatype == 'img':
            # self.sr_path = Util.get_paths_from_images(
            #    '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images('{}/hr_{}'.format(dataroot, r_resolution))
            # if self.need_LR:
            #    self.lr_path = Util.get_paths_from_images('{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            print("datatype wrong")
        else:
            img_HR = cv2.imread(self.hr_path[index]).astype(np.float32) / 255

            if self.split == 'train':
                gt_crop_image = imgproc.random_crop(img_HR, self.r_res)
                gt_crop_image = imgproc.random_rotate(gt_crop_image, [90, 180, 270])
                gt_crop_image = imgproc.random_horizontally_flip(gt_crop_image, 0.5)
                gt_crop_image = imgproc.random_vertically_flip(gt_crop_image, 0.5)
            elif self.split == 'val':
                gt_crop_image = imgproc.center_crop(img_HR, self.r_res)
            else:
                raise ValueError("Unsupported data processing model, please use `train` or `val`.")
            lr_crop_image = imgproc.image_resize(gt_crop_image, 1 / self.upscale_factor)
            sr_crop_image = imgproc.image_resize(lr_crop_image, self.upscale_factor)
            # BGR convert RGB
            gt_crop_image = cv2.cvtColor(gt_crop_image, cv2.COLOR_BGR2RGB)
            lr_crop_image = cv2.cvtColor(lr_crop_image, cv2.COLOR_BGR2RGB)
            sr_crop_image = cv2.cvtColor(sr_crop_image, cv2.COLOR_BGR2RGB)

            # Convert image data into Tensor stream format (PyTorch).
            # Note: The range of input and output is between [0, 1]
            gt_crop_tensor = imgproc.image_to_tensor(gt_crop_image, False, False)
            lr_crop_tensor = imgproc.image_to_tensor(lr_crop_image, False, False)
            sr_crop_tensor = imgproc.image_to_tensor(sr_crop_image, False, False)
            gt_tensor_coord = make_coord(gt_crop_tensor.contiguous().shape[-2:])
            gt_tensor_contiguous = gt_crop_tensor.contiguous().view(3, -1).permute(1, 0)
            gt_tensor_cell = torch.ones_like(gt_tensor_coord)
            gt_tensor_cell[:, 0] *= 2 / gt_crop_tensor.shape[-2]
            gt_tensor_cell[:, 1] *= 2 / gt_crop_tensor.shape[-1]

            return {"hr": gt_tensor_contiguous, "cell": gt_tensor_cell, "coord": gt_tensor_coord, "lr": lr_crop_tensor, "sr": sr_crop_tensor, 'Index': index}


class MyDataset_infer(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1,
                 need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.upscale_factor = self.r_res//self.l_res  # random.uniform(1.01, scale)


        if datatype == 'img':
            # self.sr_path = Util.get_paths_from_images(
            #    '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images('{}/hr_{}'.format(dataroot, r_resolution))
            # if self.need_LR:
            #    self.lr_path = Util.get_paths_from_images('{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            print("datatype wrong")
        else:
            img_HR = cv2.imread(self.hr_path[index]).astype(np.float32) / 255
            lr_crop_image = imgproc.image_resize(img_HR, 1 / self.upscale_factor)
            sr_crop_image = imgproc.image_resize(lr_crop_image, self.upscale_factor)
            img_name = self.hr_path[index].split("/")[-1]


            # BGR convert RGB
            lr_crop_image = cv2.cvtColor(lr_crop_image, cv2.COLOR_BGR2RGB)
            sr_crop_image = cv2.cvtColor(sr_crop_image, cv2.COLOR_BGR2RGB)

            # Convert image data into Tensor stream format (PyTorch).
            # Note: The range of input and output is between [0, 1]
            lr_crop_tensor = imgproc.image_to_tensor(lr_crop_image, False, False)
            sr_crop_tensor = imgproc.image_to_tensor(sr_crop_image, False, False)
            gt_tensor_coord = make_coord(sr_crop_tensor.contiguous().shape[-2:])
            gt_tensor_contiguous = sr_crop_tensor.contiguous().view(3, -1).permute(1, 0)
            gt_tensor_cell = torch.ones_like(gt_tensor_coord)
            gt_tensor_cell[:, 0] *= 2 / sr_crop_tensor.shape[-2]
            gt_tensor_cell[:, 1] *= 2 / sr_crop_tensor.shape[-1]

            return {"hr": gt_tensor_contiguous, "cell": gt_tensor_cell, "coord": gt_tensor_coord, "lr": lr_crop_tensor,
                    "sr": sr_crop_tensor, 'name': img_name}