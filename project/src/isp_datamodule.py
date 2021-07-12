import os
import random
import numpy as np
import shutil
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils import data

from project.src.isp_dataset import ISPDataset
from project.src.classes import LASER_CLASSES


class ISPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, use_normalize=True,
                 num_workers=-1, val_fraction=0.2, test_fraction=0.5,
                 input_resize=512, random_seed=42, augment_p=0.7,
                 return_paths=False, half_normalize=False, use_binary_mask=False, recreate_data=False):
        super().__init__()
        prepared_data_dir = "prepared_data"
        data_dir_created = Path('data/laser_v2_created')
        self.root_dir = Path(data_dir)
        self.recreate_data = recreate_data

        self.dirs = {
            "root": Path(data_dir),
            "root_created": Path(data_dir_created),
            "prepared": Path('data', 'datasets', prepared_data_dir),
            "prepared_created": Path('data', 'datasets_created', prepared_data_dir)
        }

        self.data_names = {
            "train": 'train.txt',
            "val": 'val.txt',
            "test": 'test.txt',
        }

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers > 0 else os.cpu_count()
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.input_resize = input_resize
        self.random_seed = random_seed
        self.augment_p = augment_p
        self.use_normalize = use_normalize
        self.return_paths = return_paths
        self.half_normalize = half_normalize
        self.use_binary_mask = use_binary_mask

        if self.random_seed:
            random.seed(random_seed)

        transforms_composed = self._get_transforms()
        self.augments, self.preprocessing = transforms_composed

        self.dataset_train, self.dataset_val = None, None
        self.dataset_test = None

    @staticmethod
    def _get_train_transforms(p):
        # return A.Compose([A.HorizontalFlip()])
        return A.Compose([
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.3),
            A.OneOf([A.IAAAdditiveGaussianNoise(),
                     A.GaussNoise()], p=0.4),
            A.OneOf([A.MotionBlur(p=0.1),
                     A.MedianBlur(blur_limit=3, p=0.1),
                     A.Blur(blur_limit=3, p=0.1)], p=0.2),
            A.OneOf([A.CLAHE(clip_limit=2),
                     A.IAASharpen(),
                     A.IAAEmboss(),
                     A.RandomBrightnessContrast()], p=0.5),
        ], p=p)

    def _get_transforms(self):
        transforms = []

        if self.use_normalize:
            if self.half_normalize:
                transforms += [A.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5])]
            else:
                transforms += [A.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])]
        transforms += [ToTensorV2(transpose_mask=True)]
        preprocessing = A.Compose(transforms)

        return self._get_train_transforms(self.augment_p), preprocessing

    def _write_to_prep_file(self, filename, imgs, masks):
        filepath = self.dirs['prepared'].joinpath(filename)
        imgs, masks = sorted(imgs), sorted(masks)
        with open(filepath, 'w') as file:
            for img, mask in zip(imgs, masks):
                file.write(f"{img} {mask}\n")

    def prepare_data(self):
        for k, dir_ in self.dirs.items():
            if k == 'prepared' and dir_.exists():
                if not self.recreate_data:
                    return
            os.makedirs(dir_, exist_ok=True)

        all_images = sorted(list(self.dirs["root"].glob('*.png')))
        all_masks = sorted(list(self.dirs["root"].joinpath('masks').glob('*.png')))

        assert len(all_images) == len(all_masks), f"Masks an images count " \
                                                  f"should be equal: {len(all_images)} and {len(all_masks)}"
        assert len(all_images) != 0, "No images found! Check your paths"

        # path = self.dirs["root_created"]
        # path_mask = self.dirs["root_created"].joinpath('masks')
        #
        # if os.path.isdir(path):
        #     shutil.rmtree(path)
        #
        # Path(path).mkdir(parents=True, exist_ok=True)
        # Path(path_mask).mkdir(parents=True, exist_ok=True)
        #
        # for img_path, mask_path in zip(all_images, all_masks):
        #     img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        # #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        #
        #     for m in np.unique(mask):
        #         if m not in [0, 255]:
        #             mask_new = np.where(mask != m, 0, mask)
        #             cv2.imwrite(str(path) + '/' + img_path.stem + '_' + LASER_CLASSES[m] + '.png', img)
        #             cv2.imwrite(str(path_mask) + '/' + img_path.stem + '_' + LASER_CLASSES[m] + '.png', mask_new)
        #
        # all_images = sorted(list(self.dirs["root_created"].glob('*.png')))
        # all_masks = sorted(list(self.dirs["root_created"].joinpath('masks').glob('*.png')))
        #
        # assert len(all_images) == len(all_masks), f"Masks an images count " \
        #                                           f"should be equal: {len(all_images)} and {len(all_masks)}"
        # assert len(all_images) != 0, "No images found! Check your paths"

        # split train and val
        train_num, val_num = self.get_sizes_from_fraction(all_images, self.val_fraction)
        imgs_train, imgs_val = data.random_split(all_images, [train_num, val_num],
                                                 generator=torch.Generator().manual_seed(self.random_seed))
        # split val and test
        val_num, test_num = self.get_sizes_from_fraction(imgs_val, self.test_fraction)
        imgs_val, imgs_test = data.random_split(imgs_val, [val_num, test_num],
                                                generator=torch.Generator().manual_seed(self.random_seed))

        # find masks for images
        masks_train = self._fill_masks_paths(all_masks, imgs_train)
        masks_val = self._fill_masks_paths(all_masks, imgs_val)
        masks_test = self._fill_masks_paths(all_masks, imgs_test)

        # save train, val and test in files
        self._write_to_prep_file(self.data_names["train"], imgs_train, masks_train)
        self._write_to_prep_file(self.data_names["val"], imgs_val, masks_val)
        self._write_to_prep_file(self.data_names["test"], imgs_test, masks_test)

    @staticmethod
    def _fill_masks_paths(masks, imgs):
        equal_masks = []
        for mask in masks:
            for img in imgs:
                if mask.name == img.name:
                    equal_masks.append(mask)
        assert len(equal_masks) == len(imgs), "Masks and images should be equal"
        return equal_masks

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_filepath = self.dirs['prepared'].joinpath(self.data_names['train'])
            val_filepath = self.dirs['prepared'].joinpath(self.data_names['val'])

            train_data = self._read_imgs_masks(train_filepath)
            val_data = self._read_imgs_masks(val_filepath)

            self.dataset_train = ISPDataset(
                train_data,
                self.input_resize,
                augments=self.augments,
                preprocessing=self.preprocessing,
                return_paths=self.return_paths,
                use_binary_mask=self.use_binary_mask)

            self.dataset_val = ISPDataset(
                val_data,
                self.input_resize,
                preprocessing=self.preprocessing,
                return_paths=self.return_paths,
                use_binary_mask=self.use_binary_mask)

            self.dims = tuple(self.dataset_train[0][0].shape)

        if stage == 'test' or stage is None:
            test_filepath = self.dirs['prepared'].joinpath(self.data_names['test'])
            test_data = self._read_imgs_masks(test_filepath)
            self.dataset_test = ISPDataset(
                test_data,
                self.input_resize,
                preprocessing=self.preprocessing,
                return_paths=self.return_paths,
                use_binary_mask=self.use_binary_mask)

            self.dims = tuple(self.dataset_test[0][0].shape)

    @staticmethod
    def _read_imgs_masks(filepath):
        img_paths, mask_paths = [], []
        with open(filepath, 'r') as file:
            for line in file.readlines():
                img_path, mask_path = line.strip('\n').split(' ')
                img_paths.append(img_path)
                mask_paths.append(mask_path)
        return img_paths, mask_paths

    def train_dataloader(self):
        return data.DataLoader(self.dataset_train, batch_size=self.batch_size,
                               shuffle=True, num_workers=self.num_workers,
                               pin_memory=True)

    def val_dataloader(self):
        return data.DataLoader(self.dataset_val, batch_size=1,
                               shuffle=False, num_workers=self.num_workers,
                               pin_memory=True)

    def test_dataloader(self):
        return data.DataLoader(self.dataset_test, batch_size=1,
                               shuffle=False, num_workers=self.num_workers,
                               pin_memory=True)

    @staticmethod
    def get_sizes_from_fraction(dataset: List,
                                fraction: float) -> Tuple[int, int]:
        """
        Получаем из датасета размеры большой и маленькой выборки
        :param dataset: любой объект, имеющий метод len()
        :param fraction: желательно передавать в пределах 0 < x <= 0.5
        :return: два значения: размер большой выборки, размер маленькой выборки
        """
        full_len = len(dataset)
        small_size = int(full_len * fraction)
        big_size = full_len - small_size
        return big_size, small_size
