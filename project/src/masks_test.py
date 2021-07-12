import cv2
import cv2.cv2
import numpy as np
import os, shutil
import torch
from pathlib import Path
from typing import List, Tuple

from torch.utils import data
from project.src.classes import LASER_CLASSES


def prepare_data(dirs, val_fraction, test_fraction, random_seed):
    all_images = sorted(list(dirs["root"].glob('*.png')))
    all_masks = sorted(list(dirs["root"].joinpath('masks').glob('*.png')))

    assert len(all_images) == len(all_masks), f"Masks an images count " \
                                              f"should be equal: {len(all_images)} and {len(all_masks)}"
    assert len(all_images) != 0, "No images found! Check your paths"

    path = dirs["root_created"]
    path_mask = dirs["root_created"].joinpath('masks')

    if os.path.isdir(path):
        shutil.rmtree(path)

    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path_mask).mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in zip(all_images, all_masks):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        for m in np.unique(mask):
            if m not in [0, 255]:
                mask_new = np.where(mask != m, 0, mask)
                cv2.imwrite(str(path) + '/' + img_path.stem + '_' + LASER_CLASSES[m] + '.png', img)
                cv2.imwrite(str(path_mask) + '/' + img_path.stem + '_' + LASER_CLASSES[m] + '.png', mask_new)

                # mask_show = np.where(mask_new == mask_new.max(), 255, mask_new)
                #
                # cv2.imshow('img', img)
                # cv2.imshow('mask', mask_show)
                #
                # key = cv2.waitKey()

    all_images = sorted(list(dirs["root_created"].glob('*.png')))
    all_masks = sorted(list(dirs["root_created"].joinpath('masks').glob('*.png')))

    assert len(all_images) == len(all_masks), f"Masks an images count " \
                                              f"should be equal: {len(all_images)} and {len(all_masks)}"
    assert len(all_images) != 0, "No images found! Check your paths"

    # split train and val
    train_num, val_num = get_sizes_from_fraction(all_images, val_fraction)
    imgs_train, imgs_val = data.random_split(all_images, [train_num, val_num],
                                             generator=torch.Generator().manual_seed(random_seed))
    # split val and test
    val_num, test_num = get_sizes_from_fraction(imgs_val, test_fraction)
    imgs_val, imgs_test = data.random_split(imgs_val, [val_num, test_num],
                                            generator=torch.Generator().manual_seed(random_seed))

    # find masks for images
    masks_train = _fill_masks_paths(all_masks, imgs_train)
    masks_val = _fill_masks_paths(all_masks, imgs_val)
    masks_test = _fill_masks_paths(all_masks, imgs_test)

    # save train, val and test in files
    _write_to_prep_file(data_names["train"], imgs_train, masks_train)
    _write_to_prep_file(data_names["val"], imgs_val, masks_val)
    _write_to_prep_file(data_names["test"], imgs_test, masks_test)

    counter = 0

    imgs_train, masks_train = sorted(imgs_train), sorted(masks_train)

    for img_path, mask_path in zip(imgs_train, masks_train):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        print('Mask number ' + str(counter))
        print('mask.max', mask.max())
        print('np.unique(a)', np.unique(mask))

        # masks = [(mask == LASER_CLASSES.index(v)) for v in LASER_CLASSES]
        # mask = np.stack(masks, axis=-1).astype('float')

        mask = np.where(mask == mask.max(), 255, mask)

        # for m in np.unique(mask):
        #     if m == 1:
        #         mask = np.where(mask == 1, 255, mask)
        #     if m == 2:
        #         mask = np.where(mask == 2, 200, mask)
        #     if m == 3:
        #         mask = np.where(mask == 3, 150, mask)
        #     if m == 4:
        #         mask = np.where(mask == 4, 100, mask)

        cv2.imshow('img', img)
        cv2.imshow('mask', mask)

        key = cv2.waitKey()

        counter += 1

    img_path = 'data/laser_v2/20201221_146ef6fa-63bd-41b2-a14a-8657afdf2b0e_04023_1_001_laser.png'
    mask_path = 'data/laser_v2/masks/20201221_146ef6fa-63bd-41b2-a14a-8657afdf2b0e_04023_1_001_laser.png'

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask.max() == 4:
        mask = np.where(mask == 4, 255, mask)

    print('Mask number ' + str(counter))
    print('mask.max', mask.max())
    print('np.unique(a)', np.unique(mask))

    cv2.imshow('img', img)
    cv2.imshow('mask', mask)

    key = cv2.waitKey()

    counter += 1

    cv2.destroyAllWindows()


def _fill_masks_paths(masks, imgs):
    equal_masks = []
    for mask in masks:
        for img in imgs:
            if mask.name == img.name:
                equal_masks.append(mask)
    assert len(equal_masks) == len(imgs), "Masks and images should be equal"
    return equal_masks


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


def _write_to_prep_file(filename, imgs, masks):
    filepath = dirs['prepared_created'].joinpath(filename)
    imgs, masks = sorted(imgs), sorted(masks)
    with open(filepath, 'w') as file:
        for img, mask in zip(imgs, masks):
            file.write(f"{img} {mask}\n")


data_dir = Path('data/laser_v2')
data_dir_created = Path('data/laser_v2_created')
prepared_data_dir = "prepared_data"

val_fraction = 0.2
test_fraction = 0.5

random_seed = 42

root_dir = Path(data_dir)

dirs = {
    "root": Path(data_dir),
    "prepared": Path('data', 'datasets', prepared_data_dir),
    "root_created": Path(data_dir_created),
    "prepared_created": Path('data', 'datasets_created', prepared_data_dir),
}

data_names = {
    "train": 'train.txt',
    "val": 'val.txt',
    "test": 'test.txt',
}

prepare_data(dirs, val_fraction, test_fraction, random_seed)
