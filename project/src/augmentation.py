import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image

from project.src.classes import LASER_CLASSES

import albumentations as A
import shutil
import cv2


def create_augmented_imgs_for_class(data_dir, label):
    # clear folder
    dirpath = Path(data_dir).joinpath('aug')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(data_dir).joinpath('aug').mkdir(parents=True, exist_ok=True)

    dirpath = Path(data_dir).joinpath('aug').joinpath('masks')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(data_dir).joinpath('aug').joinpath('masks').mkdir(parents=True, exist_ok=True)

    dirpath = Path(data_dir).joinpath('aug').joinpath('masks_rgb')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(data_dir).joinpath('aug').joinpath('masks_rgb').mkdir(parents=True, exist_ok=True)

    all_images = sorted(list(Path(data_dir).glob('*.png')))
    all_masks = sorted(list(Path(data_dir).joinpath('masks').glob('*.png')))

    assert len(all_images) == len(all_masks), f"Masks an images count " \
                                              f"should be equal: {len(all_images)} and {len(all_masks)}"
    assert len(all_images) != 0, "No images found! Check your paths"

    print('\nClass {}. Total images {}'.format(label, len(all_images)))
    print('Augmentation...')

    p = 1

    transform = A.Compose([
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
    ], p=1)

    palette = [[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]

    for i, (img_path, mask_path) in tqdm(enumerate(zip(all_images, all_masks)), total=len(all_images)):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        transformed = transform(image=img, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        cv2.imwrite(str(Path(data_dir).joinpath('aug').joinpath('aug_' + os.path.basename(img_path))),
                    transformed_image)

        cv2.imwrite(
            str(Path(data_dir).joinpath('aug').joinpath('masks').joinpath('aug_' + os.path.basename(img_path))),
            transformed_mask)

        transformed_mask = np.stack((transformed_mask,) * 3, axis=-1)
        seg_img = Image.fromarray(transformed_mask[:, :, 0]).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        # out_mask_vis_path = "{}masks_rgb/{}.png".format('data/dataset_analyze/' + LASER_CLASSES[cl] + '/aug/',
        #                                                 mask_path.stem)
        seg_img.save(
            str(Path(data_dir).joinpath('aug').joinpath('masks_rgb').joinpath('aug_' + os.path.basename(img_path))))


def create_augmented_imgs_for_dataset(data_dir):
    # clear folder
    dirpath = Path(data_dir).joinpath('aug')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(data_dir).joinpath('aug').mkdir(parents=True, exist_ok=True)

    dirpath = Path(data_dir).joinpath('aug').joinpath('masks')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(data_dir).joinpath('aug').joinpath('masks').mkdir(parents=True, exist_ok=True)

    dirpath = Path(data_dir).joinpath('aug').joinpath('masks_rgb')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(data_dir).joinpath('aug').joinpath('masks_rgb').mkdir(parents=True, exist_ok=True)

    # create mask_rgb for dataset too
    dirpath = Path(data_dir).joinpath('masks_rgb')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(data_dir).joinpath('masks_rgb').mkdir(parents=True, exist_ok=True)

    all_images = sorted(list(Path(data_dir).glob('*.png')))
    all_masks = sorted(list(Path(data_dir).joinpath('masks').glob('*.png')))

    assert len(all_images) == len(all_masks), f"Masks an images count " \
                                              f"should be equal: {len(all_images)} and {len(all_masks)}"
    assert len(all_images) != 0, "No images found! Check your paths"

    print('\nTotal images {}'.format(len(all_images)))
    print('Augmentation...')

    p = 1

    transform = A.Compose(
        [A.OneOf([
            # A.Rotate(limit=35, p=1),
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Transpose(p=1)], p=1)], p=1)

    # transform = A.Compose([
    #     A.Rotate(limit=35, p=0.5),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.Transpose(p=0.5),
    # ], p=1)

    palette = [[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]

    for i, (img_path, mask_path) in tqdm(enumerate(zip(all_images, all_masks)), total=len(all_images)):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        transformed = transform(image=img, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        cv2.imwrite(str(Path(data_dir).joinpath('aug').joinpath('aug_' + os.path.basename(img_path))),
                    transformed_image)

        cv2.imwrite(
            str(Path(data_dir).joinpath('aug').joinpath('masks').joinpath('aug_' + os.path.basename(img_path))),
            transformed_mask)

        transformed_mask = np.stack((transformed_mask,) * 3, axis=-1)
        seg_img = Image.fromarray(transformed_mask[:, :, 0]).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(
            str(Path(data_dir).joinpath('aug').joinpath('masks_rgb').joinpath('aug_' + os.path.basename(img_path))))

        # create mask_rgb for dataset too
        mask = np.stack((mask,) * 3, axis=-1)
        seg_img = Image.fromarray(mask[:, :, 0]).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(
            str(Path(data_dir).joinpath('masks_rgb').joinpath(os.path.basename(img_path))))


if __name__ == '__main__':
    # data_dir = 'data/dataset_analyze/dataset_per_classes/'
    # for cl in LASER_CLASSES:
    #     class_data_dir = data_dir + cl
    #     create_augmented_imgs_for_class(data_dir=class_data_dir, label=cl)

    data_dir_dataset = 'data/dataset_analyze/whole_dataset/'
    create_augmented_imgs_for_dataset(data_dir=data_dir_dataset)
