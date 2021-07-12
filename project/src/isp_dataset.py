import cv2
import numpy as np
from torch.utils import data

from project.src.classes import LASER_CLASSES


class ISPDataset(data.Dataset):
    def __init__(self, data, input_resize, augments=None, preprocessing=None,
                 return_paths=False, use_binary_mask=True):
        super().__init__()
        self.imgs, self.masks = data
        self.augments = augments
        self.preprocessing = preprocessing
        self.input_resize = (input_resize, input_resize)
        self.return_paths = return_paths
        self.binary_mask = use_binary_mask

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # idx = idx % self.multiplicator
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, self.input_resize,
                         interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.input_resize,
                          interpolation=cv2.INTER_NEAREST)

        if self.binary_mask:
            masks = [(mask == LASER_CLASSES.index(v)) for v in LASER_CLASSES]
            mask = np.stack(masks, axis=-1).astype('float')

        if self.augments:
            augmented = self.augments(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img, mask = preprocessed['image'], preprocessed['mask']

        mask = mask.long()
        if self.return_paths:
            return img, mask, img_path, mask_path
        else:
            return img, mask
