import numpy as np
from pathlib import Path
import shutil
import os
from PIL import Image

from project.src.classes import LASER_CLASSES

import cv2

data_dir = Path('data/laser_v2')
data_dir_created = Path('data/laser_v2_created')
prepared_data_dir = "prepared_data"

dirs = {
    "root": Path(data_dir),
    "root_created": Path(data_dir_created)
}

all_images = sorted(list(dirs["root"].glob('*.png')))
all_masks = sorted(list(dirs["root"].joinpath('masks').glob('*.png')))

assert len(all_images) == len(all_masks), f"Masks an images count " \
                                          f"should be equal: {len(all_images)} and {len(all_masks)}"
assert len(all_images) != 0, "No images found! Check your paths"

print('Total images {}'.format(len(all_images)))

classes_count = {}
without_defects = []

for cl in LASER_CLASSES:
    Path('data/dataset_analyze/' + cl).mkdir(parents=True, exist_ok=True)
    Path('data/dataset_analyze/' + cl + '/masks').mkdir(parents=True, exist_ok=True)
    Path('data/dataset_analyze/' + cl + '/masks_rgb').mkdir(parents=True, exist_ok=True)

for index, value in enumerate(LASER_CLASSES):
    classes_count[index] = 0

palette = [[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]

for mask_path in all_masks:
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt_class = list(np.unique(gt_mask))

    if len(gt_class) == 1 and gt_class[0] == 0:
        without_defects.append(str(mask_path))

    gt_mask = np.stack((gt_mask,) * 3, axis=-1)

    for cl in gt_class:
        shutil.copy(os.path.join(data_dir, mask_path.name), os.path.join('data/dataset_analyze', LASER_CLASSES[cl]))
        shutil.copy(os.path.join(data_dir, 'masks', mask_path.name),
                    os.path.join('data/dataset_analyze', LASER_CLASSES[cl], 'masks'))


        seg_img = Image.fromarray(gt_mask[:, :, 0]).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        out_mask_vis_path = "{}masks_rgb/{}.png".format('data/dataset_analyze/' + LASER_CLASSES[cl] + '/', mask_path.stem)
        seg_img.save(out_mask_vis_path)

        classes_count[cl] += 1

print()
print('Images per labels:')
for key, cl_count in classes_count.items():
    print('Label ' + str(key) + ' (' + LASER_CLASSES[key] + '):   ' + str(cl_count))

print()
print('Images without defects', len(without_defects))
