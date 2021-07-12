from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import os
import datetime

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

import math

from project.src.isp_model import ISPModel

import torchmetrics
from project.src.classes import LASER_CLASSES

from sklearn.metrics import precision_recall_fscore_support as score
from pytorch_lightning.metrics import F1, IoU

import warnings

warnings.filterwarnings('ignore')

# VISUALIZE
DATA_DIR = Path('data_test/laser_v2')
TEST_FILE = 'test.txt'

USE_BINARY_MASK = False
RETURN_PATHS = True
NUM_WORKERS = 8
AUGMENT_P = 0.7

FPN_resnet34 = {
    'model_name': 'FPN_resnet34',
    'checkpoint': 'tb_logs/FPN_resnet34/version_19/checkpoints/FPN_resnet34_epoch=46_val_loss=0.831_val_f1_epoch=0.831_val_iou_epoch=0.504.ckpt',
    'use_normalize': True, 'half_normalize': False,
    'im_size': 512,
    'batch_size': 2}

FPN_se_resnext50_32x4d = {
    'model_name': 'FPN_se_resnext50_32x4d',
    'checkpoint': 'tb_logs/se_resnext50_32x4d/version_0/checkpoints/se_resnext50_32x4d_epoch=18_val_loss=0.465_val_f1_epoch=0.818_val_iou_epoch=0.506.ckpt',
    'use_normalize': True, 'half_normalize': False,
    'im_size': 512,
    'batch_size': 2}

FPN_efficientnet_b0 = {
    'model_name': 'FPN_efficientnet_b0',
    'checkpoint': 'tb_logs/FPN_efficientnet_b0/version_1/checkpoints/FPN_efficientnet_b0_epoch=14_val_loss=0.457_val_f1_epoch=0.818_val_iou_epoch=0.495.ckpt',
    'use_normalize': True, 'half_normalize': False,
    'im_size': 512,
    'batch_size': 2}

models_list = [FPN_resnet34]

experiment_name = ''
models = []
data_loaders = []

for m in models_list:
    experiment_name += m['model_name'] + '__'

# if os.path.exists(DATA_DIR.joinpath('visualizations').joinpath(experiment_name)):
#     os.remove(DATA_DIR.joinpath('visualizations').joinpath(experiment_name))

OUTPUT_TEST_PATH = DATA_DIR.joinpath('visualizations').joinpath(experiment_name)
OUTPUT_TEST_PATH.mkdir(parents=True, exist_ok=True)

colours = {
    0: (255, 255, 255),
    1: (0, 0, 255),  # blue
    2: (0, 255, 0),  # green
    3: (255, 0, 0),  # red
    4: (255, 255, 0),  # yellow
}

# f1 = torchmetrics.F1(num_classes=len(LASER_CLASSES))
# iou = torchmetrics.IoU(num_classes=len(LASER_CLASSES))

f1 = F1(num_classes=len(LASER_CLASSES))
iou = IoU(num_classes=len(LASER_CLASSES))


# Testing
# trainer = pl.Trainer(gpus=1)
# trainer.test(best_model1, datamodule=dm)


def _read_imgs_masks(filepath):
    img_paths, mask_paths = [], []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            img_path, mask_path = line.strip('\n').split(' ')
            img_paths.append(img_path)
            mask_paths.append(mask_path)
    return img_paths, mask_paths


def _get_transforms(use_normalize=True, half_normalize=False):
    transforms = []

    if use_normalize:
        if half_normalize:
            transforms += [A.Normalize(mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5])]
        else:
            transforms += [A.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
    transforms += [ToTensorV2(transpose_mask=True)]
    preprocessing = A.Compose(transforms)

    return preprocessing


def visualize_step(img_path, mask_path, models_list, output_path, colours, images_gt_dict, iou_dict, f1_dict
                   ):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    img_vis = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

    for model in models_list:
        best_model = ISPModel.load_from_checkpoint(model['checkpoint'])
        best_model = best_model.cuda()

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (model['im_size'], model['im_size']),
                         interpolation=cv2.INTER_NEAREST)
        gt_mask = cv2.resize(gt_mask, (model['im_size'], model['im_size']),
                             interpolation=cv2.INTER_NEAREST)

        preprocessing = _get_transforms(model['use_normalize'], model['half_normalize'])
        preprocessed = preprocessing(image=img, mask=gt_mask)
        img, gt_mask = preprocessed['image'], preprocessed['mask']

        # add dimension [3, 512, 512] -> [1, 3, 512, 512]
        img = img.unsqueeze(0)

        gt_mask = gt_mask.long().squeeze().numpy()
        pr_mask = np.zeros((len(LASER_CLASSES), model['im_size'], model['im_size']), dtype='float64')

        pr_mask_model = best_model(img.cuda()).detach().squeeze().cpu().numpy()

        pr_mask += pr_mask_model

    pr_mask = pr_mask.argmax(axis=0)

    gt_class = list(np.unique(gt_mask))
    pr_class = list(np.unique(pr_mask))

    images_gt_dict[str(img_path)] = gt_class

    iou_classes = []

    for index, value in enumerate(LASER_CLASSES):
        if index == 0:
            gt_mask_class = np.where(gt_mask == index, 255, gt_mask)
            pr_mask_class = np.where(pr_mask == index, 255, pr_mask)

            gt_mask_class = np.where(gt_mask_class != 255, 0, gt_mask_class)
            pr_mask_class = np.where(pr_mask_class != 255, 0, pr_mask_class)

            intersection = np.logical_and(pr_mask_class, gt_mask_class)
            union = np.logical_or(pr_mask_class, gt_mask_class)
            iou_score_class = np.sum(intersection) / np.sum(union)

            if math.isnan(iou_score_class):
                iou_classes.insert(index, 0)
            else:
                iou_classes.insert(index, iou_score_class)
        else:
            gt_mask_class = np.where(gt_mask == index, gt_mask, 0)
            pr_mask_class = np.where(pr_mask == index, pr_mask, 0)

            intersection = np.logical_and(pr_mask_class, gt_mask_class)
            union = np.logical_or(pr_mask_class, gt_mask_class)
            iou_score_class = np.sum(intersection) / np.sum(union)

            if math.isnan(iou_score_class):
                iou_classes.insert(index, 0)
            else:
                iou_classes.insert(index, iou_score_class)

    precision, recall, f1_score, support = score(np.array(gt_mask).reshape(-1, ).tolist(),
                                                 np.array(pr_mask).reshape(-1, ).tolist(), average=None,
                                                 labels=[index for index, value in enumerate(LASER_CLASSES)])

    iou_classes_arr = np.array(iou_classes)
    print()
    print('labels:    {}'.format([format(index, '.5f') for index, value in enumerate(LASER_CLASSES)]))
    print('precision: {}'.format(precision))
    print('recall:    {}'.format(recall))
    print('fscore:    {}'.format(f1_score))
    print('IoU:       {}'.format(iou_classes_arr))

    iou_dict[str(img_path)] = iou_classes
    f1_dict[str(img_path)] = f1_score.tolist()

    gt_mask_to_show = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=int)
    pr_mask_to_show = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=int)

    gt_mask_to_show.fill(255)
    pr_mask_to_show.fill(255)

    for c, rgb in colours.items():
        gt_mask_to_show[(gt_mask == c), :] = rgb
        pr_mask_to_show[(pr_mask == c), :] = rgb

    axes[0].imshow(img_vis)
    axes[1].imshow(gt_mask_to_show)
    axes[2].imshow(pr_mask_to_show)

    axes[0].set_title("Original image: {}".format(os.path.basename(img_path)), y=1.08)

    if len(gt_class) > 1:
        axes[1].set_title("Ground truth mask. Classes: {}".format([x for x in gt_class]))
    elif len(gt_class) == 1:
        axes[1].set_title("Ground truth mask. Class: {}".format(gt_class[0]))
    else:
        axes[1].set_title("No ground truth mask")
    if len(pr_class) == 1:
        if int(pr_class[0]) in gt_class:
            axes[2].set_title("Predicted mask. Class: {}".format(pr_class[0]))
        elif int(pr_class[0]) not in gt_class:
            axes[2].set_title("Predicted mask. Class: {}".format(pr_class[0]), backgroundcolor="red")
    elif len(pr_class) > 1:
        if list(np.intersect1d(pr_class, gt_class)) != []:
            axes[2].set_title("Predicted mask. Multiple classes detected. Classes: {}".format([p for p in pr_class]))
        else:
            axes[2].set_title("Predicted mask. Multiple classes detected. Classes: {}".format([p for p in pr_class]),
                              backgroundcolor="red")
    else:
        axes[2].set_title("No mask predicted", backgroundcolor="red")

    fig.tight_layout()
    fig.savefig(output_path.joinpath(os.path.basename(img_path)))
    fig.clear()
    plt.close(fig)

    return images_gt_dict, f1_dict, iou_dict


images_gt_dict = {}
iou_dict = {}
f1_dict = {}

img_paths, mask_paths = _read_imgs_masks(DATA_DIR.joinpath(TEST_FILE))

for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
    #  if i == 21:
    print("Valid image count: {}/{}".format(i + 1, len(img_paths)))
    images_gt_dict, f1_dict, iou_dict = visualize_step(img_path, mask_path, models_list, OUTPUT_TEST_PATH, colours,
                                                       images_gt_dict, iou_dict, f1_dict)

print()

iou_classes = {}
f1_classes = {}

for index, value in enumerate(LASER_CLASSES):
    iou_classes[index] = []
    f1_classes[index] = []

for key, gt_labels_list in images_gt_dict.items():
    for index, label in enumerate(gt_labels_list):
        iou_classes.get(label).append(iou_dict.get(key)[label])
        f1_classes.get(label).append(f1_dict.get(key)[label])

f1_total = []
print('__F1_score per classes__')
for label, values_list in f1_classes.items():
    print('label ' + str(label) + ':   ' + str(np.mean(values_list)))
    if not math.isnan(np.mean(values_list)):
        f1_total.append(np.mean(values_list))

print('F1_score total: ' + str(np.mean(f1_total)))

print()
iou_total = []
print('__IoU per classes__')
for label, values_list in iou_classes.items():
    print('label ' + str(label) + ':   ' + str(np.mean(values_list)))
    if not math.isnan(np.mean(values_list)):
        iou_total.append(np.mean(values_list))

print('IoU total: ' + str(np.mean(iou_total)))

# save test results
filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '__F1_' + str(
    round(np.mean(f1_total), 3)) + '__IoU_ ' + str(round(np.mean(iou_total), 3)) + '.txt'

path = 'data_test/laser_v2/logs/' + experiment_name
Path(path).mkdir(parents=True, exist_ok=True)

with open(path + '/' + filename, 'w+') as f:
    f.write(experiment_name + '\n\n')
    f.write('__F1_score per classes__\n')
    for label, values_list in f1_classes.items():
        f.write('label ' + str(label) + ':   ' + str(np.mean(values_list)) + '\n')
    f.write('F1_score total: ' + str(np.mean(f1_total)))
    f.write('\n\n')
    f.write('__IoU per classes__\n')
    for label, values_list in iou_classes.items():
        f.write('label ' + str(label) + ':   ' + str(np.mean(values_list)) + '\n')
    f.write('IoU total: ' + str(np.mean(iou_total)))
