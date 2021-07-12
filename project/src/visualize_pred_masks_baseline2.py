from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import os
import datetime

import albumentations as A
import cv2
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2

from pytorch_lightning.metrics import F1, IoU

import math

import shutil
from project.src.isp_datamodule import ISPDataModule
from project.src.isp_model import ISPModel

import torchmetrics
from project.src.classes import LASER_CLASSES

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, jaccard_score

from sklearn.metrics import f1_score as f1_sklearn_metrics
from sklearn.metrics import jaccard_score as jaccard_score_sklearn_metrics

from project.cloned_repo.DPT.dpt.models import DPTSegmentationModel

import warnings

warnings.filterwarnings('ignore')

# VISUALIZE
DATA_DIR = Path('data_test/laser_v2')
# DATA_DIR = Path('data/datasets/laser_v2')
TEST_FILE = 'test.txt'

USE_BINARY_MASK = False
RETURN_PATHS = True
NUM_WORKERS = 8
AUGMENT_P = 0.7

BATCH_SIZE = 2
USE_NORMALIZE = True
HALF_NORMALIZE = False
IM_SIZE = 512

experiment_name = 'baseline'

if os.path.exists(DATA_DIR.joinpath('visualizations').joinpath(experiment_name)):
    shutil.rmtree(DATA_DIR.joinpath('visualizations').joinpath(experiment_name))

OUTPUT_TEST_PATH = DATA_DIR.joinpath('visualizations').joinpath(experiment_name)
OUTPUT_TEST_PATH.mkdir(parents=True, exist_ok=True)

CHECKPOINT = 'azamat_weights/DPT_pretr_sgd_ce_noweighted_epoch=21_val_loss=0.351_val_f1_epoch=0.860_val_iou_epoch=0.603.ckpt'

best_model = ISPModel.load_from_checkpoint(checkpoint_path=CHECKPOINT, architecture='DPTSegmentationModel',
                                           encoder='vitb_rn50_384')

# best_model = ISPModel(architecture='DPTSegmentationModel', encoder='vitl16_384',
#                                            pretrained_weights=CHECKPOINT)

# best_model = DPTSegmentationModel(
#     len(LASER_CLASSES),
#     path=CHECKPOINT,
#     backbone='vitl16_384',
#     encoder_pretrained=True)

best_model = best_model.cuda()

# dm = ISPDataModule(DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
#                    augment_p=AUGMENT_P, use_normalize=USE_NORMALIZE,
#                    half_normalize=HALF_NORMALIZE, return_paths=RETURN_PATHS,
#                    input_resize=IM_SIZE, use_binary_mask=USE_BINARY_MASK)
#
# dm.prepare_data()
# dm.setup()
#
# test_loader = dm.test_dataloader()

colours = {
    0: (255, 255, 255),
    1: (0, 0, 255),  # blue
    2: (0, 255, 0),  # green
    3: (255, 0, 0),  # red
    4: (255, 255, 0),  # yellow
}


# Testing
# trainer = pl.Trainer(gpus=1)
# trainer.test(best_model, datamodule=dm)

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

    # if use_normalize:
    #     if half_normalize:
    #         transforms += [A.Normalize(mean=[0.5, 0.5, 0.5],
    #                                    std=[0.5, 0.5, 0.5])]
    #     else:
    #         transforms += [A.Normalize(mean=[0.485, 0.456, 0.406],
    #                                    std=[0.229, 0.224, 0.225])]
    transforms += [ToTensorV2(transpose_mask=True)]
    preprocessing = A.Compose(transforms)

    return preprocessing


def visualize_step(img_path, mask_path, output_path, colours, images_gt_dict, iou_dict, f1_dict, f1_score_list,
                   iou_list):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    #    img, gt_mask, img_path, mask_path = data
    #     img_path = Path(img_path[0])
    #     mask_path = Path(mask_path[0])

    img = img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img_vis = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

    preprocessing = _get_transforms(True, False)
    preprocessed = preprocessing(image=img, mask=gt_mask)
    img, gt_mask = preprocessed['image'], preprocessed['mask']

    # add dimension [3, 512, 512] -> [1, 3, 512, 512]
    img = img.unsqueeze(0)

    gt_mask = gt_mask.squeeze().numpy()

    pr_mask = best_model(img.cuda()).argmax(dim=1).detach().squeeze().cpu().numpy()

    gt_class = list(np.unique(gt_mask))
    pr_class = list(np.unique(pr_mask))

    images_gt_dict[str(img_path)] = gt_class

    iou_classes = []

    inter = 0

    '''
    Итоги по F1_score
    Вычисляем F1_score по каждому классу - их всего 5 (в том числе бэкграунд). Затем находим среднее значение.
    У f1_torchmetrics и f1_pytorch_lightning_metrics это делает параметр average='macro'. 
    По умолчанию используется значение micro, которое, как написано в документации, находит суммарно TP, FP, TN, FN и
    на основе них считает F1_score. В итоге, это значение получается на ~2% выше, чем с параметром 'macro', что считаю нерелевантным результатом.
    
    'macro' нельзя использовать т.к. в случае когда, какого-то класса просто нет на картинке и у него f1_score = 0 этот ноль тоже усредняется 
    и среднее значение становится значительно меньше. 
    
    
    Итоги по IoU
    f1_torchmetrics и f1_pytorch_lightning_metrics вычисляют IoU по каждому классу и потом усредняют. Это нормальное решение.
    '''

    for index, value in enumerate(LASER_CLASSES):
        if index == 0:
            gt_mask_class = np.where(gt_mask == index, 255, gt_mask)
            pr_mask_class = np.where(pr_mask == index, 255, pr_mask)

            gt_mask_class = np.where(gt_mask_class != 255, 0, gt_mask_class)
            pr_mask_class = np.where(pr_mask_class != 255, 0, pr_mask_class)

            gt = np.array(gt_mask_class).reshape(-1, ).tolist()
            pr = np.array(pr_mask_class).reshape(-1, ).tolist()

            # F1_score ###############################################################

            TP = 0
            FP = 0
            TN = 0
            FN = 0

            for i in range(len(gt)):
                if gt[i] == pr[i] == 255:
                    TP += 1
                if pr[i] == 255 and gt[i] != pr[i]:
                    FP += 1
                if gt[i] == pr[i] == 0:
                    TN += 1
                if pr[i] == 0 and gt[i] != pr[i]:
                    FN += 1

            print('TP {}, FP {}, TN {}, FN {}'.format(TP, FP, TN, FN))

            prec = TP / (TP + FP)
            rec = TP / (TP + FN)
            fs = 2 * TP / (2 * TP + FP + FN)

            print('prec {}, rec {}, fs {}'.format(prec, rec, fs))

            print()

            f1_sklearn_metrics_macro = f1_sklearn_metrics(gt_mask_class,
                                                          pr_mask_class,
                                                          labels=[0, 255],
                                                          average='macro')
            print('f1_sklearn_metrics_macro:', f1_sklearn_metrics_macro)

            f1_sklearn_metrics_micro = f1_sklearn_metrics(gt_mask_class,
                                                          pr_mask_class,
                                                          labels=[0, 255],
                                                          average='micro')
            print('f1_sklearn_metrics_micro:', f1_sklearn_metrics_micro)

            f1_sklearn_metrics_weighted = f1_sklearn_metrics(gt_mask_class,
                                                             pr_mask_class,
                                                             labels=[0, 255],
                                                             average='weighted')
            print('f1_sklearn_metrics_weighted:', f1_sklearn_metrics_weighted)

            f1_sklearn_metrics_binary = f1_sklearn_metrics(np.array(gt_mask_class).reshape(-1, ).tolist(),
                                                           np.array(pr_mask_class).reshape(-1, ).tolist(),
                                                           pos_label=255)
            print('f1_sklearn_metrics_binary:', np.mean(f1_sklearn_metrics_binary))

            f1_sklearn_metrics_none = f1_sklearn_metrics(np.array(gt_mask_class).reshape(-1, ).tolist(),
                                                         np.array(pr_mask_class).reshape(-1, ).tolist(),
                                                         average=None)
            print('f1_sklearn_metrics_none:', f1_sklearn_metrics_none)
            print('f1_sklearn_metrics_none_mean:', np.mean(f1_sklearn_metrics_none))

            print()

            gt_mask_class2 = np.where(gt_mask_class == 0, 100, gt_mask_class)
            pr_mask_class2 = np.where(pr_mask_class == 0, 100, pr_mask_class)

            gt_mask_class2 = np.where(gt_mask_class2 == 255, 1, gt_mask_class2)
            pr_mask_class2 = np.where(pr_mask_class2 == 255, 1, pr_mask_class2)

            gt_mask_class2 = np.where(gt_mask_class2 == 100, 0, gt_mask_class2)
            pr_mask_class2 = np.where(pr_mask_class2 == 100, 0, pr_mask_class2)

            # torchmetrics
            f1_torchmetrics = torchmetrics.F1(num_classes=3, average=None)
            f1_torchmetrics_none = f1_torchmetrics(torch.from_numpy(pr_mask_class2), torch.from_numpy(gt_mask_class2))
            print('f1_torchmetrics_none', f1_torchmetrics_none.detach().cpu().numpy())
            print('f1_torchmetrics_none_mean', np.mean(f1_torchmetrics_none.detach().cpu().numpy()))

            f1_torchmetrics = torchmetrics.F1(num_classes=3, average='macro')
            f1_torchmetrics_macro = f1_torchmetrics(torch.from_numpy(pr_mask_class2), torch.from_numpy(gt_mask_class2))
            print('f1_torchmetrics_macro', f1_torchmetrics_macro.detach().cpu().numpy())

            f1_torchmetrics = torchmetrics.F1(num_classes=3)
            f1_torchmetrics_micro = f1_torchmetrics(torch.from_numpy(pr_mask_class2), torch.from_numpy(gt_mask_class2))
            print('f1_torchmetrics_micro_default', f1_torchmetrics_micro.detach().cpu().numpy())

            print()

            # pytorch_lightning_metrics
            f1_pytorch_lightning_metrics = F1(num_classes=2, average=None)
            f1_pytorch_lightning_metrics_none = f1_pytorch_lightning_metrics(torch.from_numpy(pr_mask_class2),
                                                                             torch.from_numpy(gt_mask_class2))
            print('f1_pytorch_lightning_metrics_none', f1_pytorch_lightning_metrics_none.detach().cpu().numpy())
            print('f1_pytorch_lightning_metrics_none_mean',
                  np.mean(f1_pytorch_lightning_metrics_none.detach().cpu().numpy()))

            # np.mean(f1_pytorch_lightning_metrics_none) = f1_pytorch_lightning_metrics_macro
            f1_pytorch_lightning_metrics = F1(num_classes=2, average='macro')
            f1_pytorch_lightning_metrics_macro = f1_pytorch_lightning_metrics(torch.from_numpy(pr_mask_class2),
                                                                              torch.from_numpy(gt_mask_class2))
            print('f1_pytorch_lightning_metrics_macro', f1_pytorch_lightning_metrics_macro.detach().cpu().numpy())

            f1_pytorch_lightning_metrics = F1(num_classes=2)
            f1_pytorch_lightning_metrics_micro = f1_pytorch_lightning_metrics(torch.from_numpy(pr_mask_class2),
                                                                              torch.from_numpy(gt_mask_class2))
            print('f1_pytorch_lightning_metrics_micro_default',
                  f1_pytorch_lightning_metrics_micro.detach().cpu().numpy())

            # IoU ##########################################
            print()

            jaccard_score_sklearn_metrics_ = jaccard_score_sklearn_metrics(
                np.array(gt_mask_class).reshape(-1, ).tolist(),
                np.array(pr_mask_class).reshape(-1, ).tolist(), pos_label=255)
            print('jaccard_score_sklearn_metrics:', jaccard_score_sklearn_metrics_)

            iou_torchmetrics = torchmetrics.IoU(num_classes=2, reduction='none')
            iou_torchmetrics_ = iou_torchmetrics(torch.from_numpy(pr_mask_class2), torch.from_numpy(gt_mask_class2))
            print('iou_torchmetrics', iou_torchmetrics_.detach().cpu().numpy())

            iou_pytorch_lightning_metrics = IoU(num_classes=2)
            iou_pytorch_lightning_metrics_ = iou_pytorch_lightning_metrics(torch.from_numpy(pr_mask_class2),
                                                                           torch.from_numpy(gt_mask_class2))
            print('iou_pytorch_lightning_metrics', iou_pytorch_lightning_metrics_.detach().cpu().numpy())

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

    # f1_score = f1(pr_mask_tensor, gt_mask_tensor).numpy()

    #   print('inter', inter)

    # f1_score2 = f1(torch.from_numpy(pr_mask), torch.from_numpy(gt_mask))
    # print('f1_score2', f1_score2.detach().cpu().numpy())
    # f1_score_list.append(f1_score2.detach().cpu().numpy())

    precision, recall, f1_score, support = score(np.array(gt_mask).reshape(-1, ).tolist(),
                                                 np.array(pr_mask).reshape(-1, ).tolist(), average=None,
                                                 labels=[index for index, value in enumerate(LASER_CLASSES)])

    iou_classes_arr = np.array(iou_classes)
    print()
    print('labels:    {}'.format([format(index, '.5f') for index, value in enumerate(LASER_CLASSES)]))
    print('precision: {}'.format(precision))
    print('recall:    {}'.format(recall))
    print('f1_score:  {}'.format(f1_score))
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
    # fig.savefig(output_path.joinpath(img_path.name))
    fig.savefig(output_path.joinpath(os.path.basename(img_path)))
    fig.clear()
    plt.close(fig)

    return images_gt_dict, f1_dict, iou_dict, f1_score_list


'swin_base_patch4_window12_384'
'swin_base_patch4_window12_384_in22k'

images_gt_dict = {}
iou_dict = {}
f1_dict = {}
f1_score_list = []
iou_list = []

img_paths, mask_paths = _read_imgs_masks(DATA_DIR.joinpath(TEST_FILE))

for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
    #  if i == 21:
    print("Valid image count: {}/{}".format(i + 1, len(img_paths)))
    images_gt_dict, f1_dict, iou_dict, f1_score_list = visualize_step(img_path, mask_path, OUTPUT_TEST_PATH, colours,
                                                       images_gt_dict, iou_dict, f1_dict, f1_score_list, iou_list)

#
# for i, data in enumerate(test_loader):
#     # if i == 4:
#     print("Valid image count: {}/{}".format(i + 1, len(test_loader)))
#     images_gt_dict, f1_dict, iou_dict, f1_score_list = visualize_step(data, OUTPUT_TEST_PATH, colours,
#                                                                       images_gt_dict, iou_dict, f1_dict,
#                                                                       f1_score_list, iou_list)

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
print('F1_score total2: ' + str(np.mean(f1_score_list)))

print()
iou_total = []
print('__IoU per classes__')
for label, values_list in iou_classes.items():
    print('label ' + str(label) + ':   ' + str(np.mean(values_list)))
    if not math.isnan(np.mean(values_list)):
        iou_total.append(np.mean(values_list))

print('IoU total: ' + str(np.mean(iou_total)))
print('IoU total2: ' + str(np.mean(iou_list)))

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
