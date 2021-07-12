from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from project.src.isp_datamodule import ISPDataModule
from project.src.isp_model import ISPModel

import datetime

### train settings ##############################################
# data_dir = Path('data/laser_v2')
data_dir = Path('data/laser_v2')
recreate_data = True
return_paths = True
num_workers = 8
augment_p = 0.7
use_binary_mask = False

init_lr = 2.5e-3
min_epochs = 100
max_epochs = 1000
progress_bar_refresh_rate = 1
early_stop_patience = 40

################################################################

FPN_efficientnet_b0 = {'model_type': 'FPN_efficientnet_b0', 'architecture': 'FPN', 'encoder': 'efficientnet-b0',
                       'use_normalize': True,
                       'half_normalize': False,
                       'im_size': 512,
                       'batch_size': 8, 'weights': 'imagenet'}

FPN_resnet34 = {'model_type': 'FPN_resnet34', 'architecture': 'FPN', 'encoder': 'resnet34',
                'use_normalize': True,
                'half_normalize': False,
                'im_size': 512,
                'batch_size': 1,
                'weights': 'imagenet'}

DPTSegmentationModel_hybrid = {'model_type': 'DPTSegmentationModel_hybrid', 'architecture': 'DPTSegmentationModel',
                        'encoder': 'vitb_rn50_384',
                        'use_normalize': True,
                        'half_normalize': False,
                        'im_size': 512,
                        'batch_size': 2,
                        'weights': 'models/dpt_hybrid-ade20k-53898607.pt'}

DPTSegmentationModel_large = {'model_type': 'DPTSegmentationModel_large', 'architecture': 'DPTSegmentationModel',
                        'encoder': 'vitl16_384',
                        'use_normalize': True,
                        'half_normalize': False,
                        'im_size': 384,
                        'batch_size': 2,
                        'weights': 'models/dpt_large-ade20k-b12dca68.pt'}

models = [DPTSegmentationModel_hybrid]

for m in models:
    print('####################### START Training ' + m['model_type'] + ' #######################')

    model_type = m['model_type']
    architecture = m['architecture']
    encoder = m['encoder']
    batch_size = m['batch_size']
    im_size = m['im_size']
    use_normalize = m['use_normalize']
    half_normalize = m['half_normalize']
    pretrained_checkpoint = m['weights']

    # batch_size = 2
    # model_type = 'DPTSegmentationModel'

    # Prepare DataModule
    dm = ISPDataModule(data_dir, batch_size=batch_size, num_workers=num_workers,
                       augment_p=augment_p, use_normalize=use_normalize,
                       half_normalize=half_normalize, return_paths=return_paths,
                       input_resize=im_size, use_binary_mask=use_binary_mask, recreate_data=recreate_data)

    dm.prepare_data()
    dm.setup()  # dm.setup() will call in trainer.fit() but I call it here manually

    # Init our model
    model = ISPModel(architecture, encoder, pretrained_checkpoint, learning_rate=init_lr)

    # Logs for tensorboard
    experiment_name = model_type
    logger = TensorBoardLogger('tb_logs', name=experiment_name)
    checkpoint_name = experiment_name + '_{epoch}_{val_loss:.3f}_{val_f1_epoch:.3f}_{val_iou_epoch:.3f}'

    checkpoint_callback_loss = ModelCheckpoint(monitor='val_loss', mode='min',
                                               filename=checkpoint_name,
                                               verbose=True, save_top_k=1,
                                               save_last=False)

    checkpoint_callback_f1 = ModelCheckpoint(monitor='val_f1_epoch', mode='max',
                                             filename=checkpoint_name,
                                             verbose=True, save_top_k=1,
                                             save_last=False)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=early_stop_patience,
        verbose=True,
        mode='min'
    )

    checkpoints = [checkpoint_callback_f1, checkpoint_callback_loss, early_stop_callback]
    callbacks = checkpoints

    trainer = pl.Trainer(min_epochs=min_epochs,
                         max_epochs=max_epochs,
                         progress_bar_refresh_rate=progress_bar_refresh_rate,
                         gpus=1,
                         logger=logger,
                         callbacks=callbacks)

    # Train the model ⚡🚅⚡
    trainer.fit(model, datamodule=dm)

    # Evaluate the model on the held out test set ⚡⚡
    results = trainer.test(model, datamodule=dm)[0]

    # print(results)

    # save test results
    best_checkpoint = 'best_checkpoint: ' + trainer.checkpoint_callback.best_model_path
    results['best_checkpoint'] = best_checkpoint

    filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_test_loss_' + str(
        round(results.get('test_loss'), 4)) + '_test_f1_epoch_' + str(
        round(results.get('test_f1_epoch'), 4)) + '_test_iou_epoch_' + str(
        round(results.get('test_iou_epoch'), 4)) + '.txt'

    path = 'test_logs' + '/' + model_type
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(path + '/' + filename, 'w+') as f:
        print(results, file=f)

    print('####################### END Training ' + m['model_type'] + ' #######################')
