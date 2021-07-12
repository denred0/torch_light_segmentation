import tensorflow as tf
import keras
from keras import backend as K

import numpy as np


def Tversky_Loss(y_true, y_pred, smooth=1, alpha=0.3, beta=0.7, flatten=False):
    if flatten:
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

    TP = K.sum(y_true * y_pred)
    FP = K.sum((1 - y_true) * y_pred)
    FN = K.sum(y_true * (1 - y_pred))

    tversky_coef = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - tversky_coef


def Focal_Loss(y_true, y_pred, alpha=0.8, gamma=2.0, flatten=False):
    if flatten:
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = K.exp(-bce)

    loss = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)
    return loss


def weighted_bce(weight=0.6):
    def convert_2_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def weighted_binary_crossentropy(y_true, y_pred):
        y_pred = convert_2_logits(y_pred)
        y_pred = tf.squeeze(y_pred, 0)
        y_pred = tf.math.argmax(y_pred, 0)
        print('y_true', np.unique(y_true))
        y_true = tf.squeeze(y_true, 0)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=weight)
        return loss

    return weighted_binary_crossentropy


def Combo_Loss(y_true, y_pred, a=0.4, b=0.2, c=0.4):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    return a * weighted_bce()(y_true, y_pred) + b * Focal_Loss(y_true_f, y_pred_f) + c * Tversky_Loss(y_true_f,
                                                                                                      y_pred_f)


def Dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def Dice_loss(y_true, y_pred):
    return 1.0 - Dice_coef(y_true, y_pred)
