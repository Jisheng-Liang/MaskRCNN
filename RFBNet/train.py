from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.preprocessing import image
from nets.rfb import rfb300
from nets.rfb_training import MultiboxLoss,Generator
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from keras.optimizers import Adam,SGD
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import cv2
import keras
import os
import sys
from datetime import datetime
 
if __name__ == "__main__":
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
        '/output', 'logs', current_time + '_')
    annotation_path = '2007_train.txt'
    
    NUM_CLASSES = 2
    input_shape = (300, 300, 3)
    priors = get_anchors()
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    # 0.2用于验证，0.8用于训练
    val_split = 0.2
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    model = rfb300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights("/data/KaisingLeung/rfb_weights/rfb_weights.h5", by_name=True, skip_mismatch=True)
    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1)

    BATCH_SIZE = 4
    gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                    (input_shape[0], input_shape[1]),NUM_CLASSES)
    
    for i in range(21):
        model.layers[i].trainable = False
    # if True:
    #     model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
    #                     loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
    #     model.fit_generator(gen.generate(True), 
    #             steps_per_epoch=num_train//BATCH_SIZE/2,
    #             validation_data=gen.generate(False),
    #             validation_steps=num_val//BATCH_SIZE,
    #             epochs=15, 
    #             initial_epoch=0,
    #             callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    if True:
        model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
                        loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE/2,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=30, 
                initial_epoch=0,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    for i in range(21):
        model.layers[i].trainable = True
    if True:
        model.compile(optimizer=Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=100, 
                initial_epoch=30,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])