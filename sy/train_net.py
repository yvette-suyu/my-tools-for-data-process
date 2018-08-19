#!/usr/bin/env python


"""
This python script is train a deep feature extractor (CNN).
The command is:     python train_CNN.py Animals True 0.05
The first parameter is the super-class.
The second parameter is the flag whether data preparation is to be implemented. You should choose True at the first running.
The third parameter is the learning rate of the deep network (MobileNet).
The trained model will be saved at 'model/mobile_Animals_wgt.h5'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import os
import shutil
import sys
import tensorflow as tf


# 指定第一块GPU可用
import keras

config = tf.ConfigProto( device_count = {'GPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # Parameters
    if len(sys.argv) == 4:
        superclass = sys.argv[1]
        imgmove = sys.argv[2]
        if imgmove == 'False':
            imgmove = False
        else:
            imgmove = True
        lr = float(sys.argv[3])
    else:
        print('Parameters error')
        exit()

    # The constants
    # classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
    # testName = {'A': 'a', 'F': 'a', 'V': 'b', 'E': 'b', 'H': 'b'}
    # date = '20180321'

    trainpath = 'trainval_'+superclass+'/train'
    valpath = 'trainval_'+superclass+'/val'

    if not os.path.exists('model'):
        os.mkdir('model')

    # Train/validation data preparation
    if imgmove:
        os.mkdir('trainval_'+superclass)
        os.mkdir(trainpath)
        os.mkdir(valpath)
        sourcepath = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/train'
        categories = os.listdir(sourcepath)
        for eachclass in categories:
            print(eachclass)
            # os.mkdir(trainpath)
            # os.mkdir(valpath)
            imgs = os.listdir(sourcepath)
            idx = 0
            for im in imgs:
                if idx%8 == 0:
                    shutil.copyfile(sourcepath+'/'+im, valpath+'/'+im)
                else:
                    shutil.copyfile(sourcepath+'/'+im, trainpath+'/'+im)
                idx += 1

    # Train and validation ImageDataGenerator
    batchsize = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=5,
        height_shift_range=5,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        trainpath,
        target_size=(64, 64),
        batch_size=batchsize)

    valid_generator = test_datagen.flow_from_directory(
        valpath,
        target_size=(64, 64),
        batch_size=batchsize)

    # Train MobileNet
    # model = MobileNet(include_top=True, weights=None,
    #                   input_tensor=None, input_shape=None,
    #                   pooling=None, classes=230)
    model = VGG19(include_top=True, weights=None,
                                input_tensor=None, input_shape=None,
                                pooling=None,
                                classes=190)
    model.summary()
    model.compile(optimizer=SGD(lr=lr, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    steps_per_epoch = int(train_generator.n/batchsize)
    validation_steps = int(valid_generator.n/batchsize)

    weightname = 'model/vgg19_'+superclass+'_wgt.h5'

    checkpointer = ModelCheckpoint(weightname, monitor='val_loss', verbose=0,
                        save_best_only=True, save_weights_only=True, mode='auto', period=2)
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=100,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=[checkpointer])


if __name__ == "__main__":
    main()
