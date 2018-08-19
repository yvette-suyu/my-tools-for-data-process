#!/usr/bin/env python



"""
This python script is to extract features for all images.
The command is:     python feature_extract.py Animals model/mobile_Animals_wgt.h5
The first parameter is the super-class.
The second parameter is the model weight.
The extracted features will be saved at 'features_Animals.pickle'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from tqdm import tqdm
import numpy as np
import pickle
import sys
import os


def main():
    # Parameters
    if len(sys.argv) == 3:
        superclass = sys.argv[1]
        model_weight = sys.argv[2]
    else:
        print('Parameters error')
        exit()

    # The constants
    # classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
    # testName = {'A': 'a', 'F': 'a', 'V': 'b', 'E': 'b', 'H': 'b'}
    # date = '20180321'

    # Feature extraction model
    base_model = VGG19(include_top=True, weights=None,
                           input_tensor=None, input_shape=None,
                           pooling=None, classes=190)
    base_model.load_weights(model_weight)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('global_average_pooling2d_1').output)

    imgdir_train = 'trainval_'+superclass+'/train'
    imgdir_test = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_test_20180813/DatasetA_test'
    # categories = os.listdir(imgdir_train)
    categories = []
    categories.append('train')
    categories.append('test')

    num = 0
    for eachclass in categories:
        if eachclass == 'test':
            classpath = imgdir_test
        else:
            classpath = 'train'
        num += len(os.listdir(classpath))

    print('Total image number = '+str(num))

    features_all = np.ndarray((num, 1024))
    labels_all = list()
    images_all = list()
    idx = 0
    imagelabel = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/image_label.txt'
    readlabel = ['']
    # Feature extraction
    for iter in tqdm(range(len(categories))):
        eachclass = categories[iter]
        if eachclass[0] == '.':
            continue
        if eachclass == 'test':
            classpath = imgdir_test
        else:
            classpath = imgdir_train
        imgs = os.listdir(classpath)

        for eachimg in imgs:
            img_path = classpath+'/'+eachimg
            img = image.load_img(img_path, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)

            features_all[idx, :] = feature
            if eachclass == 'test':
                labels_all.append(eachclass)
            elif eachclass == 'train':
                with open(imagelabel) as rl:
                    readlabel = rl.readlines()
                    for isimage in range(len(readlabel)):
                        if readlabel[isimage][0] == eachimg:
                            labels_all.append(readlabel[isimage][1])
            images_all.append(eachimg)
            idx += 1

    features_all = features_all[:idx, :]
    labels_all = labels_all[:idx]
    images_all = images_all[:idx]
    data_all = {'features_all':features_all, 'labels_all':labels_all,
                'images_all':images_all}

    # Save features
    savename = 'features_' + superclass + '.pickle'
    fsave = open(savename, 'wb')
    pickle.dump(data_all, fsave)
    fsave.close()


if __name__ == "__main__":
    main()
