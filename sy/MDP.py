#!/usr/bin/env python

"""

This python script is the baseline to implement zero-shot learning on each super-class.
The command is:     python MDP.py Animals
The only parameter is the super-class name.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn.linear_model as models
import numpy as np
import pickle
import sys
import re


def main():
    if len(sys.argv) == 2:
        superclass = sys.argv[1]
    else:
        print('Parameters error')
        exit()

    file_feature = 'features_'+superclass+'.pickle'

    # The constants
    classNum = 230
    # testName = {'A': 'a', 'F': 'a', 'V': 'b', 'E': 'b', 'H': 'b'}
    # date = '20180321'

    # Load seen/unseen split
    label_list_path = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/label_list.txt'
    fsplit = open(label_list_path, 'r')
    lines_label = fsplit.readlines()
    fsplit.close()
    list_train = list()
    names_train = list()
    labeluse = list()
    for each in lines_label:
        tokens = each.split(' ')
        labeluse.append(tokens[0])
        # list_train.append(tokens[0])
        # names_train.append(tokens[1])
    labeltemp = labeluse
    train_l = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/train_label.txt'
    with open(train_l) as ff:
        bufferl = ff.readlines()
        for i in range(len(bufferl)):
            tempt = re.split(r'[\t\n]', bufferl[i])
            list_train.append(tempt[0])
            names_train.append(tempt[1])

    # test-label
    # list_test = list()
    for i in range(list_train):
            if list_train[i] in labeluse:
                labeltemp.remove(list_train[i])
    list_test = labeltemp
    # Load attributes
    attrnum = {'t':30}

    attributes_per_class_path = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/attributes_per_class.txt'
    fattr = open(attributes_per_class_path, 'r')
    lines_attr = fattr.readlines()
    fattr.close()
    attributes = dict()
    for each in lines_attr:
        tokens = each.split('\t')
        label = tokens[0]
        attr = tokens[1:]
        if not (len(attr) == attrnum[superclass[0]]):
            print('attributes number error\n')
            exit()
        attributes[label] = attr

    # Load image features
    fdata = open(file_feature, 'rb')
    features_dict = pickle.load(fdata)  # variables come out in the order you put them in
    fdata.close()
    features_all = features_dict['features_all']
    labels_all = features_dict['labels_all']
    images_all = features_dict['images_all']

    # Label mapping
    for i in range(len(labels_all)):
        if labels_all[i][2:] in names_train:
            idx = names_train.index(labels_all[i][2:])
            labels_all[i] = list_train[idx]

    # Calculate prototypes (cluster centers)
    features_all = features_all/np.max(abs(features_all))
    dim_f = features_all.shape[1]
    prototypes_train = np.ndarray((int(classNum/5*4), dim_f))

    dim_a = attrnum[superclass[0]]
    attributes_train = np.ndarray((int(classNum/5*4), dim_a))
    attributes_test = np.ndarray((int(classNum/5*1), dim_a))

    for i in range(len(list_train)):
        label = list_train[i]
        idx = [pos for pos, lab in enumerate(labels_all) if lab == label]
        prototypes_train[i, :] = np.mean(features_all[idx, :], axis=0)
        attributes_train[i, :] = np.asarray(attributes[label])

    for i in range(len(list_test)):
        label = list_test[i]
        attributes_test[i, :] = np.asarray(attributes[label])

    # Structure learning
    LASSO = models.Lasso(alpha=0.01)
    LASSO.fit(attributes_train.transpose(), attributes_test.transpose())
    W = LASSO.coef_

    # Image prototype synthesis
    prototypes_test = (np.dot(prototypes_train.transpose(), W.transpose())).transpose()

    # Prediction
    label = 'test'
    idx = [pos for pos, lab in enumerate(labels_all) if lab == label]
    features_test = features_all[idx, :]
    images_test = [images_all[i] for i in idx]
    prediction = list()

    for i in range(len(idx)):
        temp = np.repeat(np.reshape((features_test[i, :]), (1, dim_f)), len(list_test), axis=0)
        distance = np.sum((temp - prototypes_test)**2, axis=1)
        pos = np.argmin(distance)
        prediction.append(list_test[pos])

    # Write prediction
    fpred = open('pred_'+ superclass + '.txt', 'w')

    for i in range(len(images_test)):
        fpred.write(str(images_test[i])+' '+prediction[i]+'\n')
    fpred.close()


if __name__ == "__main__":
    main()
