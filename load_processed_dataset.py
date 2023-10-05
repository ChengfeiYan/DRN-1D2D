#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:19:25 2020

@author: yunda_si
"""

import os
import time
import pickle
import sys


def load_processed_dataset(path, datasets_id):
    """
    from the path according to the datasets_id import and join multiple
    datasets

    Parameters
    ----------
    path : string

    datasets_id : string or list of string

    Returns
    -------
    Dataset : list of tensor

    """
    #### check path's type ####
    if not isinstance(path, str):
        print('ERROR: path\'s is not string')
        sys.exit()
    #### check datasets_id's type ####
    if isinstance(datasets_id, list):
        if not all([isinstance(x, str) for x in datasets_id]):
            print('ERROR: sub datasets_id is not string')
            sys.exit()
    elif isinstance(datasets_id, str):
        pass
    else:
        print('ERROR: datasets_id is not string er list')
        sys.exit()

    #### check if path exists ####
    if not os.path.exists(path):
        print('ERROR: path not exist')
        sys.exit()
    #### check if dataset_id exists ####
    for x, sub_id in enumerate(datasets_id):
        if not os.path.exists(os.path.join(path, sub_id)):
            print('ERROR: ({:d}) no such {:s} dataset'.format(x, sub_id))
            sys.exit()

    #### load dataset ####
    dataset = []
    for x, sub_id in enumerate(datasets_id):
        since = time.time()
        fr = open(os.path.join(path, sub_id), 'rb+')
        file = fr.read()
        data = pickle.loads(file, encoding='bytes')
        fr.close()
        dataset = dataset + data
        print('({:d})  load:{:s}  len:{:4d}  len_sum:{:4d}  time:{:5.1f}s'.
              format(x, sub_id, len(data), len(dataset), time.time() - since))

    return dataset
