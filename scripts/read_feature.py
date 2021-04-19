#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:51:18 2020

@author: yunda_si
"""

import pickle
import os
import re
import numpy as np

def read_pssm(pssm_file):
    fr = open(pssm_file,'rb+')
    data = fr.read()
    data_dict = pickle.loads(data,encoding='bytes')
    fr.close()

    PSSM = data_dict['PSSM']
    PSFM = data_dict['PSFM']
    return PSSM,PSFM


def read_acc(acc_file):
    ACC = np.zeros((0,3),dtype=np.float32)
    fr = open(acc_file,'rb+')
    for index,row in enumerate(fr.readlines()):
        if index>2:
            datasplit = re.sub(' +',' ',row.decode()).strip().replace('\n','').split(' ')
            ACC = np.r_[ACC,np.array(datasplit[3:6],dtype=np.float32).reshape(1,3)]
    fr.close()
    return ACC


def read_ss3(acc_file):
    SS3 = np.zeros((0,3),dtype=np.float32)
    fr = open(acc_file,'rb+')
    for index,row in enumerate(fr.readlines()):
        if index>1:
            datasplit = re.sub(' +',' ',row.decode()).strip().replace('\n','').split(' ')
            SS3 = np.r_[SS3,np.array(datasplit[3:6],dtype=np.float32).reshape(1,3)]
    fr.close()
    return SS3


def read_ccmpred(mat_file):
    ccmpred = np.loadtxt(mat_file,dtype=np.float32)
    mean = np.mean(ccmpred)
    std = np.std(ccmpred)
    ccmpredZ = (ccmpred-mean)/std
    ccmpredZ = np.triu(ccmpredZ,6) + np.tril(ccmpredZ,-6)
    return ccmpred,ccmpredZ


def read_alnstats(stats_file):
    temp_pair = np.loadtxt(stats_file,dtype=np.float32)
    length = int(temp_pair.max())
    alnstats = np.zeros((length,length,3))
    for ii in range(len(temp_pair)):
        alnstats[int(temp_pair[ii][0])-1,int(temp_pair[ii][1])-1,:] = temp_pair[ii][2:]
        alnstats[int(temp_pair[ii][1])-1,int(temp_pair[ii][0])-1,:] = temp_pair[ii][2:]
    return alnstats


def cat_feature(save_path,protein_id,max_length):
    pssm_file = os.path.join(save_path, '%s/%s_hhm.pkl'%(protein_id,protein_id))
    acc_file = os.path.join(save_path, '%s/Raptor/%s.acc'%(protein_id,protein_id))
    ss3_file = os.path.join(save_path, '%s/Raptor/%s.ss3'%(protein_id,protein_id))
    mat_file = os.path.join(save_path, '%s/%s.ccmpred'%(protein_id,protein_id))
    stats_file = os.path.join(save_path, '%s/alnstats/%s.pairout'%(protein_id,protein_id))

    PSSM,PSFM = read_pssm(pssm_file)
    ACC = read_acc(acc_file)
    SS3 = read_ss3(ss3_file)
    ccmpred,ccmpredZ = read_ccmpred(mat_file)
    alnstats = read_alnstats(stats_file)

    if len(PSSM)+len(ACC)+len(SS3)+len(ccmpredZ)+len(alnstats) !=len(PSSM)*5:
        print('ERROR')

    od_martix = np.hstack((PSSM, ACC, SS3))
    [len_x, len_y] = od_martix.shape

    single_data = np.zeros((1,82,len_x, len_x),dtype=np.float32)
    for mm in range(len(od_martix)):
        for kk in range(len(od_martix)):
            single_data[0,0:len_y*3,mm,kk] = np.hstack((od_martix[mm,:],od_martix[round((mm+kk)/2),:],od_martix[kk,:]))
            single_data[0,len_y*3:3*len_y+3,mm,kk] = alnstats[mm,kk,:]
    single_data[0,81,:,:] = ccmpredZ

    return single_data




