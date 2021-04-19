#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 21:04:44 2020

@author: yunda_si
"""


from read_feature import cat_feature
import torch
import matplotlib.pyplot as plt
from ResNetB import resnet52
from generate_feature import generate_feature
import os
import numpy as np
import sys


def imshow_pred(pred,protein_id,pred_img):

    fontdict = {'weight':'book','family':'sans-serif',
            'stretch':'ultra-expanded','size':'large',
            'style':'normal'}
    gridspec_kw = {'left':0.05,'bottom':0.05,'right':0.98,'top':0.96}

    fig,ax = plt.subplots(figsize=(10,10),dpi=400,gridspec_kw=gridspec_kw)

    ax.spines['left'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['top'].set_linewidth(0.8)
    ax.grid(linestyle='--')
    ax.set_aspect(aspect='equal',anchor='C')
    ax.imshow(pred,cmap='binary')
    ax.set_title(protein_id,fontdict=fontdict)

    plt.savefig(pred_img,format='png')


def prediction(feature):

    weight_list = ['./scripts/model/1','./scripts/model/2','./scripts/model/3',
                   './scripts/model/4','./scripts/model/5','./scripts/model/6',
                   './scripts/model/7']
    device = 'cuda:0'
    Input = torch.from_numpy(feature).float()[:,0:82,:,:].to(device)
    len_seq = Input.shape[-1]
    outputs_finnal = torch.zeros(len_seq,len_seq).to(device)

    model = resnet52().to(device)
    for weight_file in weight_list:

        model.load_state_dict(torch.load(weight_file))
        model.eval()
        torch.set_grad_enabled(False)
        outputs = torch.squeeze(model(Input))
        outputs_finnal = outputs_finnal+outputs

    outputs_finnal = outputs_finnal.to('cpu')
    pred = (outputs_finnal+outputs_finnal.transpose(0,1))/len(weight_list)
    np.savetxt(pred_txt,pred.numpy())

    return pred


if __name__=="__main__":

    a3m_file=sys.argv[1]
    fasta_file = sys.argv[2]
    save_path=sys.argv[3]
    ncpu=int(sys.argv[4])

    current_path = sys.argv[0]
    current_path = os.path.abspath(current_path)
    a3m_file = os.path.abspath(a3m_file)
    fasta_file = os.path.abspath(fasta_file)
    save_path = os.path.abspath(save_path)
    root_path = os.path.abspath(current_path+'/../../')

    if os.path.exists(a3m_file):
        pass
    else:
        print('ERROR: MSA no exists')
        exit()

    if os.path.exists(fasta_file):
        pass
    else:
        print('ERROR: fasta no exists')
        exit()

    print(root_path)
    os.chdir(root_path)
    name=os.path.splitext(os.path.split(a3m_file)[1])[0]
    sub_path = os.path.join(save_path, name)
    pred_txt = os.path.join(sub_path,'%s.pred'%(name))
    pred_img = os.path.join(sub_path,'%s.png'%(name))


    generate_feature(fasta_file,a3m_file,save_path,root_path,ncpu=ncpu)
    feature = cat_feature(save_path,name,1700)

    print('predict')
    pred = prediction(feature)
    imshow_pred(pred,name,pred_img)