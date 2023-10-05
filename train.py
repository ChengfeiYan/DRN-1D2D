#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:44:20 2020

@author: yunda_si
"""

from ResNetB import resnet18
import torch
import torch.optim as optim
import time
from random import shuffle
from BCEFocalLoss import BCEFocalLoss
from load_processed_dataset import load_processed_dataset
import pickle
import os
import numpy as np
import random


def top_statistics(string, predicted, contact_map):
    clamping = {'A':(0, 4000), 'S':(6, 12), 'M':(12, 24), 'L':(24, 4000),'U':(50,4000)}
    predicted = 1/2*(predicted + predicted.transpose(0,1))

    internal = clamping[string[0]][0]
    external = clamping[string[0]][1]
    length = len(torch.where(torch.sum(contact_map,dim=0)!=-len(contact_map))[0])
    topk = int(np.ceil(length/float(string[6:8])))

    clamped = torch.triu(predicted, diagonal=internal) - torch.triu(predicted, diagonal=external)
    clamped[contact_map==-1] = -1
    index = torch.topk(clamped.view(-1), topk, largest=True, sorted=True)[1]
    oneline = contact_map.reshape(-1)
    trues = torch.sum(oneline[index]).item()

    return trues/topk


###################              load dataset               ###################
root_path = './Data_Xu/'
local_var = locals()
dataset_id = ['mix_Len0400_1.pkl']
              # 'mix_Len0400_meta_2.pkl',
              # 'mix_Len0400_meta_3.pkl',
              # 'mix_Len0400_meta_4.pkl']

if 'trainset' not in local_var:
    dataset = load_processed_dataset(root_path, dataset_id)

    validset = dataset[0:400]
    trainset = dataset[400:-1]


###################               import net                ###################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = resnet18().to(device)

criterion = BCEFocalLoss(alpha=None, inter=6, clamp=False, reduction='sum')
# optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0000,nesterov=True)

optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                        weight_decay=0.1,amsgrad=False)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                        eps=1e-6, patience=0, factor=0.1, verbose=True)
epoch_num = 25


###################             top statistics              ###################
str_statis = ['UTopLd05','LTopLd05']
dict_statics = {'min_loss':np.inf,'valid_loss':[]}
for string in str_statis:
    dict_statics[string] = {'sum_acc':0,'highest':0,'aver_acc':0,'save':'',
                            'train_acc':[],'valid_acc':[]}


###################               save model                ###################
if os.path.exists('model'):
    pass
else:
    os.mkdir('model')

savepth = './'

for string in str_statis:
    dict_statics[string]['save'] = '{0}_{1}.pth'.format(savepth, string)
loss_save = savepth+'minloss.pth'


###################                training                 ###################
for epoch in range(epoch_num):
    since = time.time()

    shuffle(validset)
    shuffle(trainset)

    print('learning rate: %8.6f' %optimizer.param_groups[0]['lr'])

    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
            dataset = trainset
        else:
            model.eval()
            dataset = validset

        running_loss = 0.0
        for string in str_statis:
            dict_statics[string]['sum_acc'] = 0
        optimizer.zero_grad()

        for d, datasplit in enumerate(dataset):
            inputs = datasplit[:,0:-1,:,:].to(device) #random.randint(0,7)
            contact_map = datasplit[0,-1,:,:].to(device)

            with torch.set_grad_enabled(phase == 'train'):
                preds = model(inputs)
                loss = criterion(preds, contact_map)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item()

            ##################          statistics           ##################
            for string in str_statis:
                accuracy = top_statistics(string, preds, contact_map)
                dict_statics[string]['sum_acc'] += 100*accuracy
                dict_statics[string]['aver_acc'] = dict_statics[string]['sum_acc']/(d+1)

            if (d+1)%60==0:
                acc_list = ['%s: %6.3f'%(i, dict_statics[i]['aver_acc']) for i in str_statis]
                print('[%3d, %4d]  loss:%11.2f  %s'%(epoch, d+1, running_loss,
                                                     '  '.join(acc_list)))
            if (d+1)==len(dataset):
                acc_list = ['%s: %6.3f'%(i, dict_statics[i]['aver_acc']) for i in str_statis]
                print('[%3d, %4d]  loss:%11.2f  %s\n'%(epoch, d+1, running_loss,
                                                     '  '.join(acc_list)))


        if phase == 'train':
            for string in str_statis:
                dict_statics[string]['train_acc'].append(dict_statics[string]['aver_acc'])
        else:
            scheduler.step(running_loss)
            dict_statics['valid_loss'].append(running_loss)
            for string in str_statis:
                dict_statics[string]['valid_acc'].append(dict_statics[string]['aver_acc'])

    ##################                 save                  ##################
    for string in str_statis:
        acc = dict_statics[string]['aver_acc']
        highest = dict_statics[string]['highest']
        if acc>highest:
            print('save_%s:%6.3f  highest: %6.3f'%(string, acc, highest))
            dict_statics[string]['highest'] = dict_statics[string]['aver_acc']

            if os.path.exists(dict_statics[string]['save']):
                os.remove(dict_statics[string]['save'])
            torch.save(model.state_dict(), dict_statics[string]['save'])

    if running_loss<dict_statics['min_loss']:
        print('save_minloss:%11.2f    %11.2f'%(running_loss,dict_statics['min_loss']))
        dict_statics['min_loss'] = running_loss
        torch.save(model.state_dict(), loss_save)

    file = open('%s_dict_statics.pkl'%(savepth), 'wb')
    pickle.dump(dict_statics, file)
    file.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

print('Finished Training')
highest_list = ['%s: %6.3f'%(i, dict_statics[i]['highest']) for i in str_statis]
print('highest:%s'%'  '.join(highest_list))





