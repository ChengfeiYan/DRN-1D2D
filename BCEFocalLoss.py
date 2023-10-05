#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:35:00 2020

@author: yunda_si
"""

import torch
import torch.nn as nn
import numpy as np

class BCEFocalLoss(nn.Module):


    def __init__(self, alpha=None, inter=None, clamp=False,reduction='sum'):
        super(BCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.inter = inter
        self.clamp = clamp
        self.reduction = reduction
        if isinstance(alpha,(float,int)):
            self.alpha = torch.tensor(alpha,dtype=torch.float32,requires_grad=True)


    def forward(self, Input, Label):

        device = Label.device
        Constant = 1#torch.tensor(1.0,dtype=torch.float32,requires_grad=True).to(device)

        if self.alpha is not None:
            self.alpha = self.alpha.to(device)#
            weight1 = self.alpha*(2-Input)**2
            weight2 = (Constant-self.alpha)*(1+Input)**3
            loss = -Label*torch.log(Input)*weight1 - (Constant-Label)*torch.log(Constant-Input)*weight2
        else:
            loss = -Label*torch.log(Input) - (Constant-Label)*torch.log(Constant-Input)

        if self.inter is not None:
            Temp = torch.ones_like(Label)
            mask = torch.tril(Temp,-self.inter)+torch.triu(Temp,self.inter)
            mask[Label==-1] = 0
            loss = mask*loss

        if self.clamp:
            length = len(torch.where(torch.sum(Label,dim=0)!=-len(Label))[0])
            topk = int(np.ceil(length*2))
            clamped_pred = torch.ones_like(loss)*0

            distance = {'long':(12,4000)}
            for key in distance:
                internal = distance[key][0]
                external = distance[key][1]

                clamped = torch.triu(Input,diagonal=internal)-torch.triu(Input,diagonal=external)
                clamped[Label==-1] = -1
                cutoff = torch.topk(clamped.view(-1), topk, largest=True, sorted=True)[0][-1]
                clamped[clamped>=cutoff] = 2
                clamped[clamped!=2] = 0
                clamped_pred = clamped_pred+clamped+clamped.transpose(0,1)
            clamped_pred = clamped_pred+1
            loss = clamped_pred*loss

        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)

        return loss