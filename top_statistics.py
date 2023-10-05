#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:02:54 2020

@author: yunda_si
"""

import torch
import numpy as np


def top_statistics(Pre_map,contact_map):
    count = 0
    single_statictics = np.ones((32))
    len_contact = len(torch.where(torch.sum(contact_map,dim=0)!=-len(contact_map))[0])

    for Distance in ['short','medium','long','extra']:

        if Distance == 'short':
            Internal = 6
            External = 12
        if Distance == 'medium':
            Internal = 12
            External = 24
        if Distance == 'long':
            Internal = 24
            External = np.inf
        if Distance == 'extra':
            Internal = 50
            External = np.inf

        TRIUP = torch.triu(Pre_map,Internal) - torch.triu(Pre_map,External)
        TRIUP[contact_map==-1] = -1
        Label = contact_map.view(-1)

        for TOP in ['TopL','TopL/2','TopL/5','TopL/10']:

            if TOP == 'TopL':
                Topk = len_contact
            if TOP == 'TopL/2':
                Topk = np.ceil(len_contact/2)
            if TOP == 'TopL/5':
                Topk = np.ceil(len_contact/5)
            if TOP == 'TopL/10':
                Topk = np.ceil(len_contact/10)
            Topk = int(Topk)

            SortedP = torch.topk(TRIUP.view(-1),Topk,largest=True, sorted=True)[1]
            TTL = torch.sum(Label[SortedP]).item()

            single_statictics[count] = Topk
            single_statictics[count+1] = TTL/Topk
            count = count+2

    return single_statictics