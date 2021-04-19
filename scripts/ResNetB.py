# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:28:07 2020

@author: yunda_si
"""

import torch.nn as nn
import torch
from collections import OrderedDict


def make_conv_layer(in_channels,
                    out_channels,
                    kernel_size,
                    padding_size,
                    non_linearity=True,
                    instance_norm=False,
                    dilated_rate=1):
    layers = []

    layers.append(
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                           padding=padding_size, dilation=dilated_rate, bias=False)))
    if instance_norm:
        layers.append(('in', nn.InstanceNorm2d(num_features=in_channels,momentum=0.1,
                                        affine=True,track_running_stats=False)))
    if non_linearity:
        layers.append(('leaky', nn.LeakyReLU(negative_slope=0.01,inplace=True)))

    layers.append(
        ('conv2', nn.Conv2d(in_channels, out_channels, kernel_size,
                           padding=padding_size, dilation=dilated_rate, bias=False)))
    if instance_norm:
        layers.append(('in2', nn.InstanceNorm2d(num_features=in_channels,momentum=0.1,
                                        affine=True,track_running_stats=False)))


    return nn.Sequential(OrderedDict(layers))


def make_1x1_layer(in_channels,
                    out_channels,
                    kernel_size,
                    padding_size,
                    non_linearity=True,
                    instance_norm=False,
                    dilated_rate=1):
    layers = []

    layers.append(
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1,
                           padding=0, dilation=1, bias=False)))
    if instance_norm:
        layers.append(('in', nn.InstanceNorm2d(num_features=out_channels,momentum=0.1,
                                        affine=True,track_running_stats=False)))
    if non_linearity:
        layers.append(('leaky', nn.LeakyReLU(negative_slope=0.01,inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class BasicBlock(nn.Module):

    def __init__(self, in_channels,
                       out_channels,
                       dilated_rate):
        super(BasicBlock, self).__init__()

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.dilated_rate = dilated_rate
        self.concatenate = False
        self.threshold = [1,20,40]
        Bool = True

        self.conv_3x3 = make_conv_layer(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(3,3),
                                        padding_size=(dilated_rate,dilated_rate),
                                        non_linearity=Bool,
                                        instance_norm=Bool,
                                        dilated_rate=(dilated_rate,dilated_rate))

        if dilated_rate in self.threshold:
            self.conv_1xn = make_conv_layer(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(1,9),
                                            padding_size=(0,4*dilated_rate),
                                            non_linearity=Bool,
                                            instance_norm=Bool,
                                            dilated_rate=(1,dilated_rate))

            self.conv_nx1 = make_conv_layer(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(9,1),
                                            padding_size=(4*dilated_rate,0),
                                            non_linearity=Bool,
                                            instance_norm=Bool,
                                            dilated_rate=(dilated_rate,1))

            if self.concatenate:
                self.conv_1x1 = make_1x1_layer(in_channels=in_channels*3,
                                               out_channels=out_channels,
                                               kernel_size=(1,1),
                                               padding_size=(0,0),
                                               non_linearity=Bool,
                                               instance_norm=Bool,
                                               dilated_rate=(1,1))

    def forward(self, x):

        out = x

        identity1 = self.conv_3x3(x)

        if self.dilated_rate in self.threshold:
            identity2 = self.conv_1xn(x)
            identity3 = self.conv_nx1(x)

            if self.concatenate:
                identity = torch.cat((identity1,identity2,identity3),1)
                identity = self.conv_1x1(identity)
            else:
                identity = identity1+identity2+identity3

        else:
            identity = identity1

        out = out+identity
        out = self.leakyrelu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, blocks_num):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.first_layer = make_1x1_layer(in_channels=82,
                                          out_channels=64,
                                          kernel_size=(1,1),
                                          padding_size=(0,0),
                                          non_linearity=True,
                                          instance_norm=True,
                                          dilated_rate=(1,1))

        self.hidden_layer = self._make_layer(64, blocks_num,1)

        self.output_layer = make_1x1_layer(in_channels=64,
                                          out_channels=1,
                                          kernel_size=(1,1),
                                          padding_size=(0,0),
                                          non_linearity=False,
                                          instance_norm=False,
                                          dilated_rate=(1,1))

        self.Sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.weight,0.001)

    def _make_layer(self, out_channel, block_num, dilated_rate):

        layers = []

        for index in range(block_num):
            layers.append(('block'+str(index),BasicBlock(self.in_channel, out_channel, dilated_rate)))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):

        x = self.first_layer(x)

        x = self.hidden_layer(x)

        x = self.output_layer(x)

        x = torch.clamp(x,min=-15,max=15)
        x = self.Sig(x)

        return x


def resnet152():
    return ResNet(75)

def resnet52():
    return ResNet(25)

def resnet26():
    return ResNet(12)

def resnet18():
    return ResNet(8)


