# Authorized by Haeyong Kang.

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random

import argparse,time
import math
from copy import deepcopy
from itertools import combinations, permutations

from utils import *

from networks.subnet import SubnetLinear, SubnetConv2d
from networks.alexnet import SubnetAlexNet_norm as AlexNet
from networks.lenet import SubnetLeNet as LeNet
from networks.utils import *

from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

import importlib
from utils_rle import compress_ndarray, decompress_ndarray, comp_decomp_mask
from utils_huffman import comp_decomp_mask_huffman


def eval():
    sclass = []
    sclass.append(' beaver, dolphin, otter, seal, whale,')  # aquatic mammals
    sclass.append(' aquarium_fish, flatfish, ray, shark, trout,')  # fish
    sclass.append(' orchid, poppy, rose, sunflower, tulip,')  # flowers
    sclass.append(' bottle, bowl, can, cup, plate,')  # food
    sclass.append(' apple, mushroom, orange, pear, sweet_pepper,')  # fruit and vegetables
    sclass.append(' clock, computer keyboard, lamp, telephone, television,')  # household electrical devices
    sclass.append(' bed, chair, couch, table, wardrobe,')  # household furniture
    sclass.append(' bee, beetle, butterfly, caterpillar, cockroach,')  # insects
    sclass.append(' bear, leopard, lion, tiger, wolf,')  # large carnivores
    sclass.append(' bridge, castle, house, road, skyscraper,')  # large man-made outdoor things
    sclass.append(' cloud, forest, mountain, plain, sea,')  # large natural outdoor scenes
    sclass.append(' camel, cattle, chimpanzee, elephant, kangaroo,')  # large omnivores and herbivores
    sclass.append(' fox, porcupine, possum, raccoon, skunk,')  # medium-sized mammals
    sclass.append(' crab, lobster, snail, spider, worm,')  # non-insect invertebrates
    sclass.append(' baby, boy, girl, man, woman,')  # people
    sclass.append(' crocodile, dinosaur, lizard, snake, turtle,')  # reptiles
    sclass.append(' hamster, mouse, rabbit, shrew, squirrel,')  # small mammals
    sclass.append(' maple_tree, oak_tree, palm_tree, pine_tree, willow_tree,')  # trees
    sclass.append(' bicycle, bus, motorcycle, pickup_truck, train,')  # vehicles 1
    sclass.append(' lawn_mower, rocket, streetcar, tank, tractor,')  # vehicles 2


    order = np.array([16, 14, 0, 15, 8, 13, 11, 12, 1, 7, 3, 17, 4, 5, 6, 2, 18, 19, 9, 10]) 
    
    path = 'results_cifar100_superclass100/csnb_cifar100_100_curr/cifar100_superclass100_SEED_0_LR_0.001_SPARSITY_0.20_hard_soft0.0_grad1.0_0.acc.npy'
    acc = np.load(path)[-1]
    new_acc = np.zeros(acc.shape)
    for i in range(len(order)):
        new_acc[order[i]] = acc[i]
    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    eval()