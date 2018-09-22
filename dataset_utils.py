

import json
from meta import *
import os
import cv2

import numpy as np

'''
block = {"entry_number": entry_number, "entries": entries}
entries = [entry]
entry = float[32768..0]
'''

def get_block_names(option):
    # train / test
    if option == 'train':
        flist = os.listdir(trainset)
    else:
        flist = os.listdir(testset)

    i = 0
    while i < len(flist):
        if "block" not in flist[i] or ".json" not in flist[i]: del flist[i]
        else: i += 1

    return flist


def get_mix_names(option):
    # train / test
    if option == 'train':
        flist = os.listdir(trainset)
    else:
        flist = os.listdir(testset)

    i = 0
    while i < len(flist):
        if "mix" not in flist[i] or ".json" not in flist[i]: del flist[i]
        else: i += 1

    return flist


def get_entry_num(blockname):
    block_dict = json.load(open(blockname, "r"))
    return block_dict['entries']


def load_trivial(option, blk):
    print(blk)
    if option=='train': dataset = trainset
    else:               dataset = testset
    block = json.load( open(dataset+blk, "r") )

    total = block['entry_number']
    specs = np.array(block['entries'][0]).reshape(total, 128, 128)
    attens = np.array(block['entries'][1]).reshape(total, 2)

    return specs, attens


def load_mix(option, blk):
    print(blk)
    if option=='train':
        dataset = trainset
        block = json.load( open(dataset+blk, "r") )

        total = block['entry_number']
        mixture = np.array(block['entries'][0]).reshape(total, 128, 128)
        attens = np.array(block['entries'][1]).reshape(total, 2)
        clean = np.array(block['entries'][2]).reshape(total, 128, 128)

        return mixture, attens, clean
    else:
        dataset = testset
        block = json.load( open(dataset+blk, "r") )

        total = block['entry_number']
        mixture = np.array(block['entries'][0]).reshape(total, 128, 128)
        a_clean = np.array(block['entries'][1]).reshape(total, 128, 128)
        b_clean = np.array(block['entries'][2]).reshape(total, 128, 128)

        return mixture, a_clean, b_clean