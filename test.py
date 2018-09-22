# test.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import cv2

import model.fb_model as model
from model.fb_model import ResDAE, ANet

import pickle
import logger
import dataset_utils
from meta import *

import os

import pdb


''' utils '''
def one_hot(zo):
    ans = np.zeros(2, dtype=float)
    ans[zo] = 1
    return ans

def atten(zo):
    try:
        if(zo[0]==1):
            return 0
        elif(zo[1]==1):
            return 1
    except:
        print("atten error")


''' main '''
if __name__=="__main__":

    clear_dir(logroot, kw="events")
    clear_dir(result, kw=".png")

    ''' dataset & indexing: ( mixed , atten , pure ) '''
    block_names = dataset_utils.get_block_names("test")
    block_names.sort(key=lambda b:int(b[5:-5]))
    mixed_names = dataset_utils.get_mix_names("test")
    mixed_names.sort(key=lambda m:int(m[6:-5]))
    test_num = len(block_names)

    ''' training prep '''
    dae = pickle.load(open("dae.pickle", "rb"))
    anet = pickle.load(open("anet.pickle", "rb"))

    lossF = nn.MSELoss()

    ''' testing '''
    for i in range(test_num):
        print("block/mixed #i={}".format(i))

        # mixed
        mixed_specs, a_clean, b_clean = dataset_utils.load_mix('test', mixed_names[i])
        mixed_specs = torch.tensor(mixed_specs, dtype=torch.float)
        bs = mixed_specs.shape[0]

        mixed_attens_a = np.zeros((bs, 2)); mixed_attens_a[:,0] = 1; mixed_attens_a = torch.tensor(mixed_attens_a, dtype=torch.float)
        mixed_attens_b = np.zeros((bs, 2)); mixed_attens_b[:,1] = 1; mixed_attens_b = torch.tensor(mixed_attens_b, dtype=torch.float)

        _a5, _a4, _a3, _a2, _a1 = anet(mixed_attens_a)
        _top = dae.upward(mixed_specs, _a5, _a4, _a3, _a2, _a1)
        recover_a = dae.downward(_top).view(bs, 128, 128)

        _a5, _a4, _a3, _a2, _a1 = anet(mixed_attens_b)
        _top = dae.upward(mixed_specs, _a5, _a4, _a3, _a2, _a1)
        recover_b = dae.downward(_top).view(bs, 128, 128)

        _top = dae.upward(torch.tensor(a_clean, dtype=torch.float))
        single_a = dae.downward(_top).view(bs, 128, 128)

        _top = dae.upward(torch.tensor(b_clean, dtype=torch.float))
        single_b = dae.downward(_top).view(bs, 128, 128)

        # [0..bs-1]
        for j in range(bs):
            dirname = "testid={}_{}/".format(i,j)
            try:
                os.mkdir(result+dirname)
            except:
                remove_dir(result+dirname)
                os.mkdir(result+dirname)

            loss = lossF(recover_a, torch.tensor(a_clean, dtype=torch.float))
            print("loss = {}".format(loss))

            cv2.imwrite(result+dirname+"testid={}_{}_mixed.png".format(i, j), mixed_specs[j].view(128, 128).detach().numpy()*255)
            cv2.imwrite(result+dirname+"testid={}_{}_recover_atten=0.png".format(i, j), recover_a[j].detach().numpy()*255)
            cv2.imwrite(result+dirname+"testid={}_{}_recover_atten=1.png".format(i, j), recover_b[j].detach().numpy()*255)
            cv2.imwrite(result+dirname+"testid={}_{}_clean_0.png".format(i, j), a_clean[j].reshape(128, 128)*255)
            cv2.imwrite(result+dirname+"testid={}_{}_clean_1.png".format(i, j), b_clean[j].reshape(128, 128)*255)
            cv2.imwrite(result+dirname+"testid={}_{}_single_0.png".format(i, j), single_a[j].detach().numpy().reshape(128, 128)*255)
            cv2.imwrite(result+dirname+"testid={}_{}_single_1.png".format(i, j), single_b[j].detach().numpy().reshape(128, 128)*255)

        # single
        # single_specs, single_attens = dataset_utils.load_trivial('test', block_names[i])
        # bs = single_specs.shape[0]

        # # [0..bs-1]
        # for j in range(bs):
            # cv2.imwrite(result+"testid={}_{}_clean_{}.png".format(i, j, atten(single_attens[j])), single_specs[j].reshape(128, 128)*255)
        #     cv2.imwrite(result+"atten{}_testid={}_{}.png".format(atten(single_attens[j]), i, j), recover[j].view(128, 128).detach().numpy()*255)


# end