

from meta import *
import dataset_utils
from dataset_utils import *


import numpy as np

import scipy
import scipy.io.wavfile

import cv2

import matplotlib.pyplot as plt
import gc

import random


import os


BLOCK_THRES = 150
FREQ = 512          # cut to 128
DUR = 128
SEG_STEP = 64
MAX_BLOCKS = 100


def _squeezec(spec):
    newchan = int(spec.shape[0]//2)
    clist = np.arange(newchan)
    newspec = (spec[2*clist] + spec[2*clist+1]) / 2
    return newspec

def _wav2arrays(wav):
    # spec of all
    fs, x = scipy.io.wavfile.read(wav)
    spec, _, _, _= plt.specgram(x, Fs=fs, NFFT=2048, noverlap=1900); plt.close('all'); gc.collect()

    # pre - cut/modify
    # FREQ //= 4
    spec = spec[:FREQ]
    ### mean
    spec[spec > 255] = 255; spec /= 255
    spec = _squeezec(spec)[:FREQ//4]

    # specs
    specs = []
    num_segs = 1 + (spec.shape[1] - DUR)//SEG_STEP
    real_segs = 0
    print(num_segs)
    for i in range(num_segs):
        new = spec[:, i*SEG_STEP:i*SEG_STEP+DUR]
        if np.sum(new) > 500:
            real_segs += 1
            specs.append(new.reshape( int(FREQ*DUR/4) ))

    # atten
    if 'B' in wav: tp = 1
    else:          tp = 0

    attens = np.zeros( (real_segs,2), dtype=float)
    attens[:,tp] = 1
    attens = list(attens)

    return specs, attens



global block_count
global block_specs
global block_attens

def createBlocks():
    global block_count
    global block_specs
    global block_attens

    # As and Bs
    wav_clips = os.listdir(wav_clip_root)
    r = 0
    while r < len(wav_clips):
        if '.wav' not in wav_clips[r] or '.png' in wav_clips[r]: del wav_clips[r]
        else: r += 1
    random.shuffle(wav_clips)


    # block creation
    ''' blocks have names: blockXX.json, having ~THRES items in each '''
    to_train = False
    block_count = 0
    block_specs = []
    block_attens = []
    
    def _dump_specs_attens():
        global block_count
        global block_specs
        global block_attens

        # shuffle
        indices = np.arange(len(block_attens))
        random.shuffle(indices)
        block_specs  = (np.array(block_specs)[indices]).tolist()
        block_attens = (np.array(block_attens)[indices]).tolist()

        ## only for test ##
        # [4*i, 4*i + 4]
        if not to_train:
            for i in range(len(block_specs)//4):
                end = min(4*i+4, len(block_specs))
                temp_specs  = block_specs[4*i:end]
                temp_attens = block_attens[4*i:end]
    
                # bisect index keys
                a_indexes = []
                b_indexes = []
                for j in range(len(temp_attens)):
                    if temp_attens[j][0] == 1: a_indexes.append(j)
                    else:                      b_indexes.append(j)
        
                # dump
                # jhand
                jh = open(testset + "block" + str(block_count) + ".json", "w")
    
                # dump
                print("\tready to dump")
                json_dict = { \
                    "entry_number":len(temp_specs), \
                    "entries": (temp_specs, temp_attens), \
                    "a_indexes": a_indexes, \
                    "b_indexes": b_indexes,
                }
                json.dump(json_dict, jh)
                jh.close()
                print("\tdump done")
        
                # indexing
                block_count += 1
                print( "block_count = {} with {} entries".format(block_count, len(temp_specs)) )
        else:
            # bisect index keys
            a_indexes = []
            b_indexes = []
            for j in range(len(block_attens)):
                if block_attens[j][0] == 1: a_indexes.append(j)
                else:                       b_indexes.append(j)

            # dump
            # jhand
            jh = open(trainset + "block" + str(block_count) + ".json", "w")

            # dump
            print("\tready to dump")
            json_dict = { \
                "entry_number":len(block_specs), \
                "entries": (block_specs, block_attens), \
                "a_indexes": a_indexes, \
                "b_indexes": b_indexes,
            }
            json.dump(json_dict, jh)
            jh.close()
            print("\tdump done")
    
            # indexing
            block_count += 1
            print( "block_count = {} with {} entries".format(block_count, len(block_specs)) )


        # clear truck
        block_specs = []
        block_attens = []


    for wav in wav_clips:
        if to_train:
            print("\t", wav)
            specs, attens = _wav2arrays(wav_clip_root + wav)
            num_entries = len(specs)
            
            block_specs.extend(specs)
            block_attens.extend(attens)

            if len(block_specs) + num_entries > BLOCK_THRES:
                _dump_specs_attens()

            if block_count >= MAX_BLOCKS:
                 to_train = False
                 block_count = 0
                 block_specs = []
                 block_attens = []
        else:
            print("\t", wav)
            specs, attens = _wav2arrays(wav_clip_root + wav)
            num_entries = len(specs)

            block_specs.extend(specs)
            block_attens.extend(attens)

            if len(block_specs) + num_entries > BLOCK_THRES:
                _dump_specs_attens()

            if block_count >= MAX_BLOCKS:
                 break


def mix(option):
    blocks = dataset_utils.get_block_names(option)
    print(blocks)

    if option == 'test': 
        dataset_root = testset
        mixed_count = 0
        for blk in blocks:
            print(blk)

            block = json.load(open(dataset_root+blk, "r"))

            specs = np.array(block['entries'][0])
            attens = np.array(block['entries'][1])
            a_indexes = block['a_indexes']
            b_indexes = block['b_indexes']

            if len(a_indexes)==0 or len(b_indexes)==0: continue

            mixed_list = []
            atten_a_list = []
            atten_b_list = []
            b_index_run = 0
            for a_index_run in range(len(a_indexes)):
                for b_index_run in range(len(b_indexes)):
                    mixed = specs[a_indexes[a_index_run]] + specs[b_indexes[b_index_run]]
                    mixed[mixed > 1] = 1
    
                    mixed_list.append(mixed)
                    atten_a_list.append(specs[a_indexes[a_index_run]])
                    atten_b_list.append(specs[b_indexes[b_index_run]])

            mixed_list = np.array(mixed_list).tolist()
            atten_a_list = np.array(atten_a_list).tolist()
            atten_b_list = np.array(atten_b_list).tolist()

            mixed_dict = {"entry_number": len(mixed_list), "entries": [mixed_list, atten_a_list, atten_b_list]}
            jh = open(dataset_root+"mixed_"+ blk[5:-5] +".json", "w")
            json.dump(mixed_dict, jh)
    
            mixed_count += 1

    else:
        dataset_root = trainset
        mixed_count = 0
        for blk in blocks:
            print(blk)
    
            block = json.load(open(dataset_root+blk, "r"))
    
            specs = np.array(block['entries'][0])
            attens = np.array(block['entries'][1])
            a_indexes = block['a_indexes']
            b_indexes = block['b_indexes']
    
            if len(a_indexes)==0 or len(b_indexes)==0: continue
    
            # (mixed, atten, pure) [ |block|-1 .. 0 ]
            mixed_list = []
            atten_list = []
            pure_list  = []
            b_index_run = 0
            for a_index_run in range(len(a_indexes)):
                # a_indexes[a_index_run], b_indexes[b_index_run]
                
                mixed = specs[a_indexes[a_index_run]] + specs[b_indexes[b_index_run]]
                mixed[mixed > 1] = 1
    
                atten = np.zeros(2, dtype=float); atten[0] = 1
                pure = specs[a_indexes[a_index_run]]
    
                mixed_list.append(mixed)
                atten_list.append(atten)
                pure_list.append(pure)
    
                atten = np.zeros(2, dtype=float); atten[1] = 1
                pure = specs[b_indexes[b_index_run]]
    
                mixed_list.append(mixed)
                atten_list.append(atten)
                pure_list.append(pure)
    
                b_index_run += 1
                if b_index_run >= len(b_indexes):
                    b_index_run = 0
    
            mixed_list = np.array(mixed_list).tolist()
            atten_list = np.array(atten_list).tolist()
            pure_list  = np.array(pure_list).tolist()
    
            mixed_dict = {"entry_number": len(atten_list), "entries": [mixed_list, atten_list, pure_list]}
            jh = open(dataset_root+"mixed_"+ blk[5:-5] +".json", "w")
            json.dump(mixed_dict, jh)
    
            mixed_count += 1

