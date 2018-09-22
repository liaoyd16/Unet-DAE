
import wav2blocks
import os
from meta import *

# os.chdir("trainset")
# filelist = os.listdir()
# for f in filelist:
#     if ".json" not in f: continue
#     os.remove(f)
# os.chdir(proj_root)

# os.chdir("testset")
# filelist = os.listdir()
# for f in filelist:
#     if ".json" not in f: continue
#     os.remove(f)
# os.chdir(proj_root)

wav2blocks.createBlocks()
# wav2blocks.mix('test')
# wav2blocks.mix('train')