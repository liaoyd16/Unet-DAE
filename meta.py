import os

proj_root = os.path.expanduser("~/DAE_Cocktail/")
logroot = os.path.expanduser("~/DAE_Cocktail/log/")


wav_clip_root = os.path.expanduser("~/DAE_Cocktail/wav_clips/")
trainset = os.path.expanduser("~/DAE_Cocktail/trainset/")
testset = os.path.expanduser("~/DAE_Cocktail/testset/")


training_result = os.path.expanduser("~/DAE_Cocktail/training_result/")
result = os.path.expanduser("~/DAE_Cocktail/result/")
probing = os.path.expanduser("~/DAE_Cocktail/probing/")
training_probing = os.path.expanduser("~/DAE_Cocktail/training_probing/")

debugset = os.path.expanduser("~/DAE_Cocktail/debugset/")


def clear_dir(abs_path, kw = ""):
    import shutil
    os.chdir(abs_path)
    filelist = os.listdir()
    for f in filelist:
        if not kw=="" and kw not in f: continue
        try:
            os.remove(f)
        except:
            shutil.rmtree(f)
    os.chdir(proj_root)

def remove_dir(abs_path):
    import shutil
    try:
        os.remove(abs_path)
    except:
        shutil.rmtree(abs_path)