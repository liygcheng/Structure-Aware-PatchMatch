import numpy as np

def NNF_out_of_mask(NNF):
    for i in range(512):
        for j in range(512):
            if NNF[i, j, 0] >= 180 and NNF[i, j, 0] < 240 and NNF[i, j, 1] >= 220 and NNF[i, j, 1] < 260:
                print "haha"
                return True

def NNF_out_of_range(NNF):
    if (NNF[:, :, 0] < 0).any() or (NNF[:, :, 0] >= 512).any():
        print "haha"
        return True
    if (NNF[:, :, 1] < 0).any() or (NNF[:, :, 1] >= 512).any():
        print "haha"
        return True