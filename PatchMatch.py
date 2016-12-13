import numpy as np
import numpy.random as npr
import cv2
from debug_tools import *

def _computeOffset(x, y, w, src_cord, tgtNaN, srcNaN): #TODO:x means row, y means column. not consist with conversion
    ofs = tgtNaN[x:x+1+2*w, y:y+1+2*w] - srcNaN[src_cord[0]:src_cord[0]+1+2*w,
                                         src_cord[1]:src_cord[1]+1+2*w]
    ofs = ofs[~np.isnan(ofs)].flatten()
    if len(ofs) == 0:
        return np.inf
    return sum(ofs**2) / len(ofs)


def PatchMatch(tgtImg, need_pad, srcImg, psz=9, mask=None):
    """python implementation of PatchMatch algorithm
    Args:
        tgtImg: An image that we want to reconstruct or An image region contains the miss pixels padded with pixels that is already known
        need_pad: If tgtImg is a full image, this must be true.
        srcImg: An image from which patches are extracted
        psz: patch size (psz x psz), psz must be odd
        mask: 0/1 mask indicates that which part of the srcImg can we extract patch. If not None, mask should be the same size as srcImg with only one channel
    Return:
        NNF: Nearest Neighbor Field, which contains indices of srcImg for each corresponding indices of tgtImg
    """
    npr.seed(123)
    assert psz % 2 == 1
    w = (psz - 1) / 2
    tgt = cv2.normalize(tgtImg, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    src = cv2.normalize(srcImg, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #-------------initializae-----------------
    print "initialize..."
    max_iters = 4
    ssz = src.shape[0:2]
    tsz = tgt.shape[0:2]
    if not need_pad:
        tsz = (tsz[0]-2*w, tsz[1]-2*w)

    if mask is None:
        mask = np.ones(ssz)

    radius = ssz[0] / 4.0 #TODO
    alpha = .5

    Radius = np.round(radius * alpha ** (np.arange(-np.floor(np.log(radius) / np.log(alpha)) + 1)))
    lenRad = len(Radius)

    if need_pad:
        tgt_padded = np.empty((tsz[0]+2*w, tsz[1]+2*w, 3)) * np.nan
        tgt_padded[w:tsz[0]+w, w:tsz[1]+w] = tgt
    else:
        tgt_padded = tgt
    src_NaN = np.empty((ssz[0]+2*w, ssz[1]+2*w, 3,)) * np.nan
    mask[np.where(mask==0)] = np.nan
    src_NaN[w:ssz[0]+w, w:ssz[1]+w] = src * mask[..., np.newaxis] #mask become (512, 512, 1) for broadcast

    #build initialized NNF
    mask1_idx = np.where(~np.isnan(mask)) #type tuple. (x_idx_array, y_idx_array)
    rand_mask1_idx = npr.randint(0, len(mask1_idx[0]), tsz)
    NNF = np.empty(tsz+(2,), dtype=np.int64)
    NNF[:, :, 0] = mask1_idx[0][rand_mask1_idx[:, :]]
    NNF[:, :, 1] = mask1_idx[1][rand_mask1_idx[:, :]]

    #TODO:refine
    for i in xrange(tsz[0]):
        for j in xrange(tsz[1]):
            tgt_padded[i+w, j+w] = src[NNF[i, j, 0], NNF[i, j, 1]]

    offsets = np.empty(tsz)
    for i in xrange(tsz[0]):
        for j in xrange(tsz[1]):
            offsets[i, j] = _computeOffset(i, j, w, NNF[i, j], tgt_padded, src_NaN)


    #--------------main iteration-------------
    for iter in xrange(max_iters):
        is_reverse = iter % 2
        ofs_prp = np.empty(3)
        candidate = np.empty((2, 2,), np.int64)
        i_seq = range(0, tsz[0])
        j_seq = range(0, tsz[1])
        if not is_reverse:
            print "%d th iteration (raster scan order) start!." % (iter+1)
        else:
            print "%d th iteration (reverse raster scan order) start!." % (iter+1)
            i_seq = i_seq[::-1]
            j_seq = j_seq[::-1]


        for i in i_seq:
            for j in j_seq:
                if not is_reverse:
                    #center
                    ofs_prp[0] = offsets[i, j]

                    #top
                    top_nn = NNF[max(0, i-1), j]
                    candidate[0] = top_nn.copy() #top prop candidate
                    candidate[0][0] = candidate[0][0] + 1
                    if candidate[0][0] < ssz[0] and ~np.isnan(mask[candidate[0][0], candidate[0][1]]):
                        ofs_prp[1] = _computeOffset(i, j, w, candidate[0], tgt_padded, src_NaN)
                    else:
                        ofs_prp[1] = np.inf

                    #left
                    left_nn = NNF[i, max(0, j-1)]
                    candidate[1] = left_nn.copy()
                    candidate[1][1] = candidate[1][1] + 1
                    if candidate[1][1] < ssz[1] and ~np.isnan(mask[candidate[1][0], candidate[1][1]]):
                        ofs_prp[2] = _computeOffset(i, j, w, candidate[1], tgt_padded, src_NaN)
                    else:
                        ofs_prp[2] = np.inf

                else:
                    #center, bottom. right
                    ofs_prp[0] = offsets[i, j]

                    #bottom
                    bottom_nn = NNF[min(i+1, tsz[0]-1), j]
                    candidate[0] = bottom_nn.copy()
                    candidate[0][0] = candidate[0][0] - 1
                    if candidate[0][0] >= 0 and ~np.isnan(mask[candidate[0][0], candidate[0][1]]):
                        ofs_prp[1] = _computeOffset(i, j, w, candidate[0], tgt_padded, src_NaN)
                    else:
                        ofs_prp[1] = np.inf

                    #right
                    right_nn = NNF[i, min(j+1, tsz[1]-1)]
                    candidate[1] = right_nn.copy()
                    candidate[1][1] = candidate[1][1] - 1
                    if candidate[1][1] >= 0 and ~np.isnan(mask[candidate[1][0], candidate[1][1]]):
                        ofs_prp[2] = _computeOffset(i, j, w, candidate[1], tgt_padded, src_NaN)
                    else:
                        ofs_prp[2] = np.inf

                idx = np.argmin(ofs_prp)
                offsets[i, j] = ofs_prp[idx]
                if idx == 1: #propagate from top/bottom
                    NNF[i, j] = candidate[0].copy()

                elif idx == 2: #propagate from left/right
                    NNF[i, j] = candidate[1].copy()
                tgt_padded[w + i, w + j] = src[NNF[i, j, 0], NNF[i, j, 1]]
                #-------------Random Search-------------------

                i_min = NNF[i, j, 0] - Radius
                i_min[np.where(i_min<0)] = 0
                i_max = NNF[i, j, 0] + Radius
                i_max[np.where(i_max>ssz[0]-1)] = ssz[0] -1
                j_min = NNF[i, j, 1] - Radius
                j_min[np.where(j_min<0)] = 0
                j_max = NNF[i, j, 1] + Radius
                j_max[np.where(j_max>ssz[1]-1)] = ssz[1] - 1
        
                iis = (np.floor(npr.rand(lenRad) * (i_max - i_min)) + i_min).astype(np.int)
                jjs = (np.floor(npr.rand(lenRad) * (j_max - j_min)) + j_min).astype(np.int)

                ofs_rs = np.empty(lenRad+1)
                ofs_rs[0] = offsets[i, j]
                tmp_idx = 1
                for ni, nj in zip(iis, jjs):
                    if np.isnan(mask[ni, nj]): #TODO
                        ofs_rs[tmp_idx] = np.inf
                        tmp_idx += 1
                        continue
                    tmp = tgt_padded[i:i+1+2*w, j:j+1+2*w] - src_NaN[ni:ni+1+2*w, nj:nj+1+2*w]
                    tmp = tmp[~np.isnan(tmp)].flatten()
                    if len(tmp) == 0:
                        ofs_rs[tmp_idx] = np.inf
                    else:
                        ofs_rs[tmp_idx] = sum(tmp**2) / len(tmp)
                    tmp_idx += 1
                idx = np.argmin(ofs_rs)
                offsets[i, j] = ofs_rs[idx]
                if idx != 0: #random search got one other pixel
                    NNF[i, j] = np.array([iis[idx-1], jjs[idx-1]])
                    tgt_padded[w + i, w + j] = src[NNF[i, j, 0], NNF[i, j, 1]]

    return tgt_padded, NNF
