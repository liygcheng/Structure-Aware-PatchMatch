import cv2
import numpy as np
from PatchMatch import PatchMatch
from debug_tools import *
#tgt_img = cv2.imread('lena.bmp')
#src_img = cv2.imread('lena.bmp')


tgt_img = cv2.imread('img.jpg')
tgt_img[180:240, 220:260, :] = 0
src_img = cv2.imread('img.jpg')
src_img[180:240, 220:260, :] = 0
mask = np.ones((512, 512))
mask[180:240, 220:260] = 0
rebuild_padded, NNF = PatchMatch(tgt_img[176:244, 216:264], False, src_img, mask=mask)

rebuild_padded = cv2.normalize(rebuild_padded, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
tgt_img[180:240, 220:260] = rebuild_padded[4:-4, 4:-4]

cv2.imshow('win', tgt_img)
cv2.waitKey()


