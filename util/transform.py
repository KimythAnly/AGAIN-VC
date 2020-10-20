import numpy as np
import cv2

def pad(x, seglen, mode='wrap'):
    pad_len = seglen - x.shape[1]
    y = np.pad(x, ((0,0), (0,pad_len)), mode=mode)
    return y

def segment(x, seglen=128, r=None, return_r=False):
    if x.shape[1] < seglen:
        y = pad(x, seglen)
    elif x.shape[1] == seglen:
        y = x
    else:
        if r is None:
            r = np.random.randint(x.shape[1] - seglen)
        y = x[:,r:r+seglen]
    if return_r:
        return y, r
    else:
        return y

def resize(x, dim):
    return cv2.resize(x, dim, interpolation=cv2.INTER_AREA)

def random_scale(mel, allow_flip=False, r=None, return_r=False):
    if r is None:
        r = np.random.random(3)
    rate = r[0] * 0.6 + 0.7 # 0.7-1.3
    dim = (int(mel.shape[1] * rate), mel.shape[0])
    r_mel = resize(mel, dim)

    rate = r[1] * 0.4 + 0.3 # 0.3-0.7
    trans_point = int(dim[0] * rate)
    dim = (mel.shape[1]-trans_point, mel.shape[0])
    if r_mel.shape[1] < mel.shape[1]:
        r_mel = pad(r_mel, mel.shape[1])
    # r_mel[:,trans_point:mel.shape[1]] = cv2.resize(r_mel[:,trans_point:], dim, interpolation=cv2.INTER_AREA)
    r_mel[:,trans_point:mel.shape[1]] = resize(r_mel[:,trans_point:], dim)
    if r[2] > 0.5 and allow_flip:
        ret = r_mel[:,:mel.shape[1]][:,::-1].copy()
    else:
        ret = r_mel[:,:mel.shape[1]]
    if return_r:
        return ret, r
    else:
        return ret