# -*- coding: UTF-8 -*-
"""
@author ：yqy
@Project ：地震相实验
@Date ：2022/1/13 9:40
"""
import numpy as np
from scipy.fftpack import fft, ifft, hilbert
import pandas as pd
profileDir = 'seismic'    #存放原始数据文件夹
freq = 1/897
T = 4
H = 1006
W = 590

#瞬时振幅
def getEnv(profile):
    #profile.shape = (590,1006)
    res = []
    for i in range(590):
        st = profile[i, :]
        hilbert_f = hilbert(st)
        res.append(st ** 2 + hilbert_f ** 2)

    res = np.array(res, dtype=np.float32)
    # print(res.shape)
    return res

def getPha(profile):
    res = []
    for i in range(590):
        st = profile[:, i]
        hilbert_f = hilbert(st)
        res.append(st ** 2 + hilbert_f ** 2)

    res = np.array(res, dtype=np.float32)
    return res

def getIfr(profile):
    res = []
    for i in range(590):
        st = profile[:, i]
        hilbert_f = hilbert(st)
        pha = np.arctan(hilbert_f / st)
        ifr = list((pha[1:] - pha[:-1]) / freq)
        ifr.insert(0, ifr[0])
        ifr = np.array(ifr)
        res.append(ifr)

    res = np.array(res, dtype=np.float32)
    print(res.shape)
    return res

#均方根振幅
def getAmp(profile):
    res = []
    for i in range(590):
        k = profile[i,:]
        msa = [sum(k[i:i + T] ** 2) / T for i in range(k.shape[0] - T + 1)]
        for j in reversed(range((T - 1) // 2)):
            msa.insert(0, sum(k[:j + 1 + (T - 1) // 2] ** 2) / ((T - 1) // 2 + j + 1))
            msa.append(sum(k[-j - 1 - (T - 1) // 2:] ** 2) / ((T - 1) // 2 + j + 1))
        msa.append(msa[-1])
        msa = np.sqrt(msa)
        res.append(msa)

    res = np.array(res, dtype=np.float32)
    # print(res.shape)
    return res

def getHz40(profile):
    res = []
    for i in range(590):
        k = profile[:,i]
        sampling_rate = 1 / freq
        fft_size = k.shape[0]  # 频率, 采样点数
        xf = np.fft.rfft(k)
        tmp = np.zeros_like(xf)
        tmp[3 * 10:(3 + 1) * 10] = xf[3 * 10:(3 + 1) * 10]
        x1 = np.fft.irfft(tmp)
        res.append(x1)

    res = np.array(res, dtype=np.float32)
    print(res.shape)
    return res

def getHz50(profile):
    res = []
    for i in range(590):
        k = profile[:,i]
        sampling_rate = 1 / freq
        fft_size = k.shape[0]  # 频率, 采样点数
        xf = np.fft.rfft(k)
        tmp = np.zeros_like(xf)
        tmp[4 * 10:(4 + 1) * 10] = xf[4 * 10:(4 + 1) * 10]
        x1 = np.fft.irfft(tmp)
        res.append(x1)

    res = np.array(res, dtype=np.float32)
    print(res.shape)
    return res

