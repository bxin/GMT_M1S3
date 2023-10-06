import os
import sys
import h5py
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

from os.path import expanduser

#import MySQLdb as mdb
#from sqlalchemy import create_engine
##from sqlalchemy import exc

#from FATABLE import *

#m3ORC = 2.508
#m3IRC = 0.550
#m1ORC = 4.18
#m1IRC = 2.558

home = os.path.expanduser("~")
dataDir  = './' #os.path.join(home, 'largeData', 'M1M3_ML')
#BMPatchDir = os.path.join(dataDir, 'LSST_BM_patch_190508')

#fat = np.array(FATABLE)
#actID = np.int16(fat[:, FATABLE_ID])
#nActuator = actID.shape[0]
#xact = np.float64(fat[:, FATABLE_XPOSITION])
#yact = np.float64(fat[:, FATABLE_YPOSITION])

def readH5Map(fileset, dataset = '/dataset'):
    '''
    The method takes a list of h5 files, get the images, and average them to get a combined image array.
    input:
    fileset is a list of filenames, which can be relative path for the dataset, or absolute path.
    dataset refers to where the image array is stored in the h5 file
    output:
    data, which is the averaged image array
    centerRow, centerCol, pixelSize
    '''
    i = 0
    if len(fileset) == 0:
        print('Error: empty fileset')
        sys.exit()
    for filename in fileset:
        #print(filename)
        h5file = os.path.join(dataDir, filename)
        f = h5py.File(h5file,'r')
        data0 = f[dataset]
        if 'date' in data0.attrs.keys():
            if len(data0.attrs['date']) == 1:
                timeStamp = data0.attrs['date'][0].decode('ascii')
            else:
                timeStamp = data0.attrs['date'].decode('ascii')
        else:
            timeStamp = 'date not in h5 file.'
        if i==0:
            centerRow = data0.attrs['centerRow']
            centerCol = data0.attrs['centerCol']
            pixelSize = data0.attrs['pixelSize']
            data = data0[:]
        else:
            data += data0[:]
        f.close()
        if filename.find(dataDir) == 0:
            filenameShort = filename[len(dataDir):]
        else:
            filenameShort = filename
        #print('%s: %s is %d x %d, pixelSize = %.4f'%(
        #    filenameShort, dataset, data.shape[0], data.shape[1], pixelSize))

        print('%s: %s '%(filenameShort, timeStamp))
        i+=1
    data /= i
    data = np.rot90(data, 1) # so that we can use imshow(data, origin='lower')
    return data, centerRow, centerCol, pixelSize

def getH5date():
    return 1

def mkXYGrid(s, centerRow, centerCol, pixelSize):
    '''
    construct the x and y mesh grid corresponding to the image array in the h5 files.
    '''
    [row, col] = s.shape
    xVec = np.arange(1, col+1)
    xVec = (xVec - centerCol) * pixelSize
    yVec = np.arange(1, row+1)
    yVec = (yVec - centerRow) * pixelSize #if we don't put negative sign, we have to flipud the image array
    [x, y] = np.meshgrid(xVec, yVec)
    return x,y

def mkM1M3disp(m1s, m3s, x1, y1, x3, y3):
    '''
    takes the m1 and m3 surfaces, interpolate m3 onto m1 grid, so that we can display then as one plot.
    '''
    s = m1s
    r1 = np.sqrt(x1**2 + y1**2)
    idx = (r1<m3ORC)*(r1>m3IRC)
    m3s[np.isnan(m3s)] = 0
    f = interpolate.interp2d(x3[0,:], y3[:,0], m3s)
    s_temp = f(x1[0,:], y1[:,0])
    s[idx] = s_temp[idx]
    return s


def ZernikeMaskedFit(S, x, y, numTerms, mask, e):

    j, i = np.nonzero(mask[:])
    S = S[i, j]
    x = x[i, j]
    y = y[i, j]
    if (e > 0):
        Z = ZernikeAnnularFit(S, x, y, numTerms, e)
    else:
        Z = ZernikeFit(S, x, y, numTerms)
    return Z

def ZernikeFit(S, x, y, numTerms):
    # print x.shape
    # if x,y are 2D, m1,m2 are lists, still (m1!=m2) below works
    m1 = x.shape
    m2 = y.shape
    if((m1 != m2)):
        print('x & y are not the same size')

    S = S[:].copy()
    x = x[:].copy()
    y = y[:].copy()

    i = np.isfinite(S + x + y)
    S = S[i]
    x = x[i]
    y = y[i]

    H = np.zeros((len(S), int(numTerms)))

    for i in range(int(numTerms)):
        Z = np.zeros(int(numTerms))
        Z[i] = 1
        H[:, i] = ZernikeEval(Z, x, y)

    Z = np.dot(np.linalg.pinv(H), S)

    return Z


def ZernikeEval(Z, x, y):
    '''Evaluate Zernicke'''

    # if x,y are 2D, m1,m2 are lists, still (m1!=m2) below works
    m1 = x.shape
    m2 = y.shape

    # print Z.shape
    # print Z

    if((m1 != m2)):
        print('x & y are not the same size')
        exit()

    if(len(Z) > 28):
        print('ZernikeEval() is not implemented with >28 terms')
        return
    elif len(Z) < 28:
        Z = np.hstack((Z, np.zeros(28 - len(Z))))

    r2 = x * x + y * y
    r = np.sqrt(r2)
    r3 = r2 * r
    r4 = r2 * r2
    r5 = r3 * r2
    r6 = r3 * r3

    t = np.arctan2(y, x)
    s = np.sin(t)
    c = np.cos(t)
    s2 = np.sin(2 * t)
    c2 = np.cos(2 * t)
    s3 = np.sin(3 * t)
    c3 = np.cos(3 * t)
    s4 = np.sin(4 * t)
    c4 = np.cos(4 * t)
    s5 = np.sin(5 * t)
    c5 = np.cos(5 * t)
    s6 = np.sin(6 * t)
    c6 = np.cos(6 * t)

    S = Z[0] * (1 + 0 * x)  # 0*x to set NaNs properly
    S = S + Z[1] * 2 * r * c
    S = S + Z[2] * 2 * r * s
    S = S + Z[3] * np.sqrt(3) * (2 * r2 - 1)
    S = S + Z[4] * np.sqrt(6) * r2 * s2
    S = S + Z[5] * np.sqrt(6) * r2 * c2
    S = S + Z[6] * np.sqrt(8) * (3 * r3 - 2 * r) * s
    S = S + Z[7] * np.sqrt(8) * (3 * r3 - 2 * r) * c
    S = S + Z[8] * np.sqrt(8) * r3 * s3
    S = S + Z[9] * np.sqrt(8) * r3 * c3
    S = S + Z[10] * np.sqrt(5) * (6 * r4 - 6 * r2 + 1)
    S = S + Z[11] * np.sqrt(10) * (4 * r4 - 3 * r2) * c2
    S = S + Z[12] * np.sqrt(10) * (4 * r4 - 3 * r2) * s2
    S = S + Z[13] * np.sqrt(10) * r4 * c4
    S = S + Z[14] * np.sqrt(10) * r4 * s4
    S = S + Z[15] * np.sqrt(12) * (10 * r5 - 12 * r3 + 3 * r) * c
    S = S + Z[16] * np.sqrt(12) * (10 * r5 - 12 * r3 + 3 * r) * s
    S = S + Z[17] * np.sqrt(12) * (5 * r5 - 4 * r3) * c3
    S = S + Z[18] * np.sqrt(12) * (5 * r5 - 4 * r3) * s3
    S = S + Z[19] * np.sqrt(12) * r5 * c5
    S = S + Z[20] * np.sqrt(12) * r5 * s5
    S = S + Z[21] * np.sqrt(7) * (20 * r6 - 30 * r4 + 12 * r2 - 1)
    S = S + Z[22] * np.sqrt(14) * (15 * r6 - 20 * r4 + 6 * r2) * s2
    S = S + Z[23] * np.sqrt(14) * (15 * r6 - 20 * r4 + 6 * r2) * c2
    S = S + Z[24] * np.sqrt(14) * (6 * r6 - 5 * r4) * s4
    S = S + Z[25] * np.sqrt(14) * (6 * r6 - 5 * r4) * c4
    S = S + Z[26] * np.sqrt(14) * r6 * s6
    S = S + Z[27] * np.sqrt(14) * r6 * c6
    
    return S


