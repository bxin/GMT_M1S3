import os
import sys
import h5py
import numpy as np
from scipy import interpolate
import scipy.io
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


saID_ml = np.loadtxt('saID_ml.txt')
saID_sw = np.loadtxt('saID_sw.txt')

print('## bending modes & influence matrices etc from Buddy #####################')
dataFolder = '/Users/bxin/Library/CloudStorage/OneDrive-SharedLibraries-GMTOCorp/M1S Portal - Documents'

#read SA data
dfSA = scipy.io.loadmat(dataFolder+'/M1 Testing/RFCML Optical Testing/bending modes/actCoords.mat')
sax_ml = dfSA['yAct']/1e3 #turn into meter #swap x/y to get to M1B (M1DCS uses M1B!!!)
say_ml = dfSA['xAct']/1e3 #turn into meter
print('ML actuators = ', len(sax_ml), len(say_ml))

#read Afz (Fz influence matrix)
df = scipy.io.loadmat(dataFolder+'/M1 Testing/RFCML Optical Testing/bending modes/influenceFunctions.mat')
Afn_ml = df['interactionMat']
fv_ml = df['forceMat'] #fv = fv^T
print('Afn = ',Afn_ml.shape)
print('fv = ', fv_ml.shape)
# this is Afz only; it is 6991 x 165.

#read Fz Bending Mode
mat = scipy.io.loadmat(dataFolder+'/M1 Testing/RFCML Optical Testing/bending modes/SVD.mat')
UMat_ml = mat['U']
SMat_ml = mat['S']
VMat_ml = mat['V']
print('U matrix', UMat_ml.shape)

#read FEA nodes data
mat = scipy.io.loadmat(dataFolder+'/M1 Testing/RFCML Optical Testing/bending modes/nodeCoords.mat')
nodex_ml = mat['y']/1e3 #turn into meter #swap x/y to get to M1B (M1DCS uses M1B!!!)
nodey_ml = mat['x']/1e3 #turn into meter
print('N node = ', len(nodex_ml))

############normalize bending modes to RMS = 1um ###################
UMat_ml *= np.sqrt(UMat_ml.shape[0])
for modeID in range(1, UMat_ml.shape[1]+1):
    VMat_ml[:, modeID-1] *= 1e3/SMat_ml[modeID-1, modeID-1]*np.sqrt(UMat_ml.shape[0])
    #1e3 due to nanometer to micron conversion; RFCML mode shapes are in nanometers

print('## bending modes & influence matrices etc from Trupti #####################')
dataFolder = '/Users/bxin/Library/CloudStorage/OneDrive-SharedLibraries-GMTOCorp/M1S Portal - Documents'

#read SA data
dfSA = pd.read_excel(dataFolder+'/2.4 Utilities/03. Utilities Distribution/utility_mapping_M1B_labels-16-Feb-2023.xlsx')
sax = np.array(dfSA['x_m']) #in M1B
say = np.array(dfSA['y_m'])
saz = np.array(dfSA['z_m'])
saID = np.array(dfSA['LSNo'])
saReqMaxFx_N = np.array(dfSA['ReqMaxFx_N']) #max Fx the SA is allowed to produce??
nact = len(dfSA)
print('N actuators = ', nact)

#read Afz (Fz influence matrix)
df = pd.read_csv(dataFolder+'/influnce_matrix_files/Afz-13-Apr-2023.csv', header=None)
Afz = np.array(df)
print('Afz = ',Afz.shape)
# this is Afz only; it is 27685 x 170.

#read Afx (Fx influence matrix)
df = pd.read_csv(dataFolder+'/influnce_matrix_files/Afx-24-Jul-2023.csv', header=None)
Afx = np.array(df)
print('Afx = ', Afx.shape)
#read Afy (Fy influence matrix)
df = pd.read_csv(dataFolder+'/influnce_matrix_files/Afy-24-Jul-2023.csv', header=None)
Afy = np.array(df)
print('Afy = ', Afy.shape)

#read Fz Bending Modes & forces
df = pd.read_csv(dataFolder+'/influnce_matrix_files/U-13-Apr-2023.csv', header=None)
UMat = np.array(df)
print('U matrix', UMat.shape)
df = pd.read_csv(dataFolder+'/influnce_matrix_files/V-13-Apr-2023.csv', header=None)
VMat = np.array(df)
print('V matrix', VMat.shape)
df = pd.read_csv(dataFolder+'/influnce_matrix_files/S-13-Apr-2023.csv', header=None)
SMat = np.array(df)
print('S matrix', SMat.shape)

#read FEA nodes data
mat = scipy.io.loadmat(dataFolder+'/influnce_matrix_files/NodeXYZsurface_meters.mat')
nodeID = mat['NodeXYZsurface_meters'][:,0]
nodex = mat['NodeXYZsurface_meters'][:,2] #swap x/y to get to M1B
nodey = mat['NodeXYZsurface_meters'][:,1]
nodez = mat['NodeXYZsurface_meters'][:,3]
print('N node = ', len(nodeID))

############normalize bending modes to RMS = 1um ###################
UMat *= np.sqrt(UMat.shape[0])
for modeID in range(1, UMat.shape[1]+1):
    VMat[:, modeID-1] *= 1e-6/SMat[modeID-1, modeID-1]*np.sqrt(UMat.shape[0])
    #1e-6 due to meter to micron conversion

npuck = np.zeros(nact)
for i in range(nact):
    if dfSA['LSActType'][i] == 0: #Single Axis on Single Puck
        npuck[i] = 1
    elif dfSA['LSActType'][i] == 1: #Single Axis on Dual Puck LS
        npuck[i] = 2
    elif dfSA['LSActType'][i] == 2: #Single Axis on triple puck LS
        npuck[i] = 3
    elif dfSA['LSActType'][i] == 30: #Single Axis on Single double triple LS puck
        npuck[i] = 1
    elif dfSA['LSActType'][i] == 31: #Single Axis on Single double triple LS bar
        npuck[i] = 2
    elif dfSA['LSActType'][i] == 40: #Triple Axis RH on triple puck LS
        npuck[i] = 3
    elif dfSA['LSActType'][i] == 41: #Triple Axis LH on triple puck LS
        npuck[i] = 3
    elif dfSA['LSActType'][i] == 5: #Triple Axis on quad puck LS
        npuck[i] = 2

def mlFvec2gmtFvec(mlFvec):
    '''
    convert a ML force vector (165x1) into a GMT force vector (170x1)
    input:
        mlFvec: ML force vector (165x1)
    output:
        GMT force vector (170x1)
    '''
    gmtFvec = np.zeros(nact)
    for i in range(nact):
        gmtFvec[i] = mlFvec[saID2mlModeID(saID[i])-1]
        if np.array(dfSA['LSActType'])[i]==5: #quad
            gmtFvec[i] /= 2.
    return gmtFvec

def saID2mlModeID(gmtsaID):
    '''
    convert GMT actuator ID to modeID in ML list.
    input:
        gmtsaID: integer (cannot be an array or list)
        This can be 170 different numbers: 101, 102, etc.
    output:
        ML mode ID. This number is in the range [1, 165]
    '''
    idx = np.where(saID_ml==gmtsaID)[0]
    idx1 = np.where(saID_ml%1e6==gmtsaID)[0]
    idx2 = np.where(saID_ml//1e6==gmtsaID)[0]
    if len(idx)==1:
        return idx[0]+1
    elif len(idx1) == 1:
        return idx1[0]+1
    elif len(idx2) == 1:
        return idx2[0]+1

def swFvec2gmtFvec(swFvec):
    '''
    convert Steve West's force vector (170x1) into a GMT force vector (170x1)
    input:
        swFvec: SW force vector (170x1)
    output:
        GMT force vector (170x1)
    '''
    gmtFvec = np.zeros(nact)
    for i in range(nact):
        gmtFvec[i] = swFvec[saID2swModeID(saID[i])-1]

    return gmtFvec

def saID2swModeID(gmtsaID):
    '''
    convert GMT actuator ID to modeID in Steve West's list.
    input:
        gmtsaID: integer (cannot be an array or list)
        This can be 170 different numbers: 101, 102, etc.
    output:
        SW mode ID. This number is in the range [1, 170]
    '''
    idx = np.where(saID_sw==gmtsaID)[0]
    return idx[0]+1

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


