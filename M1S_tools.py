import os
import sys
import h5py
import numpy as np
from scipy import interpolate
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
reversed_cmap = plt.cm.coolwarm.reversed()
from datetime import datetime
from datetime import timedelta

from os.path import expanduser

from pymongo import MongoClient
HOST = "localhost"
PORT = 27017

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

saID_ml = np.loadtxt('../model_data/saID_ml.txt')
saID_sw = np.loadtxt('../model_data/saID_sw.txt')

r_S1_center_in_S7 = 8.710 #distance to S1 optical axis in m
diameter_of_CA = 8.365 #CA diamter in m
radius_of_curvature = 36.000 #radius of curvature in m
conic = -0.998286
theta_M1S_deg = 13.601685
theta_M1B_deg = 13.522
radius_of_CA = diameter_of_CA/2.0
def surfFunc(r2):
    return r2/(radius_of_curvature+np.sqrt(radius_of_curvature**2-(1+conic)*r2))

lbs2N = 4.4482216153
in2mm = 25.4
N2kg = 0.10197
rad2arcsec = 180./np.pi*3600
arcsec2rad = 1./rad2arcsec
HP2XYZ_force = np.loadtxt('../model_data/HP2XYZ_force.txt')
XYZ2HP_position = np.loadtxt('../model_data/XYZ2HP_position.txt')
HP2XYZ_position = np.linalg.pinv(XYZ2HP_position)

rb_label = ['x','y','z','Rx','Ry','Rz']

print('## bending modes & influence matrices etc from Buddy #####################')

try:
    #read SA data
    dfSA = scipy.io.loadmat('../model_data/actCoords_ml.mat')
    sax_ml = dfSA['yAct']/1e3 #turn into meter #swap x/y to get to M1B (M1DCS uses M1B!!!)
    say_ml = dfSA['xAct']/1e3 #turn into meter
    print('ML actuators = ', len(sax_ml), len(say_ml))

    #read Afz (Fz influence matrix)
    df = scipy.io.loadmat('../model_data/influenceFunctions_ml.mat')
    Afn_ml = df['interactionMat']
    fv_ml = df['forceMat'] #fv = fv^T
    print('Afn = ',Afn_ml.shape)
    print('fv = ', fv_ml.shape)
    nact_ml = Afn_ml.shape[1]
    # this is Afz only; it is 6991 x 165.

    #read Fz Bending Mode
    mat = scipy.io.loadmat('../model_data/SVD_ml.mat')
    UMat_ml = mat['U']
    SMat_ml = mat['S']
    VMat_ml = mat['V']
    print('U matrix', UMat_ml.shape)

    #read FEA nodes data
    mat = scipy.io.loadmat('../model_data/nodeCoords_ml.mat')
    nodex_ml = mat['y']/1e3 #turn into meter #swap x/y to get to M1B (M1DCS uses M1B!!!)
    nodey_ml = mat['x']/1e3 #turn into meter
    print('N node = ', len(nodex_ml))

    ############normalize bending modes to RMS = 1um ###################
    UMat_ml *= np.sqrt(UMat_ml.shape[0])
    for modeID in range(1, UMat_ml.shape[1]+1):
        VMat_ml[:, modeID-1] *= 1e3/SMat_ml[modeID-1, modeID-1]*np.sqrt(UMat_ml.shape[0])
        #1e3 due to nanometer to micron conversion; RFCML mode shapes are in nanometers
except FileNotFoundError:
    print('***Data not exist. Are you sure they are there?***')
    
print('## bending modes & influence matrices etc from Trupti #####################')
dataFolder = '/Users/bxin/Library/CloudStorage/OneDrive-SharedLibraries-GMTOCorp/M1S Portal - Documents'

try:
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

    #for making plots, use Optical Coordinate System (ocs)
    sax_ocs = say
    say_ocs = sax
    
except FileNotFoundError:
    print('Data do not exist. Are you sure they are there?')
                
def mlFvec2gmtFvec(mlFvec):
    '''
    convert a ML force vector (165x1) into a GMT force vector (170x1)
    input:
        mlFvec: ML force vector (165x1)
        Buddy has 165 acts, 1 under each quad. the force is 2*F.
    output:
        GMT force vector (170x1)
        GMT has 170 acts, including 5 pairs under quad loadspreaders. 
        For example, one pair is 144 and 1144. They always have equal force, F.
    '''
    gmtFvec = np.zeros(nact)
    for i in range(nact):
        gmtFvec[i] = mlFvec[saID2mlModeID(saID[i])-1]
        if np.array(dfSA['LSActType'])[i]==5: #quad
            gmtFvec[i] /= 2.
    return gmtFvec

def gmtFvec2mlFvec(gmtFvec):
    '''
    convert a GMT force vector (170x1) into a ML force vector (165x1)
    input:
        gmtFvec: GMT force vector (170x1)
    output:
        mlFvec: ML force vector (165x1)
    '''
    mlFvec = np.zeros(nact_ml)
    for i in range(nact_ml):
        idx = np.where(saID==(saID_ml[i]))[0]
        if len(idx) == 1:
            mlFvec[i] = gmtFvec[idx]
        if saID_ml[i]>1e6:
            idx1 = np.where(saID==(saID_ml[i]%1e6))[0] #e.g. saID_ml can be 144001144
            idx2 = np.where(saID==(saID_ml[i]//1e6))[0]
            if len(idx1) == 1:
                mlFvec[i] += gmtFvec[idx1]
            if len(idx2) == 1:
                mlFvec[i] += gmtFvec[idx2]
    return mlFvec

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

def m1b_to_mlcs(m1b_vec):
    mlcs_vec = np.zeros_like(m1b_vec)
    if m1b_vec.ndim == 2:
        if m1b_vec.shape[1] >= 3: # e.g. (170,3)
            mlcs_vec[:,0] = m1b_vec[:,1] #m1b y is mlcs x
            mlcs_vec[:,1] = m1b_vec[:,0] #m1b x is mlcs y
            mlcs_vec[:,2] = -m1b_vec[:,2] #m1b z is mlcs -z
        if m1b_vec.shape[1] == 6: # e.g. (100,6)
            mlcs_vec[:,3] = m1b_vec[:,4] #m1b Ry is mlcs Rx
            mlcs_vec[:,4] = m1b_vec[:,3]  #m1b Rx is mlcs Ry
            mlcs_vec[:,5] = -m1b_vec[:,5] #m1b Rz is mlcs -Rz
    elif m1b_vec.ndim == 3 and m1b_vec.shape[2] == 3: # e.g. (100,170,3)
        mlcs_vec[:,:,0] = m1b_vec[:,:,1] #m1b y is mlcs x
        mlcs_vec[:,:,1] = m1b_vec[:,:,0] #m1b x is mlcs y
        mlcs_vec[:,:,2] = -m1b_vec[:,:,2] #m1b z is mlcs -z
    elif m1b_vec.ndim == 1: #e.g. (170,)
        mlcs_vec = -m1b_vec
    else:
        raise TypeError(f"Unknown data type with dimension: {m1b_vec.shape}.")
    return mlcs_vec
def mlcs_to_m1b(mlcs_vec):
    '''
    m1cs_to_m1b is the reverse of m1b_to_mlcs
    '''
    return m1b_to_mlcs(mlcs_vec)

def plotRB(mirror_z, length_unit='mm', angle_unit='arcsec'):
    '''
        input mirror position as a Tx6 numpy array. T is the number of time samples
    '''
    fig, ax = plt.subplots(1,2,figsize=(15, 4))
    if length_unit == 'mm':
        f=1e3
    elif length_unit == 'um':
        f=1e6
    elif length_unit == 'nm':
        f=1e9
    for i in [0,1,2]:
        ax[0].plot(tt-tt[0], mirror_z[:,i]*f, '-o', label=rb_label[i]);
    ax[0].set_xlabel('time (s)')
    ax[0].set_title('Translations (%s) in Optical Coordinate System'%length_unit);
    #plt.gca().invert_yaxis();
    ax[0].legend()
    ax[0].grid();

    if angle_unit == 'arcsec':
        f=180/np.pi*3600
    elif length_unit == 'mrad':
        f=1e3
    elif length_unit == 'deg':
        f=180/np.pi
    for i in [3,4,5]:
        ax[1].plot(tt-tt[0], mirror_z[:,i]*f, '-o', label=rb_label[i]);
    ax[1].set_xlabel('time (s)')
    ax[1].set_title('Rotations (%s) in Optical Coordinate System'%angle_unit);
    #plt.gca().invert_yaxis();
    ax[1].legend()
    ax[1].grid();

def showSurfMap(m1s, m3s, x1, y1, x3, y3):
    '''
    takes the m1 and m3 surfaces, interpolate m3 onto m1 grid, so that we can display then as one plot.
    '''
    print('input forces and output figure both in Optical Coordinate System (OCS)')
    s = m1s
    r1 = np.sqrt(x1**2 + y1**2)
    idx = (r1<m3ORC)*(r1>m3IRC)
    m3s[np.isnan(m3s)] = 0
    f = interpolate.interp2d(x3[0,:], y3[:,0], m3s)
    s_temp = f(x1[0,:], y1[:,0])
    s[idx] = s_temp[idx]
    return s

def showForceMap(forces, figure_title):
    '''
    input: 
        forces should already be in optical CS (OCS)
    output:
        force map displayed in optical CS (OCS)
        
    '''
    print('input forces and output figure both in Optical Coordinate System (OCS)')
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    plt.scatter(sax_ocs, say_ocs, c=forces) #, cmap=reversed_cmap)
    #plt.scatter(sax_ml, say_ml, s=100, facecolors='none', edgecolors='k')
    for i in range(len(sax)):
        if (np.any(abs(sax_ocs[i]+say_ocs[i]-sax_ocs[:i]-say_ocs[:i])<1e-4)):
            plt.text(sax_ocs[i]+.05, say_ocs[i]-0.15, '%.0f'%forces[i],color='r',fontsize=8)
        else:
            plt.text(sax_ocs[i]+.05, say_ocs[i]+.05, '%.0f'%forces[i],color='r',fontsize=8)
    plt.axis('equal')
    plt.xlabel('x in meter')
    plt.ylabel('y in meter')
    plt.colorbar()
    plt.title(figure_title)

def showForceMap_M1B(forces, figure_title):
    '''
    input: 
        forces should already be in M1B
    output:
        force map displayed in M1B
        
    '''
    print('input forces and output figure both in M1B')
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    plt.scatter(sax, say, c=forces, cmap=reversed_cmap)
    #plt.scatter(sax_ml, say_ml, s=100, facecolors='none', edgecolors='k')
    for i in range(len(sax)):
        if (np.any(abs(sax[i]+say[i]-sax[:i]-say[:i])<1e-4)):
            plt.text(sax[i]+.05, say[i]-0.15, '%.0f'%forces[i],color='r',fontsize=8)
        else:
            plt.text(sax[i]+.05, say[i]+.05, '%.0f'%forces[i],color='r',fontsize=8)
    plt.axis('equal')
    plt.xlabel('x in meter')
    plt.ylabel('y in meter')
    plt.colorbar()
    plt.title(figure_title)
    
client = MongoClient(HOST, PORT)
tele = client.gmt_tele_1.tele_events

def printDBVar(myt, table_name, duration_in_s=1):
    '''
    print out values of table_name in the duration following myt timestamp (to the minute)
    '''
    print(table_name)
    try:
        [month, day, hour, minute] = myt
        t0 = float(datetime(2024, month, day, hour, minute, 0).strftime('%s'))
    except TypeError:
        t0 = myt
        print(datetime.fromtimestamp(myt).strftime('%Y-%m-%d %H:%M:%S'))
    start_time = t0
    end_time = t0+duration_in_s

    records = tele.find({"ts":{"$gt":start_time*1000000000.0,"$lt":end_time*1000000000.0},
                     "src":{"$eq":f"{table_name}"}})
    for record in records:
        print(record['value'])
        
def getDBData(myt, table_name, duration_in_s=60, samples=60):
    '''
    return a numpy array containing the forces in the duration following myt timestamp (to the minute)
        first dimension: sampling time (1-samples)
        second dimension: actuator (1-170)
        third dimension: x, y, or z
    '''
    print(table_name)
    try:
        [month, day, hour, minute] = myt
        t0 = float(datetime(2024, month, day, hour, minute, 0).strftime('%s'))
        print(' duration = ', duration_in_s, ' s')
    except TypeError:
        t0 = myt
        print(datetime.fromtimestamp(myt).strftime('%Y-%m-%d %H:%M:%S'), ' duration = ', duration_in_s, ' s')
    start_time = t0
    end_time = t0+duration_in_s

    records = tele.find({"ts":{"$gt":start_time*1000000000.0,"$lt":end_time*1000000000.0},
                     "src":{"$eq":f"{table_name}"}})
    force_data = []
    ts_data = []
    desired_interval = duration_in_s/samples
    record_clock = 0
    for record in records:
        record_clock += 0.1
        if record_clock > desired_interval-0.01:
            force_data.append(record['value'])
            ts_data.append(record["ts"]/ 1000000000.0)
            #print(record_clock)
            record_clock = 0
        #print()
    print(np.array(force_data).shape)
    if 'dewpoint' in table_name:
        #no sign change
        aa = np.array(force_data) - 273.15
    else:
        #we convert all data to be consistent with RFCML surface maps (so that our heads won't be spinning!)
        aa = m1b_to_mlcs(np.array(force_data))
    return aa, np.array(ts_data)

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


