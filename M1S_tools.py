import os
import sys
import h5py
import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata

import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
reversed_cmap = plt.cm.coolwarm.reversed()
from datetime import datetime
from datetime import timedelta
import glob
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
ml_data_dir = "/Users/bxin/GMT_docs/1_M1/Analysis/ml_data/data/Optical Data/"

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

    TRDate = '09Jan2025'

    TRIFFolder = '/influnce_matrix_files/OA_influence_matrices_all/OA_surface_normal_*_%s/'%TRDate
    TRIFFolder = glob.glob(dataFolder+TRIFFolder)[0]
    print(TRIFFolder)
    Dec24IFFolder = '/influnce_matrix_files/OA_influence_matrices_all/OA_surface_normal_normalised_20Dec2024/'
    #read Afz (Fz influence matrix) - 01/02/2025. Trupti confirmed on slack that this is surface normal
    df = pd.read_csv(TRIFFolder+'Afz-nohp-%s-%s-%s.csv'%(TRDate[:2],TRDate[2:5],TRDate[5:]), header=None)
    Afz = np.array(df)
    print('Afz = ',Afz.shape)
    # this is Afz only; it is 27547 x 165.

    #read Afx (Fx influence matrix)
    df = pd.read_csv(TRIFFolder+'Afz-nohp-%s-%s-%s.csv'%(TRDate[:2],TRDate[2:5],TRDate[5:]), header=None)
    Afx = np.array(df)
    print('Afx = ', Afx.shape)

    #read Afy (Fy influence matrix)
    df = pd.read_csv(TRIFFolder+'Afz-nohp-%s-%s-%s.csv'%(TRDate[:2],TRDate[2:5],TRDate[5:]), header=None)
    Afy = np.array(df)
    print('Afy = ', Afy.shape)

    #read Fz Bending Modes & forces (note: Uzm is for the moments.)
    df = pd.read_csv(TRIFFolder+'Uz_norm-nohp-%s-%s-%s.csv'%(TRDate[:2],TRDate[2:5],TRDate[5:]), header=None)
    UMat = np.array(df)
    print('U matrix', UMat.shape)

    df = pd.read_csv(TRIFFolder+'Sz_norm-nohp-%s-%s-%s.csv'%(TRDate[:2],TRDate[2:5],TRDate[5:]), header=None)
    SMat = np.array(df)
    print('S matrix', SMat.shape)

    df = pd.read_csv(TRIFFolder+'Vz_norm-nohp-%s-%s-%s.csv'%(TRDate[:2],TRDate[2:5],TRDate[5:]), header=None)
    VMat = np.array(df)
    print('V matrix', VMat.shape)    
    
    #read FEA nodes data
    df = pd.read_csv(dataFolder+Dec24IFFolder+'surfacenodes_M1B-20-Dec-2024.csv')
    nodeID = np.array(df['nodeID'])
    nodex = np.array(df['X'])
    nodey = np.array(df['Y'])
    nodez = np.array(df['Z'])
    print('N node = ', len(nodeID))

    # use these indices to remove surface nodes outside of CA
    noder = np.sqrt(nodex**2+nodey**2)
    insideCA = noder< np.max(nodex_ml) #diameter_of_CA/2.0
    nodeID = nodeID[insideCA]
    nodex = nodex[insideCA]
    nodey = nodey[insideCA]
    nodez = nodez[insideCA]

    ############normalize bending modes to RMS = 1um ###################
    #UMat is already normalized to RMS of 1.
    VMat = np.linalg.pinv(SMat@VMat.T/VMat.shape[0])*1e-6

    #make a Afz with 170 columns to avoid breaking previous code
    Afz170 = np.zeros((Afz.shape[0], nact))
    Afz170[:,:165] = Afz
    Afz170[:,165:] = Afz[:,160:]

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

def mlFvec2gmt165Fvec(mlFvec):
    '''
    convert a ML force vector (165x1) into a GMT force vector (165x1)
    input:
        mlFvec: ML force vector (165x1)
        Buddy has 165 acts, 1 under each quad. the force is 2*F.
    output:
        GMT force vector (170x1)
        GMT has 170 acts, including 5 pairs under quad loadspreaders. 
        For example, one pair is 144 and 1144. They always have equal force, F.
    '''
    gmtFvec = np.zeros(165)
    for i in range(165):
        gmtFvec[i] = mlFvec[saID2mlModeID(saID[i])-1]
        #if np.array(dfSA['LSActType'])[i]==5: #quad
        #    gmtFvec[i] /= 2.
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

def gmt165Fvec2mlFvec(gmtFvec):
    '''
    convert a GMT force vector (165x1) into a ML force vector (165x1)
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
            idx1 = np.where(saID==(saID_ml[i]%1e6))[0] #e.g. saID_ml can be 144001144, this gives index for SA1144
            idx2 = np.where(saID==(saID_ml[i]//1e6))[0] #this gives index for SA144
            #if len(idx1) == 1:
            #    mlFvec[i] += gmtFvec[idx1]
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

def readH5Map(fileset, dataset = '/dataset', verbose = True):
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
        if verbose:
            print('%s: %s '%(filenameShort, timeStamp))
        i+=1
    data /= i
    data = np.rot90(data, 1) # so that we can use imshow(data, origin='lower')
    return data, centerRow, centerCol, pixelSize, timeStamp

Sxn = 853
Syn = 853
def getH5date(h5file):
    f = h5py.File(h5file,'r')
    data0 = f[dataset]
    if 'date' in data0.attrs.keys():
        if len(data0.attrs['date']) == 1:
            timeStamp = data0.attrs['date'][0].decode('ascii')
        else:
            timeStamp = data0.attrs['date'].decode('ascii')
    else:
        timeStamp = 'date not in h5 file.'    
    datetime_obj = datetime.strptime(timeStamp, "%a %b %d %H:%M:%S %Y")
    unix_timestamp = int(datetime_obj.timestamp())
    return unix_timestamp

def writeH5map(map_file, map_data, dataset = '/dataset'):
    with h5py.File(map_file, 'w') as h5f:
        # Create the dataset in the default folder '/dataset'
        #why the transpose and fliplr? 
        #this is the way to write it in a way that is consistent with the read out (readH5Map)
        #the way to check it is to plot it, then save it, read it out, and compare
        dataset = h5f.create_dataset(dataset, data=np.fliplr(map_data.T))
        
        # Add attributes
        dataset.attrs['centerRow'] = centerRow
        dataset.attrs['centerCol'] = centerCol
        dataset.attrs['pixelSize'] = pixelSize
    
def unix_ts(h5string):
    datetime_obj = datetime.strptime(h5string, "%a %b %d %H:%M:%S %Y")
    unix_timestamp = int(datetime_obj.timestamp())
    return unix_timestamp

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

def saID2pixx(mysaID, centerRow, centerCol, pixelSize):
    '''
    input: GMT SA ID, e.g., 101
    output: pixel x location on ML surface map (after transformed into GMT M1B)
    '''
    return centerCol+sax[saID==mysaID]/pixelSize

def saID2pixy(mysaID, centerRow, centerCol, pixelSize):
    '''
    input: GMT SA ID, e.g., 101
    output: pixel x location on ML surface map (after transformed into GMT M1B)
    '''
    return centerRow+say[saID==mysaID]/pixelSize

def showTMaps(tss):
    #timestamps (tss) example: 
    #["Fri Jan 10 14:04:57 2025", "Fri Jan 10 14:14:57 2025", "Fri Jan 10 15:20:57 2025", "Fri Jan 10 15:30:57 2025"]

    # Define the grid for interpolation
    grid_size = 500  # Resolution of the grid
    xi = np.linspace(-radius_of_CA, radius_of_CA, grid_size)
    yi = np.linspace(-radius_of_CA, radius_of_CA, grid_size)
    xi, yi = np.meshgrid(xi, yi)

    # Create a circular mask for the given radius
    mask = np.sqrt(xi**2 + yi**2) <= radius_of_CA

    # Plot the result
    fig, ax = plt.subplots(len(tss),3,figsize=(18,4*len(tss)))
    for i,ts in enumerate(tss):
        print('----------------  ', ts)#, unix_ts(ts))
        tc, tt = getDBData(unix_ts(ts),'m1_s1_thermal_ctrl/i/tc_temperature/value', duration_in_s=100, samples=1)
        #back plate
        x = tc_locs[idx_mirror_b][:,0] 
        y = tc_locs[idx_mirror_b][:,1]
        z = tc[0,idx_mirror_b]
        # Interpolate z-values to the grid
        zi = griddata((x, y), z, (xi, yi), method='linear')
        zi[~mask] = np.nan  # Set values outside the circle to NaN
        
        contour = ax[i][0].contourf(xi, yi, zi, levels=100, cmap='jet')
        fig.colorbar(contour, ax=ax[i][0])#, label='')
        ax[i][0].scatter(x, y, c=z, edgecolor='k', cmap='jet', label='TCs')
        ax[i][0].set_aspect('equal', adjustable='box')
        ax[i][0].set_title('Back, PV = %.2f K'%(np.max(z)-np.min(z)))
        #ax[i][0].set_xlabel('X (m)')
        ax[i][0].set_ylabel('Y (m)')
        ax[i][0].legend()
        
        #levels = np.linspace(np.min(z), np.max(z), 100)
        #front plate
        x = tc_locs[idx_mirror_f][:,0] 
        y = tc_locs[idx_mirror_f][:,1]
        z = tc[0,idx_mirror_f]
        zi = griddata((x, y), z, (xi, yi), method='linear')
        zi[~mask] = np.nan  # Set values outside the circle to NaN
        
        contour = ax[i][1].contourf(xi, yi, zi, levels=100, cmap='jet')
        fig.colorbar(contour, ax=ax[i][1])#, label='Z values')
        ax[i][1].scatter(x, y, c=z, edgecolor='k', cmap='jet', label='TCs')
        ax[i][1].set_aspect('equal', adjustable='box')
        ax[i][1].set_title('Front, PV = %.2f K'%(np.max(z)-np.min(z)))
        #ax[i][0].set_xlabel('X (m)')
        ax[i][1].set_ylabel('Y (m)')
        ax[i][1].legend()
        
        #back-front
        z = tc[0,idx_mirror_b] - tc[0,idx_mirror_f]
        zi = griddata((x, y), z, (xi, yi), method='linear')
        zi[~mask] = np.nan  # Set values outside the circle to NaN
        
        contour = ax[i][2].contourf(xi, yi, zi, levels=100, cmap='jet')
        fig.colorbar(contour, ax=ax[i][2])#, label='Z values')
        ax[i][2].scatter(x, y, c=z, edgecolor='k', cmap='jet', label='TCs')
        ax[i][2].set_aspect('equal', adjustable='box')
        ax[i][2].set_title('Back-Front, PV = %.2f K'%(np.max(z)-np.min(z)))
        #ax[i][0].set_xlabel('X (m)')
        ax[i][2].set_ylabel('Y (m)')
        ax[i][2].legend()
    plt.show()
    
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
    n_interval = 0
    for record in records:
        ts_s = record["ts"]/ 1000000000.0
        if len(ts_data) == 0:
            ts_data.append(ts_s)
            force_data.append(record['value'])
            n_interval +=1
        else:
            if ts_s - ts_data[0]> n_interval*desired_interval-0.01:
                ts_data.append(ts_s)
                force_data.append(record['value'])
                n_interval +=1
    print(np.array(force_data).shape)
    if 'dewpoint' in table_name:
        #no sign change
        aa = np.array(force_data) - 273.15
    elif 'mirror_temperature' in table_name:
        aa = np.array(force_data) - 273.15
    elif 'ambient_temperature' in table_name:
        aa = np.array(force_data) - 273.15
    elif 'tc_temperature' in table_name:
        aa = np.array(force_data) - 273.15
        aa = aa.reshape((-1, aa.shape[1]* aa.shape[2])) #n_time x 192
    else:
        #we convert all data to be consistent with RFCML surface maps (so that our heads won't be spinning!)
        aa = m1b_to_mlcs(np.array(force_data))
    print(aa.shape)
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

#https://github.com/CanisUrsa/ocs_m1_dcs/blob/master/src/etc/conf/m1_thermal_pkg/common/m1_s3_tc_label_conf.coffee
tc_labels = [
    [ # TC Scanner 1
        "TC1REF",    # Channel 0 - REF
        "MTCIN002B", # Channel 1 - MTCIN002B - Mirror
        "MTCIN002M", # Channel 2 - MTCIN002M - Mirror
        "MTCIN002F", # Channel 3 - MTCIN002F - Mirror
        "MTC013B",   # Channel 4 - MTC013B - Mirror
        "MTC013F",   # Channel 5 - MTC013F - Mirror
        "MTC014B",   # Channel 6 - MTC014B - Mirror
        "MTC014F",   # Channel 7 - MTC014F - Mirror
        "MTC015B",   # Channel 8 - MTC015B - Mirror
        "MTC015F",   # Channel 9 - MTC015F - Mirror
        "MTCOW005B", # Channel 10 - MTCOW005B - Mirror
        "MTCOW005M", # Channel 11 - MTCOW005M - Mirror
        "MTCOW005F", # Channel 12 - MTCOW005F - Mirror
        "MTC016B",   # Channel 13 - MTC016B - Mirror
        "MTC016F",   # Channel 14 - MTC016F - Mirror
        "MTCOW006B", # Channel 15 - MTCOW006B - Mirror
        "MTCOW006M", # Channel 16 - MTCOW006M - Mirror
        "MTCOW006F", # Channel 17 - MTCOW006F - Mirror
        "CTCUP001",  # Channel 18 - CTCUP001 - Upper Plenum
        "CTCUP002",  # Channel 19 - CTCUP002 - Upper Plenum
        "CTCUP003",  # Channel 20 - CTCUP003 - Upper Plenum
        "CTCLP001",  # Channel 21 - CTCLP001 - Lower Plenum
        "CTCCW001T", # Channel 22 - CTCCW001T - Weldment
        "",          # Channel 23 - NC
        "",          # Channel 24 - NC
        "",          # Channel 25 - NC
        "",          # Channel 26 - NC
        "",          # Channel 27 - NC
        "",          # Channel 28 - NC
        "",          # Channel 29 - NC
        "",          # Channel 30 - NC
        "CTCIB001",  # Channel 31 - CTCIB001 - ITJB
    ],
    [ # TC Scanner 2
        "TC2REF",    # Channel 0 - REF
        "MTC017B",   # Channel 1 - MTC017B - Mirror
        "MTC017F",   # Channel 2 - MTC017F - Mirror
        "MTC018B",   # Channel 3 - MTC018B - Mirror
        "MTC018F",   # Channel 4 - MTC018F - Mirror
        "MTCIN003B", # Channel 5 - MTCIN003B - Mirror
        "MTCIN003M", # Channel 6 - MTCIN003M - Mirror
        "MTCIN003F", # Channel 7 - MTCIN003F - Mirror
        "MTC019B",   # Channel 8 - MTC019B - Mirror
        "MTC019F",   # Channel 9 - MTC019F - Mirror
        "MTC020B",   # Channel 10 - MTC020B - Mirror
        "MTC020F",   # Channel 11 - MTC020F - Mirror
        "MTC021B",   # Channel 12 - MTC021B - Mirror
        "MTC021F",   # Channel 13 - MTC021F - Mirror
        "MTCOW007B", # Channel 14 - MTCOW007B - Mirror
        "MTCOW007M", # Channel 15 - MTCOW007M - Mirror
        "MTCOW007F", # Channel 16 - MTCOW007F - Mirror
        "MTC022B",   # Channel 17 - MTC022B - Mirror
        "MTC022F",   # Channel 18 - MTC022F - Mirror
        "CTCUP004",  # Channel 19 - CTCUP004 - Upper Plenum
        "CTCUP005",  # Channel 20 - CTCUP005 - Upper Plenum
        "CTCUP006",  # Channel 21 - CTCUP006 - Upper Plenum
        "CTCAA001",  # Channel 22 - CTCAA001 - Ambient
        "CTCCW001W", # Channel 23 - CTCCW001W - Weldment
        "CTCCW001F", # Channel 24 - CTCCW001F - Weldment
        "",          # Channel 25 - NC
        "",          # Channel 26 - NC
        "",          # Channel 27 - NC
        "",          # Channel 28 - NC
        "",          # Channel 29 - NC
        "",          # Channel 30 - NC
        "CTCIB002",  # Channel 31 - CTCIB002 - ITJB
    ],
    [ # TC Scanner 3
        "TC3REF",    # Channel 0 - REF
        "MTCOW008B", # Channel 1 - MTCOW008B - Mirror
        "MTCOW008M", # Channel 2 - MTCOW008M - Mirror
        "MTCOW008F", # Channel 3 - MTCOW008F - Mirror
        "MTC023B",   # Channel 4 - MTC023B - Mirror
        "MTC023F",   # Channel 5 - MTC023F - Mirror
        "MTC024B",   # Channel 6 - MTC024B - Mirror
        "MTC024F",   # Channel 7 - MTC024F - Mirror
        "MTC025B",   # Channel 8 - MTC025B - Mirror
        "MTC025F",   # Channel 9 - MTC025F - Mirror
        "MTC026B",   # Channel 10 - MTC026B - Mirror
        "MTC026F",   # Channel 11 - MTC026F - Mirror
        "MTCOW009B", # Channel 12 - MTCOW009B - Mirror
        "MTCOW009M", # Channel 13 - MTCOW009M - Mirror
        "MTCOW009F", # Channel 14 - MTCOW009F - Mirror
        "MTC027B",   # Channel 15 - MTC027B - Mirror
        "MTC027F",   # Channel 16 - MTC027F - Mirror
        "MTC028B",   # Channel 17 - MTC028B - Mirror
        "MTC028F",   # Channel 18 - MTC028F - Mirror
        "MTCOW010B", # Channel 19 - MTCOW010B - Mirror
        "MTCOW010M", # Channel 20 - MTCOW010M - Mirror
        "MTCOW010F", # Channel 21 - MTCOW010F - Mirror
        "CTCUP007",  # Channel 22 - CTCUP007 - Upper Plenum
        "CTCUP008",  # Channel 23 - CTCUP008 - Upper Plenum
        "CTCLP002",  # Channel 24 - CTCLP002 - Lower Plenum
        "CTCCW002T", # Channel 25 - CTCCW002T - Weldment
        "",          # Channel 26 - NC
        "",          # Channel 27 - NC
        "",          # Channel 28 - NC
        "",          # Channel 29 - NC
        "",          # Channel 30 - NC
        "CTCIB003",  # Channel 31 - CTCIB003 - ITJB
    ],
    [ # TC Scanner 4
        "TC4REF",    # Channel 0 - REF
        "MTCIN004B", # Channel 1 - MTCIN004B - Mirror
        "MTCIN004M", # Channel 2 - MTCIN004M - Mirror
        "MTCIN004F", # Channel 3 - MTCIN004F - Mirror
        "MTC029B",   # Channel 4 - MTC029B - Mirror
        "MTC029F",   # Channel 5 - MTC029F - Mirror
        "MTC030B",   # Channel 6 - MTC030B - Mirror
        "MTC030F",   # Channel 7 - MTC030F - Mirror
        "MTC031B",   # Channel 8 - MTC031B - Mirror
        "MTC031F",   # Channel 9 - MTC031F - Mirror
        "MTCOW011B", # Channel 10 - MTCOW011B - Mirror
        "MTCOW011M", # Channel 11 - MTCOW011M - Mirror
        "MTCOW011F", # Channel 12 - MTCOW011F - Mirror
        "MTC032B",   # Channel 13 - MTC032B - Mirror
        "MTC032F",   # Channel 14 - MTC032F - Mirror
        "MTCOW012B", # Channel 15 - MTCOW012B - Mirror
        "MTCOW012M", # Channel 16 - MTCOW012M - Mirror
        "MTCOW012F", # Channel 17 - MTCOW012F - Mirror
        "CTCUP009",  # Channel 18 - CTCUP009 - Upper Plenum
        "CTCUP010",  # Channel 19 - CTCUP010 - Upper Plenum
        "CTCAA002",  # Channel 20 - CTCAA002 - Ambient
        "CTCCW002W", # Channel 21 - CTCCW002W - Weldment
        "CTCCW002F", # Channel 22 - CTCCW002F - Weldment
        "",          # Channel 23 - NC
        "",          # Channel 24 - NC
        "",          # Channel 25 - NC
        "",          # Channel 26 - NC
        "",          # Channel 27 - NC
        "",          # Channel 28 - NC
        "",          # Channel 29 - NC
        "",          # Channel 30 - NC
        "CTCIB004",  # Channel 31 - CTCIB004 - ITJB
    ],
    [ # TC Scanner 5
        "TC5REF",    # Channel 0 - REF
        "MTC001B",   # Channel 1 - MTC001B - Mirror
        "MTC001F",   # Channel 2 - MTC001F - Mirror
        "MTC002B",   # Channel 3 - MTC002B - Mirror
        "MTC002F",   # Channel 4 - MTC002F - Mirror
        "MTCIN001B", # Channel 5 - MTCIN001B - Mirror
        "MTCIN001M", # Channel 6 - MTCIN001M - Mirror
        "MTCIN001F", # Channel 7 - MTCIN001F - Mirror
        "MTC003B",   # Channel 8 - MTC003B - Mirror
        "MTC003F",   # Channel 9 - MTC003F - Mirror
        "MTC004B",   # Channel 10 - MTC004B - Mirror
        "MTC004F",   # Channel 11 - MTC004F - Mirror
        "MTC005B",   # Channel 12 - MTC005B - Mirror
        "MTC005F",   # Channel 13 - MTC005F - Mirror
        "MTCOW001B", # Channel 14 - MTCOW001B - Mirror
        "MTCOW001M", # Channel 15 - MTCOW001M - Mirror
        "MTCOW001F", # Channel 16 - MTCOW001F - Mirror
        "MTC006B",   # Channel 17 - MTC006B - Mirror
        "MTC006F",   # Channel 18 - MTC006F - Mirror
        "CTCUP013",  # Channel 19 - CTCUP013 - Upper Plenum
        "CTCUP014",  # Channel 20 - CTCUP014 - Upper Plenum
        "CTCLP003",  # Channel 21 - CTCLP003 - Lower Plenum
        "CTCCW003T", # Channel 22 - CTCCW003T - Weldment
        "",          # Channel 23 - NC
        "",          # Channel 24 - NC
        "",          # Channel 25 - NC
        "",          # Channel 26 - NC
        "",          # Channel 27 - NC
        "",          # Channel 28 - NC
        "",          # Channel 29 - NC
        "",          # Channel 30 - NC
        "CTCIB005",  # Channel 31 - CTCIB005 - ITJB
    ],
    [ # TC Scanner 6
        "TC6REF",    # Channel 0 - REF
        "MTCOW002B", # Channel 1 - MTCOW002B - Mirror
        "MTCOW002M", # Channel 2 - MTCOW002M - Mirror
        "MTCOW002F", # Channel 3 - MTCOW002F - Mirror
        "MTC007B",   # Channel 4 - MTC007B - Mirror
        "MTC007F",   # Channel 5 - MTC007F - Mirror
        "MTC008B",   # Channel 6 - MTC008B - Mirror
        "MTC008F",   # Channel 7 - MTC008F - Mirror
        "MTC009B",   # Channel 8 - MTC009B - Mirror
        "MTC009F",   # Channel 9 - MTC009F - Mirror
        "MTC010B",   # Channel 10 - MTC010B - Mirror
        "MTC010F",   # Channel 11 - MTC010F - Mirror
        "MTCOW003B", # Channel 12 - MTCOW003B - Mirror
        "MTCOW003M", # Channel 13 - MTCOW003M - Mirror
        "MTCOW003F", # Channel 14 - MTCOW003F - Mirror
        "MTC011B",   # Channel 15 - MTC011B - Mirror
        "MTC011F",   # Channel 16 - MTC011F - Mirror
        "MTC012B",   # Channel 17 - MTC012B - Mirror
        "MTC012F",   # Channel 18 - MTC012F - Mirror
        "MTCOW004B", # Channel 19 - MTCOW004B - Mirror
        "MTCOW004M", # Channel 20 - MTCOW004M - Mirror
        "MTCOW004F", # Channel 21 - MTCOW004F - Mirror
        "CTCUP011",  # Channel 22 - CTCUP011 - Upper Plenum
        "CTCUP012",  # Channel 23 - CTCUP012 - Upper Plenum
        "CTCAA003",  # Channel 24 - CTCAA003 - Ambient
        "CTCCW003W", # Channel 25 - CTCCW003W - Weldment
        "CTCCW003F", # Channel 26 - CTCCW003F - Weldment
        "",          # Channel 27 - NC
        "",          # Channel 28 - NC
        "",          # Channel 29 - NC
        "",          # Channel 30 - NC
        "CTCIB006",  # Channel 31 - CTCIB006 - ITJB
    ]
]
tc_labels = [item for sublist in tc_labels for item in sublist]

#https://github.com/CanisUrsa/ocs_m1_dcs/blob/master/src/etc/conf/m1_thermal_pkg/common/m1_s3_tc_position_conf.coffee
tc_locs = [
    [ # TC Scanner 1
        [ 0, 0, 0 ],                                   # Channel 0 - CTCIB001 - ITJB
        [ -0.332903464 ,-0.474509544 ,-0.03175 ],      # Channel 1 - MTCIN002B - Mirror
        [ -0.332903464 ,-0.474509544 ,-0.236886226 ],  # Channel 2 - MTCIN002M - Mirror
        [ -0.332892 ,-0.384404 ,-0.442022451 ],        # Channel 3 - MTCIN002F - Mirror
        [ -1.742551113 ,-1.198263494 ,-0.03175 ],      # Channel 4 - MTC013B - Mirror
        [ -1.664538 ,-1.153211 ,-0.492287168 ],        # Channel 5 - MTC013F - Mirror
        [ -2.908297565 ,-2.482149258 ,-0.03175 ],      # Channel 6 - MTC014B - Mirror
        [ -2.97942 ,-2.537714 ,-0.640703237 ],         # Channel 7 - MTC014F - Mirror
        [ -1.742552378 ,-2.543673903 ,-0.03175 ],      # Channel 8 - MTC015B - Mirror
        [ -1.664462 ,-2.498623 ,-0.554477486 ],        # Channel 9 - MTC015F - Mirror
        [ -2.086749319 ,-3.6143556 ,-0.03175 ],        # Channel 10 - MTCOW005B - Mirror
        [ -2.086749319 ,-3.6143556 ,-0.343210699 ],    # Channel 11 - MTCOW005M - Mirror
        [ -2.041652 ,-3.536086 ,-0.654677144 ],        # Channel 12 - MTCOW005F - Mirror
        [ -0.66580512 ,-3.549738657 ,-0.03175 ],       # Channel 13 - MTC016B - Mirror
        [ -0.665734 ,-3.459632 ,-0.596353309 ],        # Channel 14 - MTC016F - Mirror
        [ -0.095 ,-4.172416979 ,-0.03175 ],            # Channel 15 - MTCOW006B - Mirror
        [ -0.095002175 ,-4.172416979 ,-0.343210699 ],  # Channel 16 - MTCOW006M - Mirror
        [ -0.092939 ,-4.082059 ,-0.651491443 ],        # Channel 17 - MTCOW006F - Mirror
        [ 0, 0, 0 ],                                   # Channel 18 - CTCUP001 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 19 - CTCUP002 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 20 - CTCUP003 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 21 - CTCLP001 - Lower Plenum
        [ 0, 0, 0 ],                                   # Channel 22 - CTCCW001T - Weldment
        [ 0, 0, 0 ],                                   # Channel 23 - NC
        [ 0, 0, 0 ],                                   # Channel 24 - NC
        [ 0, 0, 0 ],                                   # Channel 25 - NC
        [ 0, 0, 0 ],                                   # Channel 26 - NC
        [ 0, 0, 0 ],                                   # Channel 27 - NC
        [ 0, 0, 0 ],                                   # Channel 28 - NC
        [ 0, 0, 0 ],                                   # Channel 29 - NC
        [ 0, 0, 0 ],                                   # Channel 30 - NC
        [ 0, 0, 0 ],                                   # Channel 31 - NC
    ],
    [ # TC Scanner 2
        [ 0, 0, 0 ],                                   # Channel 0 - CTCIB002 - ITJB
        [ 0 ,-1.435518857 ,-0.03175 ],                 # Channel 1 - MTC017B - Mirror
        [ 0 ,-1.345413 ,-0.460566185 ],                # Channel 2 - MTC017F - Mirror
        [ 0 ,-2.588729657 ,-0.03175 ],                 # Channel 3 - MTC018B - Mirror
        [ 0 ,-2.498623 ,-0.516516625 ],                # Channel 4 - MTC018F - Mirror
        [ 0.332903464 ,-0.474509544 ,-0.03175 ],       # Channel 5 - MTCIN003B - Mirror
        [ 0.332903464 ,-0.474509544 ,-0.236886226 ],   # Channel 6 - MTCIN003M - Mirror
        [ 0.332892 ,-0.384404 ,-0.442022451 ],         # Channel 7 - MTCIN003F - Mirror
        [ 0.66580512 ,-3.549738657 ,-0.03175 ],        # Channel 8 - MTC019B - Mirror
        [ 0.66581 ,-3.459632 ,-0.596354704 ],          # Channel 9 - MTC019F - Mirror
        [ 1.742551113 ,-1.198263494 ,-0.03175 ],       # Channel 10 - MTC020B - Mirror
        [ 1.664513 ,-1.153211 ,-0.492286037 ],         # Channel 11 - MTC020F - Mirror
        [ 1.742552378 ,-2.543673903 ,-0.03175 ],       # Channel 12 - MTC021B - Mirror
        [ 1.664513 ,-2.498623 ,-0.554479813 ],         # Channel 13 - MTC021F - Mirror
        [ 2.086749319 ,-3.6143556 ,-0.03175 ],         # Channel 14 - MTCOW007B - Mirror
        [ 2.086749319 ,-3.6143556 ,-0.343210699 ],     # Channel 15 - MTCOW007M - Mirror
        [ 2.04155 ,-3.536086 ,-0.654671399 ],          # Channel 16 - MTCOW007F - Mirror
        [ 2.908297565 ,-2.482149258 ,-0.03175 ],       # Channel 17 - MTC022B - Mirror
        [ 2.979344 ,-2.537714 ,-0.640697029 ],         # Channel 18 - MTC022F - Mirror
        [ 0, 0, 0 ],                                   # Channel 19 - CTCUP004 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 20 - CTCUP005 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 21 - CTCUP006 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 22 - CTCAA001 - Ambient
        [ 0, 0, 0 ],                                   # Channel 23 - CTCCW001W - Weldment
        [ 0, 0, 0 ],                                   # Channel 24 - CTCCW001F - Weldment
        [ 0, 0, 0 ],                                   # Channel 25 - NC
        [ 0, 0, 0 ],                                   # Channel 26 - NC
        [ 0, 0, 0 ],                                   # Channel 27 - NC
        [ 0, 0, 0 ],                                   # Channel 28 - NC
        [ 0, 0, 0 ],                                   # Channel 29 - NC
        [ 0, 0, 0 ],                                   # Channel 30 - NC
        [ 0, 0, 0 ],                                   # Channel 31 - NC
    ],
    [ # TC Scanner 3
        [ 0, 0, 0 ],                                   # Channel 0 - CTCIB003 - ITJB
        [ 3.660920396 ,-2.003934071 ,-0.03175 ],       # Channel 1 - MTCOW008B - Mirror
        [ 3.660920396 ,-2.003934071 ,-0.343210699 ],   # Channel 2 - MTCOW008M - Mirror
        [ 3.581629 ,-1.96055 ,-0.661236652 ],          # Channel 3 - MTCOW008F - Mirror
        [ 1.076730286 ,0.045052481 ,-0.03175 ],        # Channel 4 - MTC023B - Mirror
        [ 0.998728 ,-1.22707e-26 ,-0.452627296 ],      # Channel 5 - MTC023F - Mirror
        [ 3.074163522 ,-1.198264746 ,-0.03175 ],       # Channel 6 - MTC024B - Mirror
        [ 2.996184 ,-1.153211 ,-0.576623673 ],         # Channel 7 - MTC024F - Mirror
        [ 2.408356414 ,0.045055371 ,-0.03175 ],        # Channel 8 - MTC025B - Mirror
        [ 2.330323 ,-0.000025 ,-0.512435684 ],         # Channel 9 - MTC025F - Mirror
        [ 3.741620174 ,-0.783712952 ,-0.03175 ],       # Channel 10 - MTC026B - Mirror
        [ 3.660394 ,-0.760146 ,-0.627142035 ],         # Channel 11 - MTC026F - Mirror
        [ 4.173498638 ,0.000000242705 ,-0.03175 ],     # Channel 12 - MTCOW009B - Mirror
        [ 4.173498638 ,0.000000242705 ,-0.343210699 ], # Channel 13 - MTCOW009M - Mirror
        [ 4.083126 ,-2.47123e-23 ,-0.664127315 ],      # Channel 14 - MTCOW009F - Mirror
        [ 3.074163522 ,1.198265232 ,-0.03175 ],        # Channel 15 - MTC027B - Mirror
        [ 2.996133 ,1.153211 ,-0.57787628 ],           # Channel 16 - MTC027F - Mirror
        [ 3.741620174 ,0.783713437 ,-0.03175 ],        # Channel 17 - MTC028B - Mirror
        [ 3.660394 ,0.760171 ,-0.627504767 ],          # Channel 18 - MTC028F - Mirror
        [ 3.660920396 ,2.003934556 ,-0.03175 ],        # Channel 19 - MTCOW010B - Mirror
        [ 3.660920396 ,2.003934556 ,-0.343210699 ],    # Channel 20 - MTCOW010M - Mirror
        [ 3.581629 ,1.96055 ,-0.661355998 ],           # Channel 21 - MTCOW010F - Mirror
        [ 0, 0, 0 ],                                   # Channel 22 - CTCUP007 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 23 - CTCUP008 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 24 - CTCLP002 - Lower Plenum
        [ 0, 0, 0 ],                                   # Channel 25 - CTCCW002T - Weldment
        [ 0, 0, 0 ],                                   # Channel 26 - NC
        [ 0, 0, 0 ],                                   # Channel 27 - NC
        [ 0, 0, 0 ],                                   # Channel 28 - NC
        [ 0, 0, 0 ],                                   # Channel 29 - NC
        [ 0, 0, 0 ],                                   # Channel 30 - NC
        [ 0, 0, 0 ],                                   # Channel 31 - NC
    ],
    [ # TC Scanner 4
        [ 0, 0, 0 ],                                   # Channel 0 - CTCIB004 - ITJB
        [ 0.332903464 ,0.47451003 ,-0.03175 ],         # Channel 1 - MTCIN004B - Mirror
        [ 0.332903464 ,0.47451003 ,-0.236886226 ],     # Channel 2 - MTCIN004M - Mirror
        [ 0.332994 ,0.384404 ,-0.443076928 ],          # Channel 3 - MTCIN004F - Mirror
        [ 1.742551113 ,1.19826398 ,-0.03175 ],         # Channel 4 - MTC029B - Mirror
        [ 1.664462 ,1.153211 ,-0.494721549 ],          # Channel 5 - MTC029F - Mirror
        [ 2.908297565 ,2.482149743 ,-0.03175 ],        # Channel 6 - MTC030B - Mirror
        [ 2.97942 ,2.537714 ,-0.641485997 ],           # Channel 7 - MTC030F - Mirror
        [ 1.742552378 ,2.543674388 ,-0.03175 ],        # Channel 8 - MTC031B - Mirror
        [ 1.664462 ,2.498623 ,-0.55784599 ],           # Channel 9 - MTC031F - Mirror
        [ 2.086749319 ,3.614356086 ,-0.03175 ],        # Channel 10 - MTCOW011B - Mirror
        [ 2.086749319 ,3.614356086 ,-0.343210699 ],    # Channel 11 - MTCOW011M - Mirror
        [ 2.041576 ,3.536086 ,-0.655163794 ],          # Channel 12 - MTCOW011F - Mirror
        [ 0.66580512 ,3.549739143 ,-0.03175 ],         # Channel 13 - MTC032B - Mirror
        [ 0.66581 ,3.459632 ,-0.599254526 ],           # Channel 14 - MTC032F - Mirror
        [ 0.095002175 ,4.172417464 ,-0.03175 ],        # Channel 15 - MTCOW012B - Mirror
        [ 0.095002175 ,4.172417464 ,-0.343210699 ],    # Channel 16 - MTCOW012M - Mirror
        [ 0.092936 ,4.082059 ,-0.652211866 ],          # Channel 17 - MTCOW012F - Mirror
        [ 0, 0, 0 ],                                   # Channel 18 - CTCUP009 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 19 - CTCUP010 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 20 - CTCAA002 - Ambient
        [ 0, 0, 0 ],                                   # Channel 21 - CTCCW002W - Weldment
        [ 0, 0, 0 ],                                   # Channel 22 - CTCCW002F - Weldment
        [ 0, 0, 0 ],                                   # Channel 23 - NC
        [ 0, 0, 0 ],                                   # Channel 24 - NC
        [ 0, 0, 0 ],                                   # Channel 25 - NC
        [ 0, 0, 0 ],                                   # Channel 26 - NC
        [ 0, 0, 0 ],                                   # Channel 27 - NC
        [ 0, 0, 0 ],                                   # Channel 28 - NC
        [ 0, 0, 0 ],                                   # Channel 29 - NC
        [ 0, 0, 0 ],                                   # Channel 30 - NC
        [ 0, 0, 0 ],                                   # Channel 31 - NC
    ],
    [ # TC Scanner 5
        [ 0, 0, 0 ],                                   # Channel 0 - CTCIB005 - ITJB
        [ 0 ,1.435519343 ,-0.03175 ],                  # Channel 1 - MTC001B - Mirror
        [ 0 ,1.345413 ,-0.463925044 ],                 # Channel 2 - MTC001F - Mirror
        [ 0 ,2.588730143 ,-0.03175 ],                  # Channel 3 - MTC002B - Mirror
        [ 0 ,2.498623 ,-0.52102807 ],                  # Channel 4 - MTC002F - Mirror
        [ -0.332903464 ,0.47451003 ,-0.03175 ],        # Channel 5 - MTCIN001B - Mirror
        [ -0.332903464 ,0.47451003 ,-0.236886226 ],    # Channel 6 - MTCIN001M - Mirror
        [ -0.332892 ,0.384404 ,-0.443076013 ],         # Channel 7 - MTCIN001F - Mirror
        [ -0.66580512 ,3.549739143 ,-0.03175 ],        # Channel 8 - MTC003B - Mirror
        [ -0.66581 ,3.459632 ,-0.599254526 ],          # Channel 9 - MTC003F - Mirror
        [ -1.742551113 ,1.19826398 ,-0.03175 ],        # Channel 10 - MTC004B - Mirror
        [ -1.664513 ,1.153211 ,-0.494723824 ],         # Channel 11 - MTC004F - Mirror
        [ -1.742552378 ,2.543674388 ,-0.03175 ],       # Channel 12 - MTC005B - Mirror
        [ -1.664513 ,2.498623 ,-0.557848246 ],         # Channel 13 - MTC005F - Mirror
        [ -2.086749319 ,3.614356086 ,-0.03175 ],       # Channel 14 - MTCOW001B - Mirror
        [ -2.086749319 ,3.614356086 ,-0.343210699 ],   # Channel 15 - MTCOW001M - Mirror
        [ -2.04155 ,3.536086 ,-0.655162392 ],          # Channel 16 - MTCOW001F - Mirror
        [ -2.908297565 ,2.482149743 ,-0.03175 ],       # Channel 17 - MTC006B - Mirror
        [ -2.979344 ,2.537714 ,-0.641479979 ],         # Channel 18 - MTC006F - Mirror
        [ 0, 0, 0 ],                                   # Channel 19 - CTCUP013 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 20 - CTCUP014 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 21 - CTCLP003 - Lower Plenum
        [ 0, 0, 0 ],                                   # Channel 22 - CTCCW003T - Weldment
        [ 0, 0, 0 ],                                   # Channel 23 - NC
        [ 0, 0, 0 ],                                   # Channel 24 - NC
        [ 0, 0, 0 ],                                   # Channel 25 - NC
        [ 0, 0, 0 ],                                   # Channel 26 - NC
        [ 0, 0, 0 ],                                   # Channel 27 - NC
        [ 0, 0, 0 ],                                   # Channel 28 - NC
        [ 0, 0, 0 ],                                   # Channel 29 - NC
        [ 0, 0, 0 ],                                   # Channel 30 - NC
        [ 0, 0, 0 ],                                   # Channel 31 - NC
    ],
    [ # TC Scanner 6
        [ 0, 0, 0 ],                                   # Channel 0 - CTCIB006 - ITJB
        [ -3.660920396 ,2.003934556 ,-0.03175 ],       # Channel 1 - MTCOW002B - Mirror
        [ -3.660920396 ,2.003934556 ,-0.343210699 ],   # Channel 2 - MTCOW002M - Mirror
        [ -3.581629 ,1.96055 ,-0.661355998 ],          # Channel 3 - MTCOW002F - Mirror
        [ -1.076730286 ,-0.045051996 ,-0.03175 ],      # Channel 4 - MTC007B - Mirror
        [ -0.998728 ,0 ,-0.452627296 ],                # Channel 5 - MTC007F - Mirror
        [ -3.074163522 ,1.198265232 ,-0.03175 ],       # Channel 6 - MTC008B - Mirror
        [ -2.996184 ,1.153211 ,-0.577880375 ],         # Channel 7 - MTC008F - Mirror
        [ -2.408356414 ,-0.045051129 ,-0.03175 ],      # Channel 8 - MTC009B - Mirror
        [ -2.330323 ,0 ,-0.512435731 ],                # Channel 9 - MTC009F - Mirror
        [ -3.741620173 ,0.783713437 ,-0.03175 ],       # Channel 10 - MTC010B - Mirror
        [ -3.660394 ,0.760146 ,-0.627504278 ],         # Channel 11 - MTC010F - Mirror
        [ -4.173498638 ,0 ,-0.03175 ],                 # Channel 12 - MTCOW003B - Mirror
        [ -4.173498638 ,0 ,-0.343210699 ],             # Channel 13 - MTCOW003M - Mirror
        [ -4.083126 ,0 ,-0.664127315 ],                # Channel 14 - MTCOW003F - Mirror
        [ -3.074163522 ,-1.198264746 ,-0.03175 ],      # Channel 15 - MTC011B - Mirror
        [ -2.996133 ,-1.153211 ,-0.57661952 ],         # Channel 16 - MTC011F - Mirror
        [ -3.741620173 ,-0.783712952 ,-0.03175 ],      # Channel 17 - MTC012B - Mirror
        [ -3.660394 ,-0.760171 ,-0.627142516 ],        # Channel 18 - MTC012F - Mirror
        [ -3.660920396 ,-2.003934071 ,-0.03175 ],      # Channel 19 - MTCOW004B - Mirror
        [ -3.660920396 ,-2.003934071 ,-0.343210699 ],  # Channel 20 - MTCOW004M - Mirror
        [ -3.581629 ,-1.96055 ,-0.661236652 ],         # Channel 21 - MTCOW004F - Mirror
        [ 0, 0, 0 ],                                   # Channel 22 - CTCUP011 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 23 - CTCUP012 - Upper Plenum
        [ 0, 0, 0 ],                                   # Channel 24 - CTCAA003 - Ambient
        [ 0, 0, 0 ],                                   # Channel 25 - CTCCW003W - Weldment
        [ 0, 0, 0 ],                                   # Channel 26 - CTCCW003F - Weldment
        [ 0, 0, 0 ],                                   # Channel 27 - NC
        [ 0, 0, 0 ],                                   # Channel 28 - NC
        [ 0, 0, 0 ],                                   # Channel 29 - NC
        [ 0, 0, 0 ],                                   # Channel 30 - NC
        [ 0, 0, 0 ],                                   # Channel 31 - NC
    ]
]
tc_locs = np.array([item for sublist in tc_locs for item in sublist])
idx_mirror_f = [(label.startswith("MTC") and label.endswith("F")) for label in tc_labels]
print('number of Mirror Front surface TCs = ', sum(idx_mirror_f))
idx_mirror_b = [(label.startswith("MTC") and label.endswith("B")) for label in tc_labels]
print('number of Mirror Back surface TCs = ', sum(idx_mirror_b))
idx_mirror_m = [(label.startswith("MTC") and label.endswith("M")) for label in tc_labels]
print('number of Mirror Middle TCs = ', sum(idx_mirror_m))


from scipy.special import factorial

def nm2noll(n, m):
    """Convert indices `(n, m)` to the Noll's index `k`.

    Note that Noll's index `k` starts from one and Python indexing is
    zero-based.

    """
    k = n * (n + 1) // 2 + abs(m)
    if (m <= 0 and n % 4 in (0, 1)) or (m >= 0 and n % 4 in (2, 3)):
        k += 1
    return k

nmax = 5000
narray = np.zeros(nmax)
marray = np.zeros(nmax)
for ni in range(nmax):
    for mi in range(-ni, ni + 1, 2):
        idx = nm2noll(ni,mi)-1
        #print(ni, mi, idx)
        if idx<nmax:
            narray[idx] = ni
            marray[idx] = mi

def noll_to_nm(N):
    """
    Convert Noll index to radial degree n and azimuthal order m.
    """
    n = int(narray[N-1])
    m = int(marray[N-1])
    return n, m        
        
def radial_polynomial(n, m, rho):
    """
    Compute the radial Zernike polynomial Rnm(rho) for a given radial degree n
    and azimuthal order m at the radial distance rho.
    """
    R = np.zeros_like(rho)
    R[np.isnan(rho)] = np.nan
    for s in range((n - abs(m)) // 2 + 1):
        c = (-1)**s * factorial(n - s) / (
            factorial(s) * factorial((n + abs(m)) // 2 - s) * factorial((n - abs(m)) // 2 - s)
        )
        R += c * rho**(n - 2 * s)
    return R


def zernike_polynomial(N, x, y):
    """
    Compute the Zernike polynomial Znm(x, y) for a given N.
    N starts from 1.
    """
    # Convert Cartesian coordinates (x, y) to polar coordinates (rho, theta)
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Set values outside the unit disk to NaN (invalid)
    rho[rho > 1] = np.nan
    
    # Convert Noll index N to n and m
    n,m = noll_to_nm(N)
    
    # Compute the radial part of the polynomial
    Rnm = radial_polynomial(n, m, rho)
    
    # Compute the angular part (cosine or sine)
    if m >= 0:
        angular = np.cos(m * theta)
    else:
        angular = np.sin(abs(m) * theta)
    
    # Combine radial and angular parts to get the Zernike polynomial
    Z = Rnm * angular
    
    # Normalize the polynomial to have RMS = 1
    normalization = np.sqrt(( (n + 1)) * (1 if m == 0 else 2))  # Standard normalization factor
    return Z * normalization
