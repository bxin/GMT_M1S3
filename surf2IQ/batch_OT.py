import numpy as np
import ceo
import sys
sys.path.insert(0, '../')
from M1S_tools import *

from scipy.interpolate import griddata
from scipy.ndimage import rotate

# entrance pupil sampling
n_px = 512  #*8 #each measured seg is 853pixels
L = 25.5

gmt = ceo.GMT_MX()
src = ceo.Source('V',rays_box_size=L,rays_box_sampling=n_px)
src>>(gmt,)
+src
pssn0 ,pssn_data = gmt.PSSn(src,save=True)

segment_n_px = 990 #this is the largest grid so far out of RFCML
#this cel is simply for defining the grid
# we want to define a grid that is slighly larger than the clear aperture so we can feed it to CEO
dateID = sys.argv[1]
setID = sys.argv[2]
globstring = '../opti_h5/%s*set%s*h5'%(dateID, setID)
fileset = sorted(glob.glob(globstring))
assert len(fileset)==1
h5file = fileset

#h5file = ['../opti_h5/250131 GMT3 set4 27 modes gmtMorph170614 LFSsub comp - stitched patched.h5']
for i in range(1):
    m1s,centerRow,centerCol,pixelSize, ts = readH5Map([h5file[i]])
    [x, y] = mkXYGrid(m1s, centerRow, centerCol, pixelSize)
    if m1s.shape[0]< segment_n_px: 
        #get the smallest map, and all others need to shrink to match
        #if we get the largest map, the smaller one will need to be extrapolated which is very tricky
        #even the smallest is larger than the Clear Aperture, so we are good!
        segment_n_px = m1s.shape[0]
        xm = x.copy()
        ym = y.copy()
m1sArray = np.zeros((segment_n_px, segment_n_px, 7))

for i in range(7):
    idx = ~np.isnan(m1s)
    #here we do not have 7 maps on 7 sometimes different grids, so the interpolation here is only for filling the fiducial holes+drain hole+reflection
    m1sArray[:,:,i] = griddata(np.vstack((x[idx], y[idx])).transpose(), m1s[idx], (xm, ym), method='nearest')
    
segment_L = np.max(xm)-np.min(xm)
segment_clear_aperture = 8.365
u = np.linspace(-1,1,segment_n_px)
x,y = np.meshgrid(u,u)
r = np.hypot(x,y)
o = np.arctan2(y,x)
phase_map = m1s

mask = np.ones_like(r)
mask[r>(segment_clear_aperture/segment_L)] = np.nan

phase_map7 = np.zeros((segment_n_px, segment_n_px, 7))
#rotation_angles = [0, 0, 0, 0, 0, 0, 0]
rotation_angles = [0, 30, 220, 20, 50, 199, 322] #random rotations, but repeatable
#rotation_angles = np.random.uniform(0, 360, 7)
print(rotation_angles)
for i, angle in enumerate(rotation_angles):
    phase_map7[:,:,i] = rotate(m1sArray[:,:,i], angle, reshape=False)
    
# writing to the "CEO" file format
import os 
from collections import OrderedDict

data = OrderedDict()
data['Ni']     = np.array( segment_n_px, dtype=np.int32)
data['L']      = np.array( segment_L,    dtype=np.double)
data['N_SET']  = np.array( 7,     dtype=np.int32)
data['N_MODE'] = np.array( 1,     dtype=np.int32)
data['s2b']    = np.array( [0,1,2,3,4,5,6], dtype=np.int32) #which segment uses which basis set
data['M'] = phase_map7.flatten(order='F')

path_to_ceo = "/home/ubuntu/CEO/" # CHANGE THIS TO THE CEO PATH ON YOUR MACHINE
filename = "made-up_mode"
path_to_modes = os.path.join( path_to_ceo , 'gmtMirrors' , filename+'.ceo' )
with open(path_to_modes,'w') as f:
    for key in data:
        data[key].tofile(f)

# loading the new mode and setting it up on all M1 segments with 1e-7m amplitude
gmt = ceo.GMT_MX(M1_mirror_modes="made-up_mode",M1_N_MODE=1)
src = ceo.Source('V',rays_box_size=L,rays_box_sampling=n_px)
state = gmt.state
state["M1"]["modes"][:,0] = 1e-6
gmt^=state
src>>(gmt,)
+src

# "aberrated" GMT WFE RMS [nm] & PSSn
print('-------Surface RMS = %.0f nm,'%(np.std(m1s[~np.isnan(m1s)])*1e3), 
      ' wavefront RMS = %.1f nm, '%src.wavefront.rms(-9)[0], 
      'PSSn = %.3f'%(gmt.PSSn(src,**pssn_data)[0]))
