import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.jet()

import sys
sys.path.insert(0, '../')
from M1S_tools import *

import matplotlib.animation as animation
from scipy.interpolate import griddata

#starting from a few hours before turning on the suckulators
#ending a few hours after turning on the suckulators
start_time = 1734445080
end_time = 1734549480
duration = end_time - start_time
#duration = 10000 #for testing.
nsamples = 100
tc, tt = getDBData(start_time,'m1_s1_thermal_ctrl/i/tc_temperature/value', duration_in_s=duration, samples=nsamples)
tmirror, tt = getDBData(start_time,'m1_s1_thermal_ctrl/i/mirror_temperature/value', duration_in_s=duration, samples=nsamples)
tambient, tt = getDBData(start_time,'m1_s1_thermal_ctrl/i/ambient_temperature/value', duration_in_s=duration, samples=nsamples)
time_steps = tc.shape[0]  # Number of time steps

# Define the grid for interpolation
grid_size = 500  # Resolution of the grid
xi = np.linspace(-radius_of_CA, radius_of_CA, grid_size)
yi = np.linspace(-radius_of_CA, radius_of_CA, grid_size)
xi, yi = np.meshgrid(xi, yi)

# Create a circular mask for the given radius
mask = np.sqrt(xi**2 + yi**2) <= radius_of_CA

# Set the fixed color limits (vmin, vmax) for the contour plot
tc[tc>30] = np.nan
tc[tc<10] = np.nan
vmin = np.nanmin(tambient)  # Minimum value for the colorbar
vmax = np.nanmax(tmirror)  # Maximum value for the colorbar
#levels = np.linspace(vmin, vmax, 100)  # Explicitly define levels
vmax_index = np.nanargmax(tmirror)
tt0 = tt[vmax_index]
#print('vmin, vmax, ==== ', vmin, vmax)
cbar_span = 0.6 #deg

# Function to update the plot for each time step
def update_plot(frame):

    if hasattr(update_plot, 'timeline'):
        update_plot.timeline.remove()
    update_plot.timeline = ax1.axvline(x=(tt[frame]-tt0)/3600., color='r', linestyle='--', label='Current Time')
    ax1.set_title('Current Time = %.1f hours'%((tt[frame]-tt0)/3600.))

    ax2.clear()  # **Clear the right plot (ax2)**
    # Get the data for the current time step
    x = tc_locs[idx_mirror_f][:,0]
    y = tc_locs[idx_mirror_f][:,1]
    z = tc[frame, idx_mirror_f]  # Extract z-values for the current time step

    # Interpolate z-values to the grid
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Apply the circular mask
    zi[~mask] = np.nan  # Set values outside the circle to NaN

    midpoint = np.nanmean(zi)
    levels = np.linspace(midpoint-cbar_span/2., midpoint+cbar_span/2., 100)
    zi = np.clip(zi, np.min(levels), np.max(levels))

    # Plot the contour plot
    contour = ax2.contourf(xi, yi, zi, levels=levels, cmap='jet') #use levels=levels if you want colorbar fixed!!!
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title('Front Surface, Max - Min = %.2f Deg'%(np.nanmax(z)-np.nanmin(z)))
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    if hasattr(update_plot, 'cbar2'):
        update_plot.cbar2.remove()
    #update_plot.cbar2 = fig.colorbar(contour, ax=ax2, label='Colorbar Max - Min = %.2f Deg'%cbar_span)  # Store the colorbar
    update_plot.cbar2 = fig.colorbar(contour, ax=ax2, label='Temperature')  # Store the colorbar
    #update_plot.cbar2.set_ticks(np.linspace(vmin, vmax, num=5))
    ax2.scatter(x, y, c=z, edgecolor='k', cmap='jet', vmin=np.min(levels), vmax=np.max(levels), label='TCs')
    ax2.legend(loc='upper right')

    #ax3 for middle
    ax3.clear()
    x = tc_locs[idx_mirror_m][:,0]
    y = tc_locs[idx_mirror_m][:,1]
    z = tc[frame, idx_mirror_m]
    zi = griddata((x, y), z, (xi, yi), method='linear')
    zi[~mask] = np.nan  # Set values outside the circle to NaN

    midpoint = np.nanmean(zi)
    levels = np.linspace(midpoint-cbar_span/2., midpoint+cbar_span/2., 100)
    zi = np.clip(zi, np.min(levels), np.max(levels))

    contour = ax3.contourf(xi, yi, zi, levels=levels, cmap='jet')
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title('Middle, Max - Min = %.2f Deg'%(np.nanmax(zi)-np.nanmin(zi)))
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    if hasattr(update_plot, 'cbar3'):
        update_plot.cbar3.remove()
    update_plot.cbar3 = fig.colorbar(contour, ax=ax3, label='Temperature')  # Store the colorbar
    ax3.scatter(x, y, c=z, edgecolor='k', vmin=np.min(levels), vmax=np.max(levels), cmap='jet', label='TCs')
    ax3.legend(loc='upper right')

    #ax4 for back
    ax4.clear()
    x = tc_locs[idx_mirror_b][:,0]
    y = tc_locs[idx_mirror_b][:,1]
    z = tc[frame, idx_mirror_b]
    zi = griddata((x, y), z, (xi, yi), method='linear')
    zi[~mask] = np.nan  # Set values outside the circle to NaN

    midpoint = np.nanmean(zi)
    levels = np.linspace(midpoint-cbar_span/2., midpoint+cbar_span/2., 100)
    zi = np.clip(zi, np.min(levels), np.max(levels))

    contour = ax4.contourf(xi, yi, zi, levels=levels, cmap='jet')
    ax4.set_aspect('equal', adjustable='box')
    ax4.set_title('Back Surface, Max - Min = %.2f Deg'%(np.nanmax(zi)-np.nanmin(zi)))
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    if hasattr(update_plot, 'cbar4'):
        update_plot.cbar4.remove()
    update_plot.cbar4 = fig.colorbar(contour, ax=ax4, label='Temperature')  # Store the colorbar
    ax4.scatter(x, y, c=z, edgecolor='k', cmap='jet', vmin=np.min(levels), vmax=np.max(levels), label='TCs')
    ax4.legend(loc='upper right')


    # Redraw the figure
    plt.draw()

# Create the figure and axis for the subplots (2 panels)
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))  # **Create 2 subplots: left (ax1) for tmirror, right (ax2) for contour plot**
#fig.subplots_adjust(hspace=0.4, wspace=0.3)
fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(12)

ax1 = plt.subplot2grid(shape=(2, 3), loc=(0,0), colspan=2) #HD
ax2 = plt.subplot2grid(shape=(2, 3), loc=(1,0)) #front surf
ax3 = plt.subplot2grid(shape=(2, 3), loc=(0,2)) #middle
ax4 = plt.subplot2grid(shape=(2, 3), loc=(1,1)) #back

# Plot the tmirror on the left (ax1)  **New: Plot `tmirror` over time on the left panel**
ax1.plot((tt-tt0)/3600., tmirror, label="Mirror Temperature", color="b")
ax1.plot((tt-tt0)/3600., tambient, label="Ambient Temperature", color="k")
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Mirror Temperature (°C)')
#ax1.set_title('Mirror Temperature over Time')
ax1.set_ylim(min(tambient)-0.2*(vmax-vmin), max(tmirror)+0.2*(vmax-vmin))  # Adjust y-axis based on your data
timeline = ax1.axvline(x=(tt[0]-tt0)/3600., color='r', linestyle='--', label='Current Time')
ax1.legend(loc='upper right')
timeline.remove()
ax1.grid()

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=time_steps, repeat=False)

# Save the animation as an MP4 file
ani.save('tc_surface_animation.mp4', writer='ffmpeg', fps=10)  # Modify fps as needed

#plt.show()

