from Load_data import Data_from_nc
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors as cls

def set_plot_param():
    """Set my own customized plotting parameters"""

    mpl.rc('axes', edgecolor='dimgrey')
    mpl.rc('axes', labelcolor='dimgrey')
    mpl.rc('xtick', color='dimgrey')
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', color='dimgrey')
    mpl.rc('ytick', labelsize=12)
    mpl.rc('axes', titlesize=14)
    mpl.rc('axes', labelsize=12)
    mpl.rc('legend', fontsize='large')
    mpl.rc('text', color='dimgrey')

data_surface = Data_from_nc('surfdata_irrigation_method.nc')   #load the data
data_irrigation_method = data_surface.load_variable('irrigation_method')


data_lon = data_surface.load_variable('LONGXY')
data_lon = data_lon[0,:]
data_lat = data_surface.load_variable('LATIXY')
data_lat = data_lat[:,0]
data_lat = data_lat[29 :]

data_maize = data_irrigation_method[3, :, :]
data_maize = data_maize[29 :, :]

data_rice = data_irrigation_method[47, :, :]
data_rice = data_rice[29 :, :]

data_wheat = data_irrigation_method[5, :, :]
data_wheat = data_wheat[29 :, :]

data_pulses = data_irrigation_method[43, :, :]
data_pulses = data_pulses[29 :, :]

f = plt.figure(figsize = (12, 8), dpi=1000)  # initiate the figure
set_plot_param()
f.subplots_adjust(hspace=0.15, wspace=0.1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.text(0.01,0.92,'a',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='BrBG'
bwr = mpl.cm.get_cmap('BrBG')
colors = ['whitesmoke','red','blue','green','white']
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [0.1,1.1,2.1,3.1]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)


h = ax1.pcolormesh(data_lon,data_lat, data_maize, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='neither', shrink = 0.8, pad = 0, aspect = 50)
cbar.set_label('irrigation method',fontsize = 12)

#ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(["","drip             ","sprinkler        ","flood            "], fontsize=10, horizontalalignment='right')
#plt.title('corn and wheat')
ax1.set_title('Temperate maize',loc='right',fontsize = 12)


ax1 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())

ax1.text(0.01,0.92,'b',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='BrBG'
bwr = mpl.cm.get_cmap('BrBG')
colors = ['whitesmoke','red','blue','green','white']
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [0.1,1.1,2.1,3.1]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)


h = ax1.pcolormesh(data_lon,data_lat,data_rice, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='neither', shrink = 0.8, pad = 0, aspect = 50)
cbar.set_label('irrigation method',fontsize = 12)

#ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(["","drip             ","sprinkler        ","flood            "], fontsize=10, horizontalalignment='right')
ax1.set_title('Rice',loc='right',fontsize = 12)

ax1 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())

ax1.text(0.01,0.92,'c',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='BrBG'
bwr = mpl.cm.get_cmap('BrBG')
colors = ['whitesmoke','red','blue','green','white']
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [0.1,1.1,2.1,3.1]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)


h = ax1.pcolormesh(data_lon,data_lat,data_wheat, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='neither', shrink = 0.8, pad = 0, aspect = 50)
cbar.set_label('irrigation method',fontsize = 12)

#ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(["","drip             ","sprinkler        ","flood            "], fontsize=10, horizontalalignment='right')
ax1.set_title('Spring wheat',loc='right',fontsize = 12)

ax1 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())

ax1.text(0.01,0.92,'c',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='BrBG'
bwr = mpl.cm.get_cmap('BrBG')
colors = ['whitesmoke','red','blue','green','white']
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [0.1,1.1,2.1,3.1]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)


h = ax1.pcolormesh(data_lon,data_lat,data_pulses, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='neither', shrink = 0.8, pad = 0, aspect = 50)
cbar.set_label('irrigation method',fontsize = 12)

#ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(["","drip             ","sprinkler        ","flood            "], fontsize=10, horizontalalignment='right')
ax1.set_title('Pulses',loc='right',fontsize = 12)

plt.savefig('IrrMeth.png')
