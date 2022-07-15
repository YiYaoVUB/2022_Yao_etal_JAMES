from Load_data import Data_from_nc
from Calculation_utils import Calculation
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

data_E_NOI = Data_from_nc('NOIRR\EFLX_LH_TOT_timmean.nc')   #load the data
data_variable_E_NOI = data_E_NOI.load_variable('EFLX_LH_TOT')
data_variable_E_NOI[data_variable_E_NOI >= 10000000000] = None
data_variable_E_NOI = data_variable_E_NOI[0, :, :]
data_variable_E_NOI = data_variable_E_NOI[29 :, :]
data_lon = data_E_NOI.load_variable('lon')
data_lat = data_E_NOI.load_variable('lat')
data_lat = data_lat[29 :]

data_E_IRR = Data_from_nc('IRR_satu\EFLX_LH_TOT_timmean.nc')   #load the data
data_variable_E_IRR = data_E_IRR.load_variable('EFLX_LH_TOT')
data_variable_E_IRR[data_variable_E_IRR >= 10000000000] = None
data_variable_E_IRR = data_variable_E_IRR[0, :, :]
data_variable_E_IRR = data_variable_E_IRR[29 :, :]

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
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),bwr(0.5),bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [-10, -5, -2, -0.1, 0.1, 2, 5, 10]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)

h = ax1.pcolormesh(data_lon,data_lat,data_variable_E_IRR - data_variable_E_NOI, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='both', shrink = 1, pad = 0, aspect = 50)
cbar.set_label('$\Delta$ LHF ($\mathregular{W/m^2}$)',fontsize = 12)
ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(ticklabs, fontsize=12)
plt.title('Latent heat flux (IRR - NOI)')

data_E_NOI = Data_from_nc('NOIRR\FSH_timmean.nc')   #load the data
data_variable_E_NOI = data_E_NOI.load_variable('FSH')
data_variable_E_NOI[data_variable_E_NOI >= 10000000000] = None
data_variable_E_NOI = data_variable_E_NOI[0, :, :]
data_variable_E_NOI = data_variable_E_NOI[29 :, :]

data_E_IRR = Data_from_nc('IRR_satu\FSH_timmean.nc')   #load the data
data_variable_E_IRR = data_E_IRR.load_variable('FSH')
data_variable_E_IRR[data_variable_E_IRR >= 10000000000] = None
data_variable_E_IRR = data_variable_E_IRR[0, :, :]
data_variable_E_IRR = data_variable_E_IRR[29 :, :]

ax1 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
ax1.text(0.01,0.92,'b',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='BrBG'
bwr = mpl.cm.get_cmap('BrBG')
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),bwr(0.5),bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [-10, -5, -2, -0.1, 0.1, 2, 5, 10]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)

h = ax1.pcolormesh(data_lon,data_lat,data_variable_E_IRR - data_variable_E_NOI, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='both', shrink = 1, pad = 0, aspect = 50)
cbar.set_label('$\Delta$ SHF ($\mathregular{W/m^2}$)',fontsize = 12)
ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(ticklabs, fontsize=12)
plt.title('Sensible heat flux (IRR - NOI)')

data_E_NOI = Data_from_nc('NOIRR\SWup_timmean.nc')   #load the data
data_variable_E_NOI = data_E_NOI.load_variable('SWup')
data_variable_E_NOI[data_variable_E_NOI >= 10000000000] = None
data_variable_E_NOI = data_variable_E_NOI[0, :, :]
data_variable_E_NOI = data_variable_E_NOI[29 :, :]

data_E_IRR = Data_from_nc('IRR_satu\SWup_timmean.nc')   #load the data
data_variable_E_IRR = data_E_IRR.load_variable('SWup')
data_variable_E_IRR[data_variable_E_IRR >= 10000000000] = None
data_variable_E_IRR = data_variable_E_IRR[0, :, :]
data_variable_E_IRR = data_variable_E_IRR[29 :, :]

ax1 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
ax1.text(0.01,0.92,'c',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='BrBG'
bwr = mpl.cm.get_cmap('BrBG')
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),bwr(0.5),bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [-5, -2, -1, -0.1, 0.1, 1, 2, 5]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)

h = ax1.pcolormesh(data_lon,data_lat,data_variable_E_IRR - data_variable_E_NOI, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='both', shrink = 1, pad = 0, aspect = 50)
cbar.set_label('$\Delta$ SWup ($\mathregular{W/m^2}$)',fontsize = 12)
ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(ticklabs, fontsize=12)
plt.title('Upwelling shortwave radiation (IRR - NOI)')


data_E_NOI = Data_from_nc('NOIRR\LWup_timmean.nc')   #load the data
data_variable_E_NOI = data_E_NOI.load_variable('LWup')
data_variable_E_NOI[data_variable_E_NOI >= 10000000000] = None
data_variable_E_NOI = data_variable_E_NOI[0, :, :]
data_variable_E_NOI = data_variable_E_NOI[29 :, :]

data_E_IRR = Data_from_nc('IRR_satu\LWup_timmean.nc')   #load the data
data_variable_E_IRR = data_E_IRR.load_variable('LWup')
data_variable_E_IRR[data_variable_E_IRR >= 10000000000] = None
data_variable_E_IRR = data_variable_E_IRR[0, :, :]
data_variable_E_IRR = data_variable_E_IRR[29 :, :]

ax1 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
ax1.text(0.01,0.92,'d',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='BrBG'
bwr = mpl.cm.get_cmap('BrBG')
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),bwr(0.5),bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [-5, -2, -1, -0.1, 0.1, 1, 2, 5]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)

h = ax1.pcolormesh(data_lon,data_lat,data_variable_E_IRR - data_variable_E_NOI, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='both', shrink = 1, pad = 0, aspect = 50)
cbar.set_label('$\Delta$ LWup ($\mathregular{W/m^2}$)',fontsize = 12)
ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(ticklabs, fontsize=12)
plt.title('Upwelling longwave radiation (IRR - NOI)')

plt.savefig('C:\Research1\Figures\\Energy.png')

f = plt.figure(figsize = (12, 8), dpi=1000)  # initiate the figure
set_plot_param()

data_E_NOI = Data_from_nc('NOIRR\Qle_timmean.nc')   #load the data
data_variable_E_NOI = data_E_NOI.load_variable('Qle')
data_variable_E_NOI[data_variable_E_NOI >= 10000000000] = None
data_variable_E_NOI = data_variable_E_NOI[0, :, :]
data_variable_E_NOI = data_variable_E_NOI[29 :, :] / 28.94 * 365

data_E_IRR = Data_from_nc('IRR_satu\Qle_timmean.nc')   #load the data
data_variable_E_IRR = data_E_IRR.load_variable('Qle')
data_variable_E_IRR[data_variable_E_IRR >= 10000000000] = None
data_variable_E_IRR = data_variable_E_IRR[0, :, :]
data_variable_E_IRR = data_variable_E_IRR[29 :, :] / 28.94 * 365

ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.text(0.01,0.92,'a',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='bwr_r'
bwr = mpl.cm.get_cmap('bwr_r')
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),bwr(0.5),bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [-100, -50, -20, -1, 1, 20, 50, 100]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)

h = ax1.pcolormesh(data_lon,data_lat,data_variable_E_IRR - data_variable_E_NOI, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='both', shrink = 1, pad = 0, aspect = 50)
cbar.set_label('$\Delta$ E (mm/year)',fontsize = 12)
ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(ticklabs, fontsize=12)
plt.title('Evaporation (IRR - NOI)')


data_E_NOI = Data_from_nc('NOIRR\QVEGT_timmean.nc')   #load the data
data_variable_E_NOI = data_E_NOI.load_variable('QVEGT')
data_variable_E_NOI[data_variable_E_NOI >= 10000000000] = None
data_variable_E_NOI = data_variable_E_NOI[0, :, :]
data_variable_E_NOI = data_variable_E_NOI[29 :, :] * 31536000

data_E_IRR = Data_from_nc('IRR_satu\QVEGT_timmean.nc')   #load the data
data_variable_E_IRR = data_E_IRR.load_variable('QVEGT')
data_variable_E_IRR[data_variable_E_IRR >= 10000000000] = None
data_variable_E_IRR = data_variable_E_IRR[0, :, :]
data_variable_E_IRR = data_variable_E_IRR[29 :, :] * 31536000

ax1 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
ax1.text(0.01,0.92,'b',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='bwr_r'
bwr = mpl.cm.get_cmap('bwr_r')
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),bwr(0.5),bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [-50, -20, -10, -1, 1, 10, 20, 50]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)

h = ax1.pcolormesh(data_lon,data_lat,data_variable_E_IRR - data_variable_E_NOI, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='both', shrink = 1, pad = 0, aspect = 50)
cbar.set_label('$\Delta$ TR (mm/year)',fontsize = 12)
ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(ticklabs, fontsize=12)
plt.title('Transpiration (IRR - NOI)')


data_E_NOI = Data_from_nc('NOIRR\QRUNOFF_timmean.nc')   #load the data
data_variable_E_NOI = data_E_NOI.load_variable('QRUNOFF')
data_variable_E_NOI[data_variable_E_NOI >= 10000000000] = None
data_variable_E_NOI = data_variable_E_NOI[0, :, :]
data_variable_E_NOI = data_variable_E_NOI[29 :, :] * 31536000

data_E_IRR = Data_from_nc('IRR_satu\QRUNOFF_timmean.nc')   #load the data
data_variable_E_IRR = data_E_IRR.load_variable('QRUNOFF')
data_variable_E_IRR[data_variable_E_IRR >= 10000000000] = None
data_variable_E_IRR = data_variable_E_IRR[0, :, :]
data_variable_E_IRR = data_variable_E_IRR[29 :, :] * 31536000


f.subplots_adjust(hspace=0.15, wspace=0.1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
ax1 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
ax1.text(0.01,0.92,'c',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='bwr_r'
bwr = mpl.cm.get_cmap('bwr_r')
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),bwr(0.5),bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [-100, -50, -20, -1, 1, 20, 50, 100]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)

h = ax1.pcolormesh(data_lon,data_lat,data_variable_E_IRR - data_variable_E_NOI, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='both', shrink = 1, pad = 0, aspect = 50)
cbar.set_label('$\Delta$ R (mm/year)',fontsize = 12)
ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(ticklabs, fontsize=12)
plt.title('Runoff (IRR - NOI)')

data_E_NOI = Data_from_nc('NOIRR\TOTSOILLIQ_timmean.nc')   #load the data
data_variable_E_NOI = data_E_NOI.load_variable('TOTSOILLIQ')
data_variable_E_NOI[data_variable_E_NOI >= 10000000000] = None
data_variable_E_NOI = data_variable_E_NOI[0, :, :]
data_variable_E_NOI = data_variable_E_NOI[29 :, :]

data_E_IRR = Data_from_nc('IRR_satu\TOTSOILLIQ_timmean.nc')   #load the data
data_variable_E_IRR = data_E_IRR.load_variable('TOTSOILLIQ')
data_variable_E_IRR[data_variable_E_IRR >= 10000000000] = None
data_variable_E_IRR = data_variable_E_IRR[0, :, :]
data_variable_E_IRR = data_variable_E_IRR[29 :, :]

ax1 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
ax1.text(0.01,0.92,'d',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
ax1.coastlines(linewidth=0.5)
#ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
ax1.add_feature(cfeature.OCEAN, color='lightgrey')
cmap='BrBG'
bwr = mpl.cm.get_cmap('bwr_r')
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),bwr(0.5),bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [-100, -50, -20, -1, 1, 20, 50, 100]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')
divider = make_axes_locatable(ax1)

h = ax1.pcolormesh(data_lon,data_lat,data_variable_E_IRR - data_variable_E_NOI, cmap=cmap_bias1, rasterized=True, norm=norm_bias1)
cbar   = f.colorbar(h, ax=ax1, cmap=cmap,
                               spacing='uniform',
                               orientation='horizontal',
                               extend='both', shrink = 1, pad = 0, aspect = 50)
cbar.set_label('$\Delta$ TSW ($\mathregular{kg/m^2}$)',fontsize = 12)
ticklabs = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(ticklabs, fontsize=12)
plt.title('Total soil water (IRR - NOI)')

#plt.show()

plt.savefig('C:\Research1\Figures\\Water.png')