import geopandas as gp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib import colors
import cartopy.crs as ccrs
import shapefile
import cartopy.feature as cfeature

listx1 = []
listy1 = []

listx2 = []
listy2 = []

listx3 = []
listy3 = []

listx4 = []
listy4 = []

listx5 = []
listy5 = []


IPCC_shapes = shapefile.Reader('IPCC-WGI-reference-regions-v4.shp')
IPCC_border = IPCC_shapes.shapes()
border_points = IPCC_border[6].points
for xNew, yNew in border_points:
    listx1.append(xNew)
    listy1.append(yNew)

border_points = IPCC_border[19].points
for xNew, yNew in border_points:
    listx2.append(xNew)
    listy2.append(yNew)

border_points = IPCC_border[35].points
for xNew, yNew in border_points:
    listx3.append(xNew)
    listy3.append(yNew)

border_points = IPCC_border[37].points
for xNew, yNew in border_points:
    listx4.append(xNew)
    listy4.append(yNew)

border_points = IPCC_border[38].points
for xNew, yNew in border_points:
    listx5.append(xNew)
    listy5.append(yNew)

border_points = IPCC_border[6].points
for xNew, yNew in border_points:
    listx1.append(xNew)
    listy1.append(yNew)

border_points = IPCC_border[6].points
for xNew, yNew in border_points:
    listx1.append(xNew)
    listy1.append(yNew)

border_points = IPCC_border[6].points
for xNew, yNew in border_points:
    listx1.append(xNew)
    listy1.append(yNew)

border_points = IPCC_border[6].points
for xNew, yNew in border_points:
    listx1.append(xNew)
    listy1.append(yNew)


xpoint = [8.7175, 140.0269, -96.4701]
ypoint = [45.0700, 36.0539, 41.1649]

# function definition
def subplotdefi(row, col, num, title,fontsize):
    ax = plt.subplot(row, col, num)
    plt.title(title,fontsize=fontsize)
    right_side1 = ax.spines['right']    # right_side1.set_visible(False)
    left_side1 = ax.spines['left']  # left_side1.set_visible(False)
    top_side1 = ax.spines['top']    # top_side1.set_visible(False)
    bottom_side1 = ax.spines['bottom']  # bottom_side1.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def plotmap_special(boundary, ax, colomn, cmap,  norm, legend_kwds):
    base = boundary.boundary.plot(ax=ax, edgecolor='lightgrey', linewidth=0.1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="7%", pad=0.1)
    h = data_geod.plot(column=colomn, cmap=cmap, ax=base, norm=norm, alpha=10, legend=True, cax=cax,
                       legend_kwds=legend_kwds,
                       missing_kwds={"color": "lightgray", "label": "Missing values"})    #missing_kwds={"color": "lightgrey", "edgecolor": "white", "label": "Missing values"})
    fig_Obs = h.figure
    cb_ax = fig_Obs.axes[1]
    cb_ax.tick_params(labelsize=18)

    return h

def plotmap(boundary, ax, colomn, cmap,  norm, legend_kwds):
    base = boundary.boundary.plot(ax=ax, edgecolor='lightgrey', linewidth=0.1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.1)
    h = data_geod.plot(column=colomn, cmap=cmap, ax=base, norm=norm, alpha=10, legend=True, cax=cax,
                       legend_kwds=legend_kwds,
                       missing_kwds={"color": "lightgray", "label": "Missing values"})    #missing_kwds={"color": "lightgrey", "edgecolor": "white", "label": "Missing values"})
    return h

def plotmap2(boundary, ax, colomn, cmap,  norm, legend_kwds):
    base = boundary.boundary.plot(ax=ax, edgecolor='lightgrey', linewidth=0.1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.1)
    h = data_geod.plot(column=colomn, cmap=cmap, ax=base, norm=norm, alpha=10, legend=True, cax=cax,
                       legend_kwds=legend_kwds,
                       missing_kwds={"color": "white", "label": "Missing values"})    #missing_kwds={"color": "lightgrey", "edgecolor": "white", "label": "Missing values"})
    return h


def set_plot_param():
    """Set my own customized plotting parameters"""

    mpl.rc('axes', edgecolor='lightgrey')
    mpl.rc('axes', labelcolor='dimgrey')
    mpl.rc('xtick', color='dimgrey')
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', color='dimgrey')
    mpl.rc('ytick', labelsize=12)
    mpl.rc('axes', titlesize=14)
    mpl.rc('axes', labelsize=12)
    mpl.rc('legend', fontsize='large')
    mpl.rc('text', color='dimgrey')

projection = ccrs.PlateCarree()
boundary = gp.GeoDataFrame.from_file('Irrgation_withdrawal.shp', encoding = 'gb18030')
data_geod = gp.GeoDataFrame.from_file('Irrgation_withdrawal.shp', encoding = 'gb18030')
# plot 1
f = plt.figure(figsize = (12, 7), dpi=1000)  # initiate the figure
f.subplots_adjust(hspace=0.1, wspace=0.1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
set_plot_param()
ax = subplotdefi(1,1,1,'Observed irrigation water withdrawal',18)
# plot water withdrawal
vmax = data_geod.Obs.max()
vmin = data_geod.Obs.min()
norm_withd = colors.DivergingNorm(vmin=vmin,
                             vcenter=0.01,
                             vmax=vmax)
cmap_withd='Blues'



bounds_Blues = [0, 0.5, 1, 2, 5, 10, 15, 30]
Blues = mpl.cm.get_cmap('Blues')
colors_Blues = [Blues(0.125), Blues(0.25), Blues(0.375), Blues(0.5), Blues(0.625), Blues(0.75), Blues(0.875), Blues(0.99)]

print(colors_Blues)
cmap_Blues = mpl.colors.ListedColormap(colors_Blues)
norm_Blues = mpl.colors.BoundaryNorm(bounds_Blues,cmap_Blues.N,extend='max')
h = plotmap_special(boundary, ax, 'Obs', cmap_Blues, norm_Blues,legend_kwds={'label': r"water withdrawal ($\mathregular{km^3/yr}$)",'orientation': "horizontal"})

plt.savefig('C:\Research1\Figures\\ObsIWW.png')
#plt.show()

#plot 2
f2 = plt.figure(figsize = (6, 7), dpi=1000)  # initiate the figure
f2.subplots_adjust(hspace=0.1, wspace=0.1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
ax = subplotdefi(2,1,1,'Simulated irrigation water withdrawal (CTL)',14)
ax.text(0.01,0.92,'a',color='dimgrey',fontsize=12, transform=ax.transAxes, weight = 'bold')

plt.plot(listx1,listy1, color='black')
ax.text(-136,16,'NCA',color='black',fontsize=10, weight = 'bold')
plt.plot(listx2,listy2, color='black')
ax.text(-38,30,'MED',color='black',fontsize=10, weight = 'bold')
plt.plot(listx3,listy3, color='black')
ax.text(50,7,'SAS',color='black',fontsize=10, weight = 'bold')
plt.plot(listx4,listy4, color='black')
ax.text(142,25.5,'EAS',color='black',fontsize=10, weight = 'bold')
plt.plot(listx5,listy5, color='black')
ax.text(93,-20,'SEA',color='black',fontsize=10, weight = 'bold')


ax1 = subplotdefi(2,1,2,'Bias (CTL - OBS)',14)
ax1.text(0.01,0.92,'b',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')

plt.scatter(xpoint,ypoint, color='black', s=5, marker='^' ,zorder=10)
ax1.text(-96,41,'NEB',color='black',fontsize=8, weight = 'bold')
#ax1.text(127,39,'CHE',color='black',fontsize=8, weight = 'bold')
ax1.text(141.5,29.5,'MAS',color='black',fontsize=8, weight = 'bold')
ax1.text(8.7,46,'CAS',color='black',fontsize=8, weight = 'bold')
#ax1.text(122,14,'RIF',color='black',fontsize=8, weight = 'bold')
h2 = plotmap2(boundary, ax, 'Ctl', cmap_Blues, norm_Blues,legend_kwds={'label': r"water withdrawal ($\mathregular{km^3/yr}$)",'orientation': "horizontal"})


PRGn = mpl.cm.get_cmap('PRGn_r')
colors = [PRGn(0.01),PRGn(0.1),PRGn(0.2),PRGn(0.3),PRGn(0.5),PRGn(0.7),PRGn(0.8),PRGn(0.9),PRGn(0.99)]
colors = [PRGn(0.01),PRGn(0.1),PRGn(0.2),PRGn(0.3),'whitesmoke',PRGn(0.7),PRGn(0.8),PRGn(0.9),PRGn(0.99)]
#colors = ['darkgreen', 'forestgreen', 'limegreen', 'lime', 'white', 'plum', 'violet', 'fuchsia', 'purple']

cmap_bias = mpl.colors.ListedColormap(colors)
bounds = [-15, -10, -5, -0.001, 0.001, 5, 10, 15]
bounds = [-5, -3, -1, -0.1, 0.1, 1, 3, 5]
norm_bias = mpl.colors.BoundaryNorm(bounds,cmap_bias.N,extend='both')

bwr = mpl.cm.get_cmap('bwr_r')
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),bwr(0.5),bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
colors = [bwr(0.1),bwr(0.2),bwr(0.3),bwr(0.4),'whitesmoke',bwr(0.6),bwr(0.7),bwr(0.8),bwr(0.9)]
#colors = ['darkgoldenrod', 'goldenrod', 'gold', 'yellow', 'white', 'aqua', 'dodgerblue', 'blue', 'darkblue']
cmap_bias1 = mpl.colors.ListedColormap(colors)
bounds = [-15, -10, -5, -0.001, 0.001, 5, 10, 15]
bounds = [-10, -5, -2, -0.5, 0.5, 2, 5, 10]
norm_bias1 = mpl.colors.BoundaryNorm(bounds,cmap_bias1.N,extend='both')

h2 = plotmap(boundary, ax1, 'Bias_ctl', cmap_bias1, norm_bias1,legend_kwds={'label': r"mean bias ($\mathregular{km^3/yr}$)",'orientation': "horizontal"})
plt.savefig('C:\Research1\Figures\\CTL_Obs.png')

#plot 3

f2 = plt.figure(figsize = (12, 7), dpi=1000)  # initiate the figure
f2.subplots_adjust(hspace=0.1, wspace=0.1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
#ax1 = subplotdefi(2,2,1,'Mean error (IRR_meth - OBS)',14)
ax1 = subplotdefi(2,2,1,'',14)
ax1.set_title('IRR_0 - OBS',loc='right',fontsize = 12)
ax1.text(0.01,0.92,'a',color='dimgrey',fontsize=12, transform=ax1.transAxes, weight = 'bold')
#ax2 = subplotdefi(2,2,2,'Mean error (IRR_drai - OBS)',14)
ax2 = subplotdefi(2,2,2,'',14)
ax2.set_title('IRR_1 - OBS',loc='right',fontsize = 12)
ax2.text(0.01,0.92,'b',color='dimgrey',fontsize=12, transform=ax2.transAxes, weight = 'bold')
#ax3 = subplotdefi(2,2,3,'Mean error (IRR_pool - OBS)',14)
ax3 = subplotdefi(2,2,3,'',14)
ax3.set_title('IRR_2 - OBS',loc='right',fontsize = 12)
ax3.text(0.01,0.92,'c',color='dimgrey',fontsize=12, transform=ax3.transAxes, weight = 'bold')
#ax4 = subplotdefi(2,2,4,'Mean error (IRR_satu - OBS)',14)
ax4 = subplotdefi(2,2,4,'',14)
ax4.set_title('IRR - OBS',loc='right',fontsize = 12)
ax4.text(0.01,0.92,'d',color='dimgrey',fontsize=12, transform=ax4.transAxes, weight = 'bold')

h1 = plotmap(boundary, ax1, 'Bias_irr', cmap_bias1, norm_bias1,legend_kwds={'label': r"mean bias ($\mathregular{km^3/yr}$)",'orientation': "horizontal"})
h2 = plotmap(boundary, ax2, 'Bias_drai', cmap_bias1, norm_bias1,legend_kwds={'label': r"mean bias ($\mathregular{km^3/yr}$)",'orientation': "horizontal"})
h3 = plotmap(boundary, ax3, 'Bias_pool', cmap_bias1, norm_bias1,legend_kwds={'label': r"mean bias ($\mathregular{km^3/yr}$)",'orientation': "horizontal"})
h4 = plotmap(boundary, ax4, 'Bias_satu', cmap_bias1, norm_bias1,legend_kwds={'label': r"mean bias ($\mathregular{km^3/yr}$)",'orientation': "horizontal"})
plt.savefig('C:\Research1\Figures\\IRR_OBS.png')
#plt.show()