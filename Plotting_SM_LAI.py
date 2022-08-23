import netCDF4 as nc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def set_plot_param():
    """Set my own customized plotting parameters"""

    mpl.rc('axes', edgecolor='dimgrey')
    mpl.rc('axes', labelcolor='dimgrey')
    mpl.rc('xtick', color='dimgrey')
    mpl.rc('ytick', color='dimgrey')
    mpl.rc('legend', fontsize='large')
    mpl.rc('text', color='dimgrey')

def calcu_SM(str_ncFile, time_steps):
    file_SM = nc.Dataset(str_ncFile)
    data_SM = file_SM.variables['H2OSOI']
    var_SM = np.array(data_SM)
    plot_SM = np.zeros(time_steps)
    for i in range(time_steps):
        plot_SM[i] = (var_SM[i,0,0] * 0.02 + var_SM[i,1,0] * 0.04) / 0.06
    return plot_SM

def calcu_LAI(str_ncFile, time_steps):
    file_LAI = nc.Dataset(str_ncFile)
    data_LAI = file_LAI.variables['TLAI']
    var_LAI = np.array(data_LAI)
    plot_LAI = np.zeros(time_steps)
    for i in range(time_steps):
        plot_LAI[i] = var_LAI[i,0]
    return plot_LAI

def plot_SM_LAI(time_steps, plot_SM_1, plot_SM_2, plot_SM_3, plot_LAI, ylim_min1, ylim_max1, ylim_min2, ylim_max2, title, xtick_value, xtick_name):
    x = range(time_steps)
    plt.plot(x, plot_SM_1, 'green', label='IRR',
             linewidth=1.5, markersize=3, alpha=0.8, marker='v')
    plt.plot(x, plot_SM_2, 'blue', label='CTL',
             linewidth=1.5, markersize=3, alpha=0.8, marker='>')
    plt.plot(x, plot_SM_3, 'red', label='NOI',
             linewidth=1.5, markersize=3, alpha=0.8, marker='^')
    plt.xlabel('Date', fontsize=20)
    plt.xticks(fontsize=15)
    plt.ylabel('SM $\mathregular{(mm^3/mm^3)}$', fontsize=20)
    plt.yticks(fontsize=15)
    plt.legend(loc='lower left', fontsize=20)
    plt.ylim(ylim_min1, ylim_max1)
    ax2 = ax1.twinx()
    plt.fill_between(x, 0, plot_LAI, color='darkgreen', alpha=0.3, label='LAI')
    plt.ylabel('LAI $\mathregular{(m^2/m^2)}$', fontsize=20)
    plt.yticks(fontsize=15)
    plt.legend(loc='lower right', fontsize=20)
    plt.xlim(0, time_steps - 1)
    plt.ylim(ylim_min2, ylim_max2)
    plt.title(title, fontsize=30, loc='right')
    plt.xticks(xtick_value,
               xtick_name,
               )
    plt.tick_params(labelsize=15)


f = plt.figure(figsize = (16, 18), dpi=1000)
set_plot_param()
plot_SM_noirr = calcu_SM('C:\Research1\Final_data\\Ne1_noirr_SpGs.clm2.h1.2005-2007_H2OSOI.nc', 1095)
plot_SM_drip = calcu_SM('C:\Research1\Final_data\\Ne1_drip_SpGs.clm2.h1.2005-2007_H2OSOI.nc', 1095)
plot_SM_flood = calcu_SM('C:\Research1\Final_data\\Ne1_sprinkler_SpGs.clm2.h1.2005-2007_H2OSOI.nc', 1095)

plot_LAI_noirr = calcu_LAI('C:\Research1\Final_data\\Ne1_noirr_SpGs.clm2.h1.2005-2007_TLAI.nc', 1095)
plot_LAI_drip = calcu_LAI('C:\Research1\Final_data\\Ne1_drip_SpGs.clm2.h1.2005-2007_TLAI.nc', 1095)
plot_LAI_flood = calcu_LAI('C:\Research1\Final_data\\Ne1_sprinkler_SpGs.clm2.h1.2005-2007_TLAI.nc', 1095)

ax1 = plt.subplot(3,1,1)
xtick_value_neb = [0+31, 120+31, 243+31, 365+31, 485+31, 618+31, 730+31, 850+31, 973+31]
xtick_name_neb = ['2005-02', '2005-06', '2005-10', '2006-02', '2006-06', '2006-10', '2007-02', '2007-06', '2007-10']
plot_SM_LAI(1095, plot_SM_flood, plot_SM_drip, plot_SM_noirr, plot_LAI_flood, 0, 0.6, 0, 2.5, 'NEB',xtick_value_neb,xtick_name_neb)
plt.title('a', fontsize=30, loc='left')
plot_SM_noirr = calcu_SM('C:\Research1\Final_data\Castellaro_noirr_SpGs.clm2.h1.2009-2010_H2OSOI.nc', 730)
plot_SM_drip = calcu_SM('C:\Research1\Final_data\Castellaro_drip_SpGs.clm2.h1.2009-2010_H2OSOI.nc', 730)
plot_SM_flood = calcu_SM('C:\Research1\Final_data\Castellaro_flood_SpGs.clm2.h1.2009-2010_H2OSOI.nc', 730)

plot_LAI_noirr = calcu_LAI('C:\Research1\Final_data\Castellaro_noirr_SpGs.clm2.h1.2009-2010_TLAI.nc', 730)
plot_LAI_drip = calcu_LAI('C:\Research1\Final_data\Castellaro_drip_SpGs.clm2.h1.2009-2010_TLAI.nc', 730)
plot_LAI_flood = calcu_LAI('C:\Research1\Final_data\Castellaro_flood_SpGs.clm2.h1.2009-2010_TLAI.nc', 730)

ax1 = plt.subplot(3,1,2)
xtick_value_cas = [0+31, 120+31, 243+31, 365+31, 485+31, 618+31]
xtick_name_cas = ['2009 -02', '2009-06', '2009-10', '2010-02', '2010-06', '2010-10']
plot_SM_LAI(730, plot_SM_flood, plot_SM_drip, plot_SM_noirr, plot_LAI_flood, 0, 0.6, 0, 2.5, 'CAS', xtick_value_cas, xtick_name_cas)
plt.title('b', fontsize=30, loc='left')
plot_SM_noirr = calcu_SM('C:\Research1\Final_data\Japan_noirr_SpGs.clm2.h1.2012-01-01-00000_H2OSOI.nc', 365)
plot_SM_drip = calcu_SM('C:\Research1\Final_data\Japan_drip_SpGs.clm2.h1.2012-01-01-00000_H2OSOI.nc', 365)
plot_SM_flood = calcu_SM('C:\Research1\Final_data\Japan_flood_SpGs.clm2.h1.2012-01-01-00000_H2OSOI.nc', 365)

plot_LAI_noirr = calcu_LAI('C:\Research1\Final_data\Japan_noirr_SpGs.clm2.h1.2012-01-01-00000_TLAI.nc', 365)
plot_LAI_drip = calcu_LAI('C:\Research1\Final_data\Japan_drip_SpGs.clm2.h1.2012-01-01-00000_TLAI.nc', 365)
plot_LAI_flood = calcu_LAI('C:\Research1\Final_data\Japan_flood_SpGs.clm2.h1.2012-01-01-00000_TLAI.nc', 365)

ax1 = plt.subplot(3,1,3)
xtick_value_mas = [0+31, 120+31, 243+31]
xtick_name_mas = ['2012-02', '2012-06', '2012-10']
plot_SM_LAI(365, plot_SM_flood, plot_SM_drip, plot_SM_noirr, plot_LAI_flood, 0, 0.7, 0, 3.5, 'MAS', xtick_value_mas, xtick_name_mas)
plt.title('c', fontsize=30, loc='left')
plt.subplots_adjust(left=0.1,
                    right=0.9,
                    top=0.9,
                    bottom=0.1,
                    wspace=0.4,
                    hspace=0.5)


plt.savefig('C:\Research1\Figures\\SM_LAI.png')
plt.show()
