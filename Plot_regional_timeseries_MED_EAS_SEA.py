import scipy.io as scio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def set_plot_param():
    """Set my own customized plotting parameters"""

    mpl.rc('axes', edgecolor='dimgrey')
    mpl.rc('axes', labelcolor='dimgrey')
    mpl.rc('xtick', color='dimgrey')
    mpl.rc('xtick', labelsize=10)
    mpl.rc('ytick', color='dimgrey')
    mpl.rc('ytick', labelsize=10)
    mpl.rc('axes', titlesize=10)
    mpl.rc('axes', labelsize=10)
    mpl.rc('legend', fontsize='large')
    mpl.rc('text', color='dimgrey')

str_data_ctl = 'CTL_final.mat'
dic_data_ctl = scio.loadmat(str_data_ctl)
data_ctl = dic_data_ctl['sum_ctl']

str_data_irr = 'IRR_final.mat'
dic_data_irr = scio.loadmat(str_data_irr)
data_irr = dic_data_irr['sum_irr']

str_data_noi = 'NOI_final.mat'
dic_data_noi = scio.loadmat(str_data_noi)
data_noi = dic_data_noi['sum_noi']


# GLO
WNA_CTL_EFLX_LH_TOT = data_ctl[0, :, 0]
WNA_CTL_FSH = data_ctl[0, :, 1]
WNA_CTL_LWup = data_ctl[0, :, 2]
WNA_CTL_SWup = data_ctl[0, :, 3]
WNA_CTL_Qle = data_ctl[0, :, 4]  / 28.94
WNA_CTL_QVEGT = data_ctl[0, :, 5]  * 86400
WNA_CTL_QRUNOFF = data_ctl[0, :, 6]  * 86400
WNA_CTL_TOTSOILLIQ = data_ctl[0, :, 7]

WNA_IRR_EFLX_LH_TOT = data_irr[0, :, 0]
WNA_IRR_FSH = data_irr[0, :, 1]
WNA_IRR_LWup = data_irr[0, :, 2]
WNA_IRR_SWup = data_irr[0, :, 3]
WNA_IRR_Qle = data_irr[0, :, 4]  / 28.94
WNA_IRR_QVEGT = data_irr[0, :, 5]  * 86400
WNA_IRR_QRUNOFF = data_irr[0, :, 6]  * 86400
WNA_IRR_TOTSOILLIQ = data_irr[0, :, 7]

WNA_NOI_EFLX_LH_TOT = data_noi[0, :, 0]
WNA_NOI_FSH = data_noi[0, :, 1]
WNA_NOI_LWup = data_noi[0, :, 2]
WNA_NOI_SWup = data_noi[0, :, 3]
WNA_NOI_Qle = data_noi[0, :, 4]  / 28.94
WNA_NOI_QVEGT = data_noi[0, :, 5]  * 86400
WNA_NOI_QRUNOFF = data_noi[0, :, 6]  * 86400
WNA_NOI_TOTSOILLIQ = data_noi[0, :, 7]

def plot_var(ax1, x, var_IRR, var_CTL, var_NOI,legloc, ylabel,title, num,region):
    set_plot_param()
    plt.plot(x, var_IRR, 'green', marker='v', label='IRR', linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, var_CTL, 'blue', marker='>', label='CTL', linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, var_NOI, 'red', marker='^', label='NOI', linewidth=0.8, markersize=5, alpha=0.5)
    plt.legend(loc=legloc, frameon=False, fontsize=12)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
               [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
               fontsize=8)
    plt.title(title, fontsize=14,loc='right')
    plt.title(region, fontsize=14, loc='left')
    plt.tick_params(labelsize=14)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=14, transform=ax1.transAxes, weight='bold')

def plot_var_nolegend(ax1, x, var_IRR, var_CTL, var_NOI,legloc, ylabel,title, num,region):
    set_plot_param()
    plt.plot(x, var_IRR, 'green', marker='v', label='IRR', linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, var_CTL, 'blue', marker='>', label='CTL', linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, var_NOI, 'red', marker='^', label='NOI', linewidth=0.8, markersize=5, alpha=0.5)

    plt.ylabel(ylabel, fontsize=14)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
               [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
               fontsize=8)
    plt.title(title, fontsize=14,loc='right')
    plt.title(region, fontsize=14, loc='left')
    plt.tick_params(labelsize=14)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=14, transform=ax1.transAxes, weight='bold')

x = range(1, 13)
f = plt.figure(figsize = (16, 10), dpi=1000)
set_plot_param()
f.subplots_adjust(hspace=0.4, wspace=0.4, left = 0.07, right = 0.95, top = 0.95, bottom = 0.05)

# MED
WNA_CTL_EFLX_LH_TOT = data_ctl[20, :, 0]
WNA_CTL_FSH = data_ctl[20, :, 1]
WNA_CTL_LWup = data_ctl[20, :, 2]
WNA_CTL_SWup = data_ctl[20, :, 3]
WNA_CTL_Qle = data_ctl[20, :, 4]  / 28.94
WNA_CTL_QVEGT = data_ctl[20, :, 5]  * 86400
WNA_CTL_QRUNOFF = data_ctl[20, :, 6]  * 86400
WNA_CTL_TOTSOILLIQ = data_ctl[20, :, 7]

WNA_IRR_EFLX_LH_TOT = data_irr[20, :, 0]
WNA_IRR_FSH = data_irr[20, :, 1]
WNA_IRR_LWup = data_irr[20, :, 2]
WNA_IRR_SWup = data_irr[20, :, 3]
WNA_IRR_Qle = data_irr[20, :, 4]  / 28.94
WNA_IRR_QVEGT = data_irr[20, :, 5]  * 86400
WNA_IRR_QRUNOFF = data_irr[20, :, 6]  * 86400
WNA_IRR_TOTSOILLIQ = data_irr[20, :, 7]

WNA_NOI_EFLX_LH_TOT = data_noi[20, :, 0]
WNA_NOI_FSH = data_noi[20, :, 1]
WNA_NOI_LWup = data_noi[20, :, 2]
WNA_NOI_SWup = data_noi[20, :, 3]
WNA_NOI_Qle = data_noi[20, :, 4]  / 28.94
WNA_NOI_QVEGT = data_noi[20, :, 5]  * 86400
WNA_NOI_QRUNOFF = data_noi[20, :, 6]  * 86400
WNA_NOI_TOTSOILLIQ = data_noi[20, :, 7]

x = range(1, 13)
ax1 = plt.subplot(3, 4, 1)
plot_var(ax1, x, WNA_IRR_EFLX_LH_TOT, WNA_CTL_EFLX_LH_TOT, WNA_NOI_EFLX_LH_TOT, 'upper left', r'LHF ($\mathregular{W/m^2}$)', 'LHF', 'a', 'MED')

ax1 = plt.subplot(3, 4, 2)
plot_var_nolegend(ax1, x, WNA_IRR_FSH, WNA_CTL_FSH, WNA_NOI_FSH, 'upper left', r'SHF ($\mathregular{W/m^2}$)', 'SHF', 'b', 'MED')

ax1 = plt.subplot(3, 4, 3)
plot_var_nolegend(ax1, x, WNA_IRR_LWup, WNA_CTL_LWup, WNA_NOI_LWup, 'upper left', r'LWup ($\mathregular{W/m^2}$)', 'LWup', 'c', 'MED')

ax1 = plt.subplot(3, 4, 4)
plot_var_nolegend(ax1, x, WNA_IRR_SWup, WNA_CTL_SWup, WNA_NOI_SWup, 'upper left', r'SWup ($\mathregular{W/m^2}$)', 'SWup', 'd', 'MED')




# EAS
WNA_CTL_EFLX_LH_TOT = data_ctl[36, :, 0]
WNA_CTL_FSH = data_ctl[36, :, 1]
WNA_CTL_LWup = data_ctl[36, :, 2]
WNA_CTL_SWup = data_ctl[36, :, 3]
WNA_CTL_Qle = data_ctl[36, :, 4]  / 28.94
WNA_CTL_QVEGT = data_ctl[36, :, 5]  * 86400
WNA_CTL_QRUNOFF = data_ctl[36, :, 6]  * 86400
WNA_CTL_TOTSOILLIQ = data_ctl[36, :, 7]

WNA_IRR_EFLX_LH_TOT = data_irr[36, :, 0]
WNA_IRR_FSH = data_irr[36, :, 1]
WNA_IRR_LWup = data_irr[36, :, 2]
WNA_IRR_SWup = data_irr[36, :, 3]
WNA_IRR_Qle = data_irr[36, :, 4]  / 28.94
WNA_IRR_QVEGT = data_irr[36, :, 5]  * 86400
WNA_IRR_QRUNOFF = data_irr[36, :, 6]  * 86400
WNA_IRR_TOTSOILLIQ = data_irr[36, :, 7]

WNA_NOI_EFLX_LH_TOT = data_noi[36, :, 0]
WNA_NOI_FSH = data_noi[36, :, 1]
WNA_NOI_LWup = data_noi[36, :, 2]
WNA_NOI_SWup = data_noi[36, :, 3]
WNA_NOI_Qle = data_noi[36, :, 4]  / 28.94
WNA_NOI_QVEGT = data_noi[36, :, 5]  * 86400
WNA_NOI_QRUNOFF = data_noi[36, :, 6]  * 86400
WNA_NOI_TOTSOILLIQ = data_noi[36, :, 7]

x = range(1, 13)
ax1 = plt.subplot(3, 4, 5)
plot_var_nolegend(ax1, x, WNA_IRR_EFLX_LH_TOT, WNA_CTL_EFLX_LH_TOT, WNA_NOI_EFLX_LH_TOT, 'upper left', r'LHF ($\mathregular{W/m^2}$)', 'LHF', 'e', 'EAS')

ax1 = plt.subplot(3, 4, 6)
plot_var_nolegend(ax1, x, WNA_IRR_FSH, WNA_CTL_FSH, WNA_NOI_FSH, 'upper left', r'SHF ($\mathregular{W/m^2}$)', 'SHF', 'f', 'EAS')

ax1 = plt.subplot(3, 4, 7)
plot_var_nolegend(ax1, x, WNA_IRR_LWup, WNA_CTL_LWup, WNA_NOI_LWup, 'upper left', r'LWup ($\mathregular{W/m^2}$)', 'LWup', 'g', 'EAS')

ax1 = plt.subplot(3, 4, 8)
plot_var_nolegend(ax1, x, WNA_IRR_SWup, WNA_CTL_SWup, WNA_NOI_SWup, 'upper left', r'SWup ($\mathregular{W/m^2}$)', 'SWup', 'h', 'EAS')


# SEA
WNA_CTL_EFLX_LH_TOT = data_ctl[39, :, 0]
WNA_CTL_FSH = data_ctl[39, :, 1]
WNA_CTL_LWup = data_ctl[39, :, 2]
WNA_CTL_SWup = data_ctl[39, :, 3]
WNA_CTL_Qle = data_ctl[39, :, 4]  / 28.94
WNA_CTL_QVEGT = data_ctl[39, :, 5]  * 86400
WNA_CTL_QRUNOFF = data_ctl[39, :, 6]  * 86400
WNA_CTL_TOTSOILLIQ = data_ctl[39, :, 7]

WNA_IRR_EFLX_LH_TOT = data_irr[39, :, 0]
WNA_IRR_FSH = data_irr[39, :, 1]
WNA_IRR_LWup = data_irr[39, :, 2]
WNA_IRR_SWup = data_irr[39, :, 3]
WNA_IRR_Qle = data_irr[39, :, 4]  / 28.94
WNA_IRR_QVEGT = data_irr[39, :, 5]  * 86400
WNA_IRR_QRUNOFF = data_irr[39, :, 6]  * 86400
WNA_IRR_TOTSOILLIQ = data_irr[39, :, 7]

WNA_NOI_EFLX_LH_TOT = data_noi[39, :, 0]
WNA_NOI_FSH = data_noi[39, :, 1]
WNA_NOI_LWup = data_noi[39, :, 2]
WNA_NOI_SWup = data_noi[39, :, 3]
WNA_NOI_Qle = data_noi[39, :, 4]  / 28.94
WNA_NOI_QVEGT = data_noi[39, :, 5]  * 86400
WNA_NOI_QRUNOFF = data_noi[39, :, 6]  * 86400
WNA_NOI_TOTSOILLIQ = data_noi[39, :, 7]

x = range(1, 13)
ax1 = plt.subplot(3, 4, 9)
plot_var_nolegend(ax1, x, WNA_IRR_EFLX_LH_TOT, WNA_CTL_EFLX_LH_TOT, WNA_NOI_EFLX_LH_TOT, 'upper left', r'LHF ($\mathregular{W/m^2}$)', 'LHF', 'i', 'SEA')

ax1 = plt.subplot(3, 4, 10)
plot_var_nolegend(ax1, x, WNA_IRR_FSH, WNA_CTL_FSH, WNA_NOI_FSH, 'upper left', r'SHF ($\mathregular{W/m^2}$)', 'SHF', 'j', 'SEA')

ax1 = plt.subplot(3, 4, 11)
plot_var_nolegend(ax1, x, WNA_IRR_LWup, WNA_CTL_LWup, WNA_NOI_LWup, 'upper left', r'LWup ($\mathregular{W/m^2}$)', 'LWup', 'k', 'SEA')

ax1 = plt.subplot(3, 4, 12)
plot_var_nolegend(ax1, x, WNA_IRR_SWup, WNA_CTL_SWup, WNA_NOI_SWup, 'upper left', r'SWup ($\mathregular{W/m^2}$)', 'SWup', 'l', 'SEA')

plt.subplots_adjust(left=0.1,
                    right=0.9,
                    top=0.9,
                    bottom=0.1,
                    wspace=0.35,
                    hspace=0.3)

plt.savefig('Timeseries_energy2.png')


str_data_ctl = 'CTL_final.mat'
dic_data_ctl = scio.loadmat(str_data_ctl)
data_ctl = dic_data_ctl['sum_ctl']

str_data_irr = 'IRR_final.mat'
dic_data_irr = scio.loadmat(str_data_irr)
data_irr = dic_data_irr['sum_irr']

str_data_noi = 'NOI_final.mat'
dic_data_noi = scio.loadmat(str_data_noi)
data_noi = dic_data_noi['sum_noi']

x = range(1, 13)
f = plt.figure(figsize = (16, 10), dpi=100)
set_plot_param()
f.subplots_adjust(hspace=0.4, wspace=0.4, left = 0.07, right = 0.95, top = 0.95, bottom = 0.05)

# MED
WNA_CTL_EFLX_LH_TOT = data_ctl[20, :, 0]
WNA_CTL_FSH = data_ctl[20, :, 1]
WNA_CTL_LWup = data_ctl[20, :, 2]
WNA_CTL_SWup = data_ctl[20, :, 3]
WNA_CTL_Qle = data_ctl[20, :, 4]  / 28.94
WNA_CTL_QVEGT = data_ctl[20, :, 5]  * 86400
WNA_CTL_QRUNOFF = data_ctl[20, :, 6]  * 86400
WNA_CTL_TOTSOILLIQ = data_ctl[20, :, 7]

WNA_IRR_EFLX_LH_TOT = data_irr[20, :, 0]
WNA_IRR_FSH = data_irr[20, :, 1]
WNA_IRR_LWup = data_irr[20, :, 2]
WNA_IRR_SWup = data_irr[20, :, 3]
WNA_IRR_Qle = data_irr[20, :, 4]  / 28.94
WNA_IRR_QVEGT = data_irr[20, :, 5]  * 86400
WNA_IRR_QRUNOFF = data_irr[20, :, 6]  * 86400
WNA_IRR_TOTSOILLIQ = data_irr[20, :, 7]

WNA_NOI_EFLX_LH_TOT = data_noi[20, :, 0]
WNA_NOI_FSH = data_noi[20, :, 1]
WNA_NOI_LWup = data_noi[20, :, 2]
WNA_NOI_SWup = data_noi[20, :, 3]
WNA_NOI_Qle = data_noi[20, :, 4]  / 28.94
WNA_NOI_QVEGT = data_noi[20, :, 5]  * 86400
WNA_NOI_QRUNOFF = data_noi[20, :, 6]  * 86400
WNA_NOI_TOTSOILLIQ = data_noi[20, :, 7]

x = range(1, 13)
ax1 = plt.subplot(3, 4, 1)
plot_var(ax1, x, WNA_IRR_Qle, WNA_CTL_Qle, WNA_NOI_Qle, 'upper left', r'E ($\mathregular{mm/day}$)', 'E', 'a', 'MED')

ax1 = plt.subplot(3, 4, 2)
plot_var_nolegend(ax1, x, WNA_IRR_QVEGT, WNA_CTL_QVEGT, WNA_NOI_QVEGT, 'upper left', r'TR ($\mathregular{mm/day}$)', 'TR', 'b', 'MED')

ax1 = plt.subplot(3, 4, 3)
plot_var_nolegend(ax1, x, WNA_IRR_QRUNOFF, WNA_CTL_QRUNOFF, WNA_NOI_QRUNOFF, 'upper left', r'R ($\mathregular{mm/day}$)', 'R', 'c', 'MED')

ax1 = plt.subplot(3, 4, 4)
plot_var_nolegend(ax1, x, WNA_IRR_TOTSOILLIQ, WNA_CTL_TOTSOILLIQ, WNA_NOI_TOTSOILLIQ, 'upper left', r'TSW ($\mathregular{kg/m^2}$)', 'TSW', 'd', 'MED')

# EAS
WNA_CTL_EFLX_LH_TOT = data_ctl[36, :, 0]
WNA_CTL_FSH = data_ctl[36, :, 1]
WNA_CTL_LWup = data_ctl[36, :, 2]
WNA_CTL_SWup = data_ctl[36, :, 3]
WNA_CTL_Qle = data_ctl[36, :, 4]  / 28.94
WNA_CTL_QVEGT = data_ctl[36, :, 5]  * 86400
WNA_CTL_QRUNOFF = data_ctl[36, :, 6]  * 86400
WNA_CTL_TOTSOILLIQ = data_ctl[36, :, 7]

WNA_IRR_EFLX_LH_TOT = data_irr[36, :, 0]
WNA_IRR_FSH = data_irr[36, :, 1]
WNA_IRR_LWup = data_irr[36, :, 2]
WNA_IRR_SWup = data_irr[36, :, 3]
WNA_IRR_Qle = data_irr[36, :, 4]  / 28.94
WNA_IRR_QVEGT = data_irr[36, :, 5]  * 86400
WNA_IRR_QRUNOFF = data_irr[36, :, 6]  * 86400
WNA_IRR_TOTSOILLIQ = data_irr[36, :, 7]

WNA_NOI_EFLX_LH_TOT = data_noi[36, :, 0]
WNA_NOI_FSH = data_noi[36, :, 1]
WNA_NOI_LWup = data_noi[36, :, 2]
WNA_NOI_SWup = data_noi[36, :, 3]
WNA_NOI_Qle = data_noi[36, :, 4]  / 28.94
WNA_NOI_QVEGT = data_noi[36, :, 5]  * 86400
WNA_NOI_QRUNOFF = data_noi[36, :, 6]  * 86400
WNA_NOI_TOTSOILLIQ = data_noi[36, :, 7]

x = range(1, 13)
ax1 = plt.subplot(3, 4, 5)
plot_var_nolegend(ax1, x, WNA_IRR_Qle, WNA_CTL_Qle, WNA_NOI_Qle, 'upper left', r'E ($\mathregular{mm/day}$)', 'E', 'e', 'EAS')

ax1 = plt.subplot(3, 4, 6)
plot_var_nolegend(ax1, x, WNA_IRR_QVEGT, WNA_CTL_QVEGT, WNA_NOI_QVEGT, 'upper left', r'TR ($\mathregular{mm/day}$)', 'TR', 'f', 'EAS')

ax1 = plt.subplot(3, 4, 7)
plot_var_nolegend(ax1, x, WNA_IRR_QRUNOFF, WNA_CTL_QRUNOFF, WNA_NOI_QRUNOFF, 'upper left', r'R ($\mathregular{mm/day}$)', 'R', 'g', 'EAS')

ax1 = plt.subplot(3, 4, 8)
plot_var_nolegend(ax1, x, WNA_IRR_TOTSOILLIQ, WNA_CTL_TOTSOILLIQ, WNA_NOI_TOTSOILLIQ, 'upper left', r'TSW ($\mathregular{kg/m^2}$)', 'TSW', 'h', 'EAS')

# SEA
WNA_CTL_EFLX_LH_TOT = data_ctl[39, :, 0]
WNA_CTL_FSH = data_ctl[39, :, 1]
WNA_CTL_LWup = data_ctl[39, :, 2]
WNA_CTL_SWup = data_ctl[39, :, 3]
WNA_CTL_Qle = data_ctl[39, :, 4]  / 28.94
WNA_CTL_QVEGT = data_ctl[39, :, 5]  * 86400
WNA_CTL_QRUNOFF = data_ctl[39, :, 6]  * 86400
WNA_CTL_TOTSOILLIQ = data_ctl[39, :, 7]

WNA_IRR_EFLX_LH_TOT = data_irr[39, :, 0]
WNA_IRR_FSH = data_irr[39, :, 1]
WNA_IRR_LWup = data_irr[39, :, 2]
WNA_IRR_SWup = data_irr[39, :, 3]
WNA_IRR_Qle = data_irr[39, :, 4]  / 28.94
WNA_IRR_QVEGT = data_irr[39, :, 5]  * 86400
WNA_IRR_QRUNOFF = data_irr[39, :, 6]  * 86400
WNA_IRR_TOTSOILLIQ = data_irr[39, :, 7]

WNA_NOI_EFLX_LH_TOT = data_noi[39, :, 0]
WNA_NOI_FSH = data_noi[39, :, 1]
WNA_NOI_LWup = data_noi[39, :, 2]
WNA_NOI_SWup = data_noi[39, :, 3]
WNA_NOI_Qle = data_noi[39, :, 4]  / 28.94
WNA_NOI_QVEGT = data_noi[39, :, 5]  * 86400
WNA_NOI_QRUNOFF = data_noi[39, :, 6]  * 86400
WNA_NOI_TOTSOILLIQ = data_noi[39, :, 7]

x = range(1, 13)
ax1 = plt.subplot(3, 4, 9)
plot_var_nolegend(ax1, x, WNA_IRR_Qle, WNA_CTL_Qle, WNA_NOI_Qle, 'upper left', r'E ($\mathregular{mm/day}$)', 'E', 'i', 'SEA')

ax1 = plt.subplot(3, 4, 10)
plot_var_nolegend(ax1, x, WNA_IRR_QVEGT, WNA_CTL_QVEGT, WNA_NOI_QVEGT, 'upper left', r'TR ($\mathregular{mm/day}$)', 'TR', 'j', 'SEA')

ax1 = plt.subplot(3, 4, 11)
plot_var_nolegend(ax1, x, WNA_IRR_QRUNOFF, WNA_CTL_QRUNOFF, WNA_NOI_QRUNOFF, 'upper left', r'R ($\mathregular{mm/day}$)', 'R', 'k', 'SEA')

ax1 = plt.subplot(3, 4, 12)
plot_var_nolegend(ax1, x, WNA_IRR_TOTSOILLIQ, WNA_CTL_TOTSOILLIQ, WNA_NOI_TOTSOILLIQ, 'upper left', r'TSW ($\mathregular{kg/m^2}$)', 'TSW', 'l', 'SEA')

plt.subplots_adjust(left=0.1,
                    right=0.9,
                    top=0.9,
                    bottom=0.1,
                    wspace=0.4,
                    hspace=0.3)

plt.savefig('Timeseries_water2.png')
#plt.show()


