import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from Load_data import Data_from_nc

def set_plot_param():
    """Set my own customized plotting parameters"""

    mpl.rc('axes', edgecolor='dimgrey')
    mpl.rc('axes', labelcolor='dimgrey')
    mpl.rc('xtick', color='dimgrey')
    mpl.rc('ytick', color='dimgrey')
    mpl.rc('legend', fontsize='large')
    mpl.rc('text', color='dimgrey')

def plot_only_obs(ax1, x, x1, x2, v_OBS, v_QIRR_IRR, v_QIRR_CTL, label_OBS,
                  label_QIRR_IRR, label_QIRR_CTL,legloc_VAR, legloc_QIRR, ylabel_VAR, ylabel_QIRR, xtick_value, xtick_month, title, num, site):
    plt.plot(x, v_OBS, 'black', marker='o', label=label_OBS, linewidth=0.8, markersize=5, alpha=0.5)

    plt.ylabel(ylabel_VAR, fontsize=14)
    # plt.xlabel('month')
    plt.xticks(xtick_value,
               xtick_month,
               )
    plt.legend(loc=legloc_VAR, frameon=False, fontsize=13)
    plt.ylim(0,400)
    plt.tick_params(labelsize=14)
    plt.title(title, loc='right', fontsize=14)
    plt.title(site, loc='left', fontsize=14)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.2, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.2, label=label_QIRR_CTL)
    plt.ylim(0,3)
    plt.legend(loc=legloc_QIRR, frameon=False, fontsize=13)
    plt.ylabel(ylabel_QIRR, fontsize=14)

    plt.tick_params(labelsize=14)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=14, transform=ax1.transAxes, weight='bold')

def plot_only_obs_nolegend(ax1, x, x1, x2, v_OBS, v_QIRR_IRR, v_QIRR_CTL, label_OBS,
                  label_QIRR_IRR, label_QIRR_CTL,legloc_VAR, legloc_QIRR, ylabel_VAR, ylabel_QIRR, xtick_value, xtick_month, title, num, site):
    plt.plot(x, v_OBS, 'black', marker='o', label=label_OBS, linewidth=0.8, markersize=5, alpha=0.5)

    plt.ylabel(ylabel_VAR, fontsize=14)
    # plt.xlabel('month')
    plt.xticks(xtick_value,
               xtick_month,
               )
    #plt.legend(loc=legloc_VAR, frameon=False, fontsize=13)

    plt.tick_params(labelsize=14)
    plt.title(title, loc='right', fontsize=14)
    plt.title(site, loc='left', fontsize=14)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.2, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.2, label=label_QIRR_CTL)
    #plt.legend(loc=legloc_QIRR, frameon=False, fontsize=13)
    plt.ylabel(ylabel_QIRR, fontsize=14)

    plt.tick_params(labelsize=14)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=14, transform=ax1.transAxes, weight='bold')

def plot_with_obs(ax1, x, x1, x2, v_OBS, v_NOI, v_CTL, v_IRR, v_QIRR_IRR, v_QIRR_CTL, label_OBS, label_NOI, label_CTL, label_IRR,
                  label_QIRR_IRR, label_QIRR_CTL,legloc_VAR, legloc_QIRR, ylabel_VAR, ylabel_QIRR, xtick_value, xtick_month, title, num, site):
    plt.plot(x, v_OBS, 'black', marker='o', label=label_OBS, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_NOI, 'red', marker='v', label=label_NOI, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_CTL, 'blue', marker='>', label=label_CTL, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_IRR, 'green', marker='^', label=label_IRR, linewidth=0.8, markersize=5, alpha=0.5)
    plt.ylabel(ylabel_VAR, fontsize=14)
    # plt.xlabel('month')
    plt.xticks(xtick_value,
               xtick_month,
               )
    plt.legend(loc=legloc_VAR, frameon=False, fontsize=13)
    plt.ylim(0, 300)
    plt.tick_params(labelsize=14)
    plt.title(title, loc='right', fontsize=14)
    plt.title(site, loc='left', fontsize=14)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.2, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.2, label=label_QIRR_CTL)
    plt.legend(loc=legloc_QIRR, frameon=False, fontsize=13)
    plt.ylabel(ylabel_QIRR, fontsize=14)
    plt.ylim(0,4)
    plt.tick_params(labelsize=14)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=14, transform=ax1.transAxes, weight='bold')

def plot_with_obs_nolegend(ax1, x, x1, x2, v_OBS, v_NOI, v_CTL, v_IRR, v_QIRR_IRR, v_QIRR_CTL, label_OBS, label_NOI, label_CTL, label_IRR,
                  label_QIRR_IRR, label_QIRR_CTL,legloc_VAR, legloc_QIRR, ylabel_VAR, ylabel_QIRR, xtick_value, xtick_month, title, num, site):
    plt.plot(x, v_OBS, 'black', marker='o', label=label_OBS, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_NOI, 'red', marker='v', label=label_NOI, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_CTL, 'blue', marker='>', label=label_CTL, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_IRR, 'green', marker='^', label=label_IRR, linewidth=0.8, markersize=5, alpha=0.5)
    plt.ylabel(ylabel_VAR, fontsize=14)
    # plt.xlabel('month')
    plt.xticks(xtick_value,
               xtick_month,
               )

    plt.tick_params(labelsize=14)
    plt.title(title, loc='right', fontsize=14)
    plt.title(site, loc='left', fontsize=14)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.2, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.2, label=label_QIRR_CTL)

    plt.ylabel(ylabel_QIRR, fontsize=14)
    plt.tick_params(labelsize=14)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=14, transform=ax1.transAxes, weight='bold')

def plot_without_obs_nolegend(ax1, x, x1, x2, v_NOI, v_CTL, v_IRR, v_QIRR_IRR, v_QIRR_CTL, label_NOI, label_CTL, label_IRR,
                  label_QIRR_IRR, label_QIRR_CTL, ylabel_VAR, ylabel_QIRR, xtick_value, xtick_month, title, num, site):

    plt.plot(x, v_NOI, 'red', marker='v', label=label_NOI, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_CTL, 'blue', marker='>', label=label_CTL, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_IRR, 'green', marker='^', label=label_IRR, linewidth=0.8, markersize=5, alpha=0.5)
    plt.ylabel(ylabel_VAR, fontsize=14)
    # plt.xlabel('month')
    plt.xticks(xtick_value,
               xtick_month,
               )

    plt.tick_params(labelsize=14)
    plt.title(title, loc='right', fontsize=14)
    plt.title(site, loc='left', fontsize=14)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.2, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.2, label=label_QIRR_CTL)

    plt.ylabel(ylabel_QIRR, fontsize=14)
    plt.tick_params(labelsize=14)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=14, transform=ax1.transAxes, weight='bold')

def plot_with_obs_without(ax1, x, x1, x2, v_NOI, v_CTL, v_IRR, v_QIRR_IRR, v_QIRR_CTL, label_NOI, label_CTL, label_IRR,
                  label_QIRR_IRR, label_QIRR_CTL,legloc_VAR, legloc_QIRR, ylabel_VAR, ylabel_QIRR, xtick_value, xtick_month, title, num, site):
    plt.plot(x, v_NOI, 'red', marker='v', label=label_NOI, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_CTL, 'blue', marker='>', label=label_CTL, linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_IRR, 'green', marker='^', label=label_IRR, linewidth=0.8, markersize=5, alpha=0.5)
    plt.ylabel(ylabel_VAR, fontsize=14)
    # plt.xlabel('month')
    plt.xticks(xtick_value,
               xtick_month,
               )
    plt.legend(loc=legloc_VAR, frameon=False, fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim(-20, 50)
    plt.title(title, loc='right', fontsize=14)
    plt.title(site, loc='left', fontsize=14)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.2, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.2, label=label_QIRR_CTL)
    plt.ylim(0, 3)
    #plt.legend(loc=legloc_QIRR, frameon=False, fontsize=13)
    plt.ylabel(ylabel_QIRR, fontsize=14)
    plt.tick_params(labelsize=14)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=14, transform=ax1.transAxes, weight='bold')



str_infile_cas = 'Castellero_monthly.csv'
pd_reader_cas = pd.read_csv(str_infile_cas, header=None)
data_forc_cas = np.array(pd_reader_cas)
le_cas = data_forc_cas[:, 0]
hs_cas = data_forc_cas[:, 1]
lw_in_cas = data_forc_cas[:, 2]
sw_in_cas = data_forc_cas[:, 3]
sw_out_cas = data_forc_cas[:, 4]

le_month_cas = np.zeros(12)
hs_month_cas = np.zeros(12)
lw_in_month_cas = np.zeros(12)
sw_in_month_cas = np.zeros(12)
sw_out_month_cas = np.zeros(12)

num_cas = np.zeros(12)

data_CTL_cas = Data_from_nc('Castellaro_drip_SpGs_QIRRIG')
QIRRIG_CTL_cas = data_CTL_cas.load_variable('QIRRIG_FROM_SURFACE')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_QIRRIG')
QIRRIG_IRR_cas = data_IRR_cas.load_variable('QIRRIG_FROM_SURFACE')

QIRRIG_CTL_month_cas = np.zeros(12)
QIRRIG_IRR_month_cas = np.zeros(12)


data_NOI_cas = Data_from_nc('Castellaro_noirr_SpGs_EFLX_LH_TOT_monmean')
EFLX_LH_TOT_NOI_cas = data_NOI_cas.load_variable('EFLX_LH_TOT')
data_CTL_cas = Data_from_nc('Castellaro_drip_SpGs_EFLX_LH_TOT_monmean')
EFLX_LH_TOT_CTL_cas = data_CTL_cas.load_variable('EFLX_LH_TOT')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_EFLX_LH_TOT_monmean')
EFLX_LH_TOT_IRR_cas = data_IRR_cas.load_variable('EFLX_LH_TOT')

EFLX_LH_TOT_NOI_month_cas = np.zeros(12)
EFLX_LH_TOT_CTL_month_cas = np.zeros(12)
EFLX_LH_TOT_IRR_month_cas = np.zeros(12)

data_NOI_cas = Data_from_nc('Castellaro_noirr_SpGs_FSH_monmean')
FSH_NOI_cas = data_NOI_cas.load_variable('FSH')
data_CTL_cas = Data_from_nc('Castellaro_drip_SpGs_FSH_monmean')
FSH_CTL_cas = data_CTL_cas.load_variable('FSH')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_FSH_monmean')
FSH_IRR_cas = data_IRR_cas.load_variable('FSH')

FSH_NOI_month_cas = np.zeros(12)
FSH_CTL_month_cas = np.zeros(12)
FSH_IRR_month_cas = np.zeros(12)

data_NOI_cas = Data_from_nc('Castellaro_noirr_SpGs_LWup_monmean')
LWup_NOI_cas = data_NOI_cas.load_variable('LWup')
data_CTL_cas = Data_from_nc('Castellaro_drip_SpGs_LWup_monmean')
LWup_CTL_cas = data_CTL_cas.load_variable('LWup')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_LWup_monmean')
LWup_IRR_cas = data_IRR_cas.load_variable('LWup')

LWup_NOI_month_cas = np.zeros(12)
LWup_CTL_month_cas = np.zeros(12)
LWup_IRR_month_cas = np.zeros(12)

data_NOI_cas = Data_from_nc('Castellaro_noirr_SpGs_SWup_monmean')
SWup_NOI_cas = data_NOI_cas.load_variable('SWup')
data_CTL_cas = Data_from_nc('Castellaro_drip_SpGs_SWup_monmean')
SWup_CTL_cas = data_CTL_cas.load_variable('SWup')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_SWup_monmean')
SWup_IRR_cas = data_IRR_cas.load_variable('SWup')

SWup_NOI_month_cas = np.zeros(12)
SWup_CTL_month_cas = np.zeros(12)
SWup_IRR_month_cas = np.zeros(12)


data_NOI_cas = Data_from_nc('Castellaro_noirr_SpGs_FGR_monmean.nc')
FGR_NOI_cas = data_NOI_cas.load_variable('FGR')
data_CTL_cas = Data_from_nc('Castellaro_drip_SpGs_FGR_monmean.nc')
FGR_CTL_cas = data_CTL_cas.load_variable('FGR')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_FGR_monmean.nc')
FGR_IRR_cas = data_IRR_cas.load_variable('FGR')

FGR_NOI_month_cas = np.zeros(12)
FGR_CTL_month_cas = np.zeros(12)
FGR_IRR_month_cas = np.zeros(12)

data_NOI_cas = Data_from_nc('Castellaro_noirr_SpGs_Rnet_monmean.nc')
Rnet_NOI_cas = data_NOI_cas.load_variable('Rnet')
data_CTL_cas = Data_from_nc('Castellaro_drip_SpGs_Rnet_monmean.nc')
Rnet_CTL_cas = data_CTL_cas.load_variable('Rnet')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_Rnet_monmean.nc')
Rnet_IRR_cas = data_IRR_cas.load_variable('Rnet')

Rnet_NOI_month_cas = np.zeros(12)
Rnet_CTL_month_cas = np.zeros(12)
Rnet_IRR_month_cas = np.zeros(12)

for year in range(2):
    for month in range(12):
        le_month_cas[month] = le_month_cas[month] + le_cas[12*year + month]
        EFLX_LH_TOT_NOI_month_cas[month] = EFLX_LH_TOT_NOI_month_cas[month] + EFLX_LH_TOT_NOI_cas[12 * year + month]
        EFLX_LH_TOT_CTL_month_cas[month] = EFLX_LH_TOT_CTL_month_cas[month] + EFLX_LH_TOT_CTL_cas[12 * year + month]
        EFLX_LH_TOT_IRR_month_cas[month] = EFLX_LH_TOT_IRR_month_cas[month] + EFLX_LH_TOT_IRR_cas[12 * year + month]
        QIRRIG_CTL_month_cas[month] = QIRRIG_CTL_month_cas[month] + QIRRIG_CTL_cas[12 * year + month]
        QIRRIG_IRR_month_cas[month] = QIRRIG_IRR_month_cas[month] + QIRRIG_IRR_cas[12 * year + month]

        hs_month_cas[month] = hs_month_cas[month] + hs_cas[12*year + month]
        FSH_NOI_month_cas[month] = FSH_NOI_month_cas[month] + FSH_NOI_cas[12 * year + month]
        FSH_CTL_month_cas[month] = FSH_CTL_month_cas[month] + FSH_CTL_cas[12 * year + month]
        FSH_IRR_month_cas[month] = FSH_IRR_month_cas[month] + FSH_IRR_cas[12 * year + month]

        lw_in_month_cas[month] = lw_in_month_cas[month] + lw_in_cas[12*year + month]
        LWup_NOI_month_cas[month] = LWup_NOI_month_cas[month] + LWup_NOI_cas[12 * year + month]
        LWup_CTL_month_cas[month] = LWup_CTL_month_cas[month] + LWup_CTL_cas[12 * year + month]
        LWup_IRR_month_cas[month] = LWup_IRR_month_cas[month] + LWup_IRR_cas[12 * year + month]

        sw_in_month_cas[month] = sw_in_month_cas[month] + sw_in_cas[12*year + month]
        SWup_NOI_month_cas[month] = SWup_NOI_month_cas[month] + SWup_NOI_cas[12 * year + month]
        SWup_CTL_month_cas[month] = SWup_CTL_month_cas[month] + SWup_CTL_cas[12 * year + month]
        SWup_IRR_month_cas[month] = SWup_IRR_month_cas[month] + SWup_IRR_cas[12 * year + month]

        FGR_NOI_month_cas[month] = FGR_NOI_month_cas[month] + FGR_NOI_cas[12 * year + month]
        FGR_CTL_month_cas[month] = FGR_CTL_month_cas[month] + FGR_CTL_cas[12 * year + month]
        FGR_IRR_month_cas[month] = FGR_IRR_month_cas[month] + FGR_IRR_cas[12 * year + month]

        Rnet_NOI_month_cas[month] = Rnet_NOI_month_cas[month] + Rnet_NOI_cas[12 * year + month]
        Rnet_CTL_month_cas[month] = Rnet_CTL_month_cas[month] + Rnet_CTL_cas[12 * year + month]
        Rnet_IRR_month_cas[month] = Rnet_IRR_month_cas[month] + Rnet_IRR_cas[12 * year + month]

        sw_out_month_cas[month] = sw_out_month_cas[month] + sw_out_cas[12*year + month]
        num_cas[month] = num_cas[month] + 1

le_month_cas = le_month_cas / num_cas
EFLX_LH_TOT_NOI_month_cas = EFLX_LH_TOT_NOI_month_cas / num_cas
EFLX_LH_TOT_CTL_month_cas = EFLX_LH_TOT_CTL_month_cas / num_cas
EFLX_LH_TOT_IRR_month_cas = EFLX_LH_TOT_IRR_month_cas / num_cas

QIRRIG_CTL_month_cas = QIRRIG_CTL_month_cas / num_cas
QIRRIG_IRR_month_cas = QIRRIG_IRR_month_cas / num_cas

hs_month_cas = hs_month_cas / num_cas
FSH_NOI_month_cas = FSH_NOI_month_cas / num_cas
FSH_CTL_month_cas = FSH_CTL_month_cas / num_cas
FSH_IRR_month_cas = FSH_IRR_month_cas / num_cas

lw_in_month_cas = lw_in_month_cas / num_cas
LWup_NOI_month_cas = LWup_NOI_month_cas / num_cas
LWup_CTL_month_cas = LWup_CTL_month_cas / num_cas
LWup_IRR_month_cas = LWup_IRR_month_cas / num_cas

sw_in_month_cas = sw_in_month_cas / num_cas

sw_out_month_cas = sw_out_month_cas / num_cas
SWup_NOI_month_cas = SWup_NOI_month_cas / num_cas
SWup_CTL_month_cas = SWup_CTL_month_cas / num_cas
SWup_IRR_month_cas = SWup_IRR_month_cas / num_cas

FGR_NOI_month_cas = FGR_NOI_month_cas / num_cas
FGR_CTL_month_cas = FGR_CTL_month_cas / num_cas
FGR_IRR_month_cas = FGR_IRR_month_cas / num_cas

Rnet_NOI_month_cas = Rnet_NOI_month_cas / num_cas
Rnet_CTL_month_cas = Rnet_CTL_month_cas / num_cas
Rnet_IRR_month_cas = Rnet_IRR_month_cas / num_cas

x = range(1, 13)
x1 = np.array(x) - 0.15
x2 = np.array(x) + 0.15

f = plt.figure(figsize = (16, 10), dpi=100)
f.subplots_adjust(hspace=0.4, wspace=0.4, left = 0.07, right = 0.95, top = 0.95, bottom = 0.05)
set_plot_param()


ax1 = plt.subplot(3, 4, 5)
plot_with_obs_nolegend(ax1, x, x1, x2, le_month_cas, EFLX_LH_TOT_NOI_month_cas, EFLX_LH_TOT_CTL_month_cas, EFLX_LH_TOT_IRR_month_cas,
                  QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '($\mathregular{W/m^2}$)', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'LHF', 'e', 'CAS')

ax1 = plt.subplot(3, 4, 6)
plot_with_obs_nolegend(ax1, x, x1, x2, hs_month_cas, FSH_NOI_month_cas, FSH_CTL_month_cas, FSH_IRR_month_cas,
                  QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'SHF', 'f', 'CAS')

ax1 = plt.subplot(3, 4, 7)
plot_without_obs_nolegend(ax1, x, x1, x2, LWup_NOI_month_cas, LWup_CTL_month_cas, LWup_IRR_month_cas,
                  QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'LWup', 'g', 'CAS')

ax1 = plt.subplot(3, 4, 8)
plot_with_obs_nolegend(ax1, x, x1, x2, sw_out_month_cas, SWup_NOI_month_cas, SWup_CTL_month_cas, SWup_IRR_month_cas,
                  QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', 'Qirr (mm/day)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'SWup', 'h', 'CAS')

str_infile_neb = 'Nebraska_monthly.csv'
pd_reader_neb = pd.read_csv(str_infile_neb, header=None)
data_forc_neb = np.array(pd_reader_neb)
le_neb = data_forc_neb[:, 0]
hs_neb = data_forc_neb[:, 1]
lw_in_neb = data_forc_neb[:, 2]
sw_in_neb = data_forc_neb[:, 3]
lw_out_neb = data_forc_neb[:, 4]
sw_out_neb = data_forc_neb[:, 5]

le_month_neb = np.zeros(12)
hs_month_neb = np.zeros(12)
lw_in_month_neb = np.zeros(12)
sw_in_month_neb = np.zeros(12)
lw_out_month_neb = np.zeros(12)
sw_out_month_neb = np.zeros(12)

num_neb = np.zeros(12)


data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_QIRRIG')
QIRRIG_CTL_neb = data_CTL_neb.load_variable('QIRRIG_FROM_SURFACE')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_QIRRIG')
QIRRIG_IRR_neb = data_IRR_neb.load_variable('QIRRIG_FROM_SURFACE')

QIRRIG_CTL_month_neb = np.zeros(12)
QIRRIG_IRR_month_neb = np.zeros(12)

data_NOI_neb = Data_from_nc('Ne1_noirr_SpGs_EFLX_LH_TOT_monmean')
EFLX_LH_TOT_NOI_neb = data_NOI_neb.load_variable('EFLX_LH_TOT')
data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_EFLX_LH_TOT_monmean')
EFLX_LH_TOT_CTL_neb = data_CTL_neb.load_variable('EFLX_LH_TOT')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_EFLX_LH_TOT_monmean')
EFLX_LH_TOT_IRR_neb = data_IRR_neb.load_variable('EFLX_LH_TOT')

EFLX_LH_TOT_NOI_month_neb = np.zeros(12)
EFLX_LH_TOT_CTL_month_neb = np.zeros(12)
EFLX_LH_TOT_IRR_month_neb = np.zeros(12)

data_NOI_neb = Data_from_nc('Ne1_noirr_SpGs_FSH_monmean')
FSH_NOI_neb = data_NOI_neb.load_variable('FSH')
data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_FSH_monmean')
FSH_CTL_neb = data_CTL_neb.load_variable('FSH')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_FSH_monmean')
FSH_IRR_neb = data_IRR_neb.load_variable('FSH')

FSH_NOI_month_neb = np.zeros(12)
FSH_CTL_month_neb = np.zeros(12)
FSH_IRR_month_neb = np.zeros(12)

data_NOI_neb = Data_from_nc('Ne1_noirr_SpGs_LWup_monmean')
LWup_NOI_neb = data_NOI_neb.load_variable('LWup')
data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_LWup_monmean')
LWup_CTL_neb = data_CTL_neb.load_variable('LWup')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_LWup_monmean')
LWup_IRR_neb = data_IRR_neb.load_variable('LWup')

LWup_NOI_month_neb = np.zeros(12)
LWup_CTL_month_neb = np.zeros(12)
LWup_IRR_month_neb = np.zeros(12)

data_NOI_neb = Data_from_nc('Ne1_noirr_SpGs_SWup_monmean')
SWup_NOI_neb = data_NOI_neb.load_variable('SWup')
data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_SWup_monmean')
SWup_CTL_neb = data_CTL_neb.load_variable('SWup')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_SWup_monmean')
SWup_IRR_neb = data_IRR_neb.load_variable('SWup')

SWup_NOI_month_neb = np.zeros(12)
SWup_CTL_month_neb = np.zeros(12)
SWup_IRR_month_neb = np.zeros(12)

data_NOI_neb = Data_from_nc('Ne1_noirr_SpGs_FGR_monmean.nc')
FGR_NOI_neb = data_NOI_neb.load_variable('FGR')
data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_FGR_monmean.nc')
FGR_CTL_neb = data_CTL_neb.load_variable('FGR')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_FGR_monmean.nc')
FGR_IRR_neb = data_IRR_neb.load_variable('FGR')

FGR_NOI_month_neb = np.zeros(12)
FGR_CTL_month_neb = np.zeros(12)
FGR_IRR_month_neb = np.zeros(12)

data_NOI_neb = Data_from_nc('Ne1_noirr_SpGs_Rnet_monmean.nc')
Rnet_NOI_neb = data_NOI_neb.load_variable('Rnet')
data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_Rnet_monmean.nc')
Rnet_CTL_neb = data_CTL_neb.load_variable('Rnet')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_Rnet_monmean.nc')
Rnet_IRR_neb = data_IRR_neb.load_variable('Rnet')

Rnet_NOI_month_neb = np.zeros(12)
Rnet_CTL_month_neb = np.zeros(12)
Rnet_IRR_month_neb = np.zeros(12)

for year in range(11):
    for month in range(12):
        if lw_out_neb[12*year + month] > 100 and sw_out_neb[12*year + month] > -200:
            le_month_neb[month] = le_month_neb[month] + le_neb[12*year + month]
            EFLX_LH_TOT_NOI_month_neb[month] = EFLX_LH_TOT_NOI_month_neb[month] + EFLX_LH_TOT_NOI_neb[12 * year + 12 + month]
            EFLX_LH_TOT_CTL_month_neb[month] = EFLX_LH_TOT_CTL_month_neb[month] + EFLX_LH_TOT_CTL_neb[12 * year + 12 + month]
            EFLX_LH_TOT_IRR_month_neb[month] = EFLX_LH_TOT_IRR_month_neb[month] + EFLX_LH_TOT_IRR_neb[12 * year + 12 + month]

            QIRRIG_CTL_month_neb[month] = QIRRIG_CTL_month_neb[month] + QIRRIG_CTL_neb[12 * year + 12 + month]
            QIRRIG_IRR_month_neb[month] = QIRRIG_IRR_month_neb[month] + QIRRIG_IRR_neb[12 * year + 12 + month]


            hs_month_neb[month] = hs_month_neb[month] + hs_neb[12*year + month]
            FSH_NOI_month_neb[month] = FSH_NOI_month_neb[month] + FSH_NOI_neb[12 * year + 12 + month]
            FSH_CTL_month_neb[month] = FSH_CTL_month_neb[month] + FSH_CTL_neb[12 * year + 12 + month]
            FSH_IRR_month_neb[month] = FSH_IRR_month_neb[month] + FSH_IRR_neb[12 * year + 12 + month]


            lw_in_month_neb[month] = lw_in_month_neb[month] + lw_in_neb[12*year + month]
            LWup_NOI_month_neb[month] = LWup_NOI_month_neb[month] + LWup_NOI_neb[12 * year + 12 + month]
            LWup_CTL_month_neb[month] = LWup_CTL_month_neb[month] + LWup_CTL_neb[12 * year + 12 + month]
            LWup_IRR_month_neb[month] = LWup_IRR_month_neb[month] + LWup_IRR_neb[12 * year + 12 + month]

            sw_in_month_neb[month] = sw_in_month_neb[month] + sw_in_neb[12*year + month]
            SWup_NOI_month_neb[month] = SWup_NOI_month_neb[month] + SWup_NOI_neb[12 * year + 12 + month]
            SWup_CTL_month_neb[month] = SWup_CTL_month_neb[month] + SWup_CTL_neb[12 * year + 12 + month]
            SWup_IRR_month_neb[month] = SWup_IRR_month_neb[month] + SWup_IRR_neb[12 * year + 12 + month]

            FGR_NOI_month_neb[month] = FGR_NOI_month_neb[month] + FGR_NOI_neb[12 * year + 12 + month]
            FGR_CTL_month_neb[month] = FGR_CTL_month_neb[month] + FGR_CTL_neb[12 * year + 12 + month]
            FGR_IRR_month_neb[month] = FGR_IRR_month_neb[month] + FGR_IRR_neb[12 * year + 12 + month]

            Rnet_NOI_month_neb[month] = Rnet_NOI_month_neb[month] + Rnet_NOI_neb[12 * year + 12 + month]
            Rnet_CTL_month_neb[month] = Rnet_CTL_month_neb[month] + Rnet_CTL_neb[12 * year + 12 + month]
            Rnet_IRR_month_neb[month] = Rnet_IRR_month_neb[month] + Rnet_IRR_neb[12 * year + 12 + month]

            lw_out_month_neb[month] = lw_out_month_neb[month] + lw_out_neb[12 * year + month]


            sw_out_month_neb[month] = sw_out_month_neb[month] + sw_out_neb[12 * year + month]


            num_neb[month] = num_neb[month] + 1

le_month_neb = le_month_neb / num_neb
EFLX_LH_TOT_NOI_month_neb = EFLX_LH_TOT_NOI_month_neb / num_neb
EFLX_LH_TOT_CTL_month_neb = EFLX_LH_TOT_CTL_month_neb / num_neb
EFLX_LH_TOT_IRR_month_neb = EFLX_LH_TOT_IRR_month_neb / num_neb


QIRRIG_CTL_month_neb = QIRRIG_CTL_month_neb / num_neb
QIRRIG_IRR_month_neb = QIRRIG_IRR_month_neb / num_neb


hs_month_neb = hs_month_neb / num_neb
FSH_NOI_month_neb = FSH_NOI_month_neb / num_neb
FSH_CTL_month_neb = FSH_CTL_month_neb / num_neb
FSH_IRR_month_neb = FSH_IRR_month_neb / num_neb


lw_in_month_neb = lw_in_month_neb / num_neb
lw_out_month_neb = lw_out_month_neb / num_neb
LWup_NOI_month_neb = LWup_NOI_month_neb / num_neb
LWup_CTL_month_neb = LWup_CTL_month_neb / num_neb
LWup_IRR_month_neb = LWup_IRR_month_neb / num_neb


sw_in_month_neb = sw_in_month_neb / num_neb
sw_out_month_neb = sw_out_month_neb / num_neb
SWup_NOI_month_neb = SWup_NOI_month_neb / num_neb
SWup_CTL_month_neb = SWup_CTL_month_neb / num_neb
SWup_IRR_month_neb = SWup_IRR_month_neb / num_neb

FGR_NOI_month_neb = FGR_NOI_month_neb / num_neb
FGR_CTL_month_neb = FGR_CTL_month_neb / num_neb
FGR_IRR_month_neb = FGR_IRR_month_neb / num_neb

Rnet_NOI_month_neb = Rnet_NOI_month_neb / num_neb
Rnet_CTL_month_neb = Rnet_CTL_month_neb / num_neb
Rnet_IRR_month_neb = Rnet_IRR_month_neb / num_neb

x = range(1, 13)
x1 = np.array(x) - 0.15
x2 = np.array(x) + 0.15


ax1 = plt.subplot(3, 4, 1)
plot_with_obs(ax1, x, x1, x2, le_month_neb, EFLX_LH_TOT_NOI_month_neb, EFLX_LH_TOT_CTL_month_neb, EFLX_LH_TOT_IRR_month_neb,
                  QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '($\mathregular{W/m^2}$)', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'LHF', 'a', 'NEB')

ax1 = plt.subplot(3, 4, 2)
plot_with_obs_nolegend(ax1, x, x1, x2, hs_month_neb, FSH_NOI_month_neb, FSH_CTL_month_neb, FSH_IRR_month_neb,
                  QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'SHF', 'b', 'NEB')

ax1 = plt.subplot(3, 4, 3)
plot_with_obs_nolegend(ax1, x, x1, x2, lw_out_month_neb, LWup_NOI_month_neb, LWup_CTL_month_neb, LWup_IRR_month_neb,
                  QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'LWup', 'c', 'NEB')

ax1 = plt.subplot(3, 4, 4)
plot_with_obs_nolegend(ax1, x, x1, x2, sw_out_month_neb, SWup_NOI_month_neb, SWup_CTL_month_neb, SWup_IRR_month_neb,
                  QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', 'Qirr (mm/day)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'SWup', 'd', 'NEB')


str_infile_mas = 'Japan_monthly.csv'
pd_reader_mas = pd.read_csv(str_infile_mas, header=None)
data_forc_mas = np.array(pd_reader_mas)
le_mas = data_forc_mas[:, 0]
hs_mas = data_forc_mas[:, 1]
lw_in_mas = data_forc_mas[:, 2]
sw_in_mas = data_forc_mas[:, 3]
lw_out_mas = data_forc_mas[:, 4]
sw_out_mas = data_forc_mas[:, 5]

le_month_mas = np.zeros(12)
hs_month_mas = np.zeros(12)
lw_in_month_mas = np.zeros(12)
sw_in_month_mas = np.zeros(12)
lw_out_month_mas = np.zeros(12)
sw_out_month_mas = np.zeros(12)

num_mas = np.zeros(12)


data_CTL_mas = Data_from_nc('Japan_drip_SpGs_QIRRIG')
QIRRIG_CTL_mas = data_CTL_mas.load_variable('QIRRIG_FROM_SURFACE')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_QIRRIG')
QIRRIG_IRR_mas = data_IRR_mas.load_variable('QIRRIG_FROM_SURFACE')

QIRRIG_CTL_month_mas = np.zeros(12)
QIRRIG_IRR_month_mas = np.zeros(12)



data_NOI_mas = Data_from_nc('Japan_noirr_SpGs_EFLX_LH_TOT_monmean')
EFLX_LH_TOT_NOI_mas = data_NOI_mas.load_variable('EFLX_LH_TOT')
data_CTL_mas = Data_from_nc('Japan_drip_SpGs_EFLX_LH_TOT_monmean')
EFLX_LH_TOT_CTL_mas = data_CTL_mas.load_variable('EFLX_LH_TOT')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_EFLX_LH_TOT_monmean')
EFLX_LH_TOT_IRR_mas = data_IRR_mas.load_variable('EFLX_LH_TOT')

EFLX_LH_TOT_NOI_month_mas = np.zeros(12)
EFLX_LH_TOT_CTL_month_mas = np.zeros(12)
EFLX_LH_TOT_IRR_month_mas = np.zeros(12)

data_NOI_mas = Data_from_nc('Japan_noirr_SpGs_FSH_monmean')
FSH_NOI_mas = data_NOI_mas.load_variable('FSH')
data_CTL_mas = Data_from_nc('Japan_drip_SpGs_FSH_monmean')
FSH_CTL_mas = data_CTL_mas.load_variable('FSH')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_FSH_monmean')
FSH_IRR_mas = data_IRR_mas.load_variable('FSH')

FSH_NOI_month_mas = np.zeros(12)
FSH_CTL_month_mas = np.zeros(12)
FSH_IRR_month_mas = np.zeros(12)

data_NOI_mas = Data_from_nc('Japan_noirr_SpGs_LWup_monmean')
LWup_NOI_mas = data_NOI_mas.load_variable('LWup')
data_CTL_mas = Data_from_nc('Japan_drip_SpGs_LWup_monmean')
LWup_CTL_mas = data_CTL_mas.load_variable('LWup')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_LWup_monmean')
LWup_IRR_mas = data_IRR_mas.load_variable('LWup')

LWup_NOI_month_mas = np.zeros(12)
LWup_CTL_month_mas = np.zeros(12)
LWup_IRR_month_mas = np.zeros(12)

data_NOI_mas = Data_from_nc('Japan_noirr_SpGs_SWup_monmean')
SWup_NOI_mas = data_NOI_mas.load_variable('SWup')
data_CTL_mas = Data_from_nc('Japan_drip_SpGs_SWup_monmean')
SWup_CTL_mas = data_CTL_mas.load_variable('SWup')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_SWup_monmean')
SWup_IRR_mas = data_IRR_mas.load_variable('SWup')

SWup_NOI_month_mas = np.zeros(12)
SWup_CTL_month_mas = np.zeros(12)
SWup_IRR_month_mas = np.zeros(12)

data_NOI_mas = Data_from_nc('Japan_noirr_SpGs_FGR_monmean.nc')
FGR_NOI_mas = data_NOI_mas.load_variable('FGR')
data_CTL_mas = Data_from_nc('Japan_drip_SpGs_FGR_monmean.nc')
FGR_CTL_mas = data_CTL_mas.load_variable('FGR')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_FGR_monmean.nc')
FGR_IRR_mas = data_IRR_mas.load_variable('FGR')

FGR_NOI_month_mas = np.zeros(12)
FGR_CTL_month_mas = np.zeros(12)
FGR_IRR_month_mas = np.zeros(12)

data_NOI_mas = Data_from_nc('Japan_noirr_SpGs_Rnet_monmean.nc')
Rnet_NOI_mas = data_NOI_mas.load_variable('Rnet')
data_CTL_mas = Data_from_nc('Japan_drip_SpGs_Rnet_monmean.nc')
Rnet_CTL_mas = data_CTL_mas.load_variable('Rnet')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_Rnet_monmean.nc')
Rnet_IRR_mas = data_IRR_mas.load_variable('Rnet')

Rnet_NOI_month_mas = np.zeros(12)
Rnet_CTL_month_mas = np.zeros(12)
Rnet_IRR_month_mas = np.zeros(12)

for year in range(1):
    for month in range(12):
        if le_mas[12*year + month] > -90 and hs_mas[12*year + month] > -90:
            le_month_mas[month] = le_month_mas[month] + le_mas[12*year + month]
            EFLX_LH_TOT_NOI_month_mas[month] = EFLX_LH_TOT_NOI_month_mas[month] + EFLX_LH_TOT_NOI_mas[12 * year + month]
            EFLX_LH_TOT_CTL_month_mas[month] = EFLX_LH_TOT_CTL_month_mas[month] + EFLX_LH_TOT_CTL_mas[12 * year + month]
            EFLX_LH_TOT_IRR_month_mas[month] = EFLX_LH_TOT_IRR_month_mas[month] + EFLX_LH_TOT_IRR_mas[12 * year + month]

            QIRRIG_CTL_month_mas[month] = QIRRIG_CTL_month_mas[month] + QIRRIG_CTL_mas[12 * year + month]
            QIRRIG_IRR_month_mas[month] = QIRRIG_IRR_month_mas[month] + QIRRIG_IRR_mas[12 * year + month]


            hs_month_mas[month] = hs_month_mas[month] + hs_mas[12*year + month]
            FSH_NOI_month_mas[month] = FSH_NOI_month_mas[month] + FSH_NOI_mas[12 * year + month]
            FSH_CTL_month_mas[month] = FSH_CTL_month_mas[month] + FSH_CTL_mas[12 * year + month]
            FSH_IRR_month_mas[month] = FSH_IRR_month_mas[month] + FSH_IRR_mas[12 * year + month]


            lw_in_month_mas[month] = lw_in_month_mas[month] + lw_in_mas[12*year + month]
            lw_out_month_mas[month] = lw_out_month_mas[month] + lw_out_mas[12 * year + month]
            LWup_NOI_month_mas[month] = LWup_NOI_month_mas[month] + LWup_NOI_mas[12 * year + month]
            LWup_CTL_month_mas[month] = LWup_CTL_month_mas[month] + LWup_CTL_mas[12 * year + month]
            LWup_IRR_month_mas[month] = LWup_IRR_month_mas[month] + LWup_IRR_mas[12 * year + month]


            sw_in_month_mas[month] = sw_in_month_mas[month] + sw_in_mas[12*year + month]
            sw_out_month_mas[month] = sw_out_month_mas[month] + sw_out_mas[12 * year + month]
            SWup_NOI_month_mas[month] = SWup_NOI_month_mas[month] + SWup_NOI_mas[12 * year + month]
            SWup_CTL_month_mas[month] = SWup_CTL_month_mas[month] + SWup_CTL_mas[12 * year + month]
            SWup_IRR_month_mas[month] = SWup_IRR_month_mas[month] + SWup_IRR_mas[12 * year + month]

            FGR_NOI_month_mas[month] = FGR_NOI_month_mas[month] + FGR_NOI_mas[12 * year + month]
            FGR_CTL_month_mas[month] = FGR_CTL_month_mas[month] + FGR_CTL_mas[12 * year + month]
            FGR_IRR_month_mas[month] = FGR_IRR_month_mas[month] + FGR_IRR_mas[12 * year + month]

            Rnet_NOI_month_mas[month] = Rnet_NOI_month_mas[month] + Rnet_NOI_mas[12 * year + month]
            Rnet_CTL_month_mas[month] = Rnet_CTL_month_mas[month] + Rnet_CTL_mas[12 * year + month]
            Rnet_IRR_month_mas[month] = Rnet_IRR_month_mas[month] + Rnet_IRR_mas[12 * year + month]

            num_mas[month] = num_mas[month] + 1

le_month_mas = le_month_mas / num_mas
EFLX_LH_TOT_NOI_month_mas = EFLX_LH_TOT_NOI_month_mas / num_mas
EFLX_LH_TOT_CTL_month_mas = EFLX_LH_TOT_CTL_month_mas / num_mas
EFLX_LH_TOT_IRR_month_mas = EFLX_LH_TOT_IRR_month_mas / num_mas


QIRRIG_CTL_month_mas = QIRRIG_CTL_month_mas / num_mas
QIRRIG_IRR_month_mas = QIRRIG_IRR_month_mas / num_mas

hs_month_mas = hs_month_mas / num_mas
FSH_NOI_month_mas = FSH_NOI_month_mas / num_mas
FSH_CTL_month_mas = FSH_CTL_month_mas / num_mas
FSH_IRR_month_mas = FSH_IRR_month_mas / num_mas


lw_in_month_mas = lw_in_month_mas / num_mas
lw_out_month_mas = lw_out_month_mas / num_mas
LWup_NOI_month_mas = LWup_NOI_month_mas / num_mas
LWup_CTL_month_mas = LWup_CTL_month_mas / num_mas
LWup_IRR_month_mas = LWup_IRR_month_mas / num_mas


sw_in_month_mas = sw_in_month_mas / num_mas
sw_out_month_mas = sw_out_month_mas / num_mas
SWup_NOI_month_mas = SWup_NOI_month_mas / num_mas
SWup_CTL_month_mas = SWup_CTL_month_mas / num_mas
SWup_IRR_month_mas = SWup_IRR_month_mas / num_mas

FGR_NOI_month_mas = FGR_NOI_month_mas / num_mas
FGR_CTL_month_mas = FGR_CTL_month_mas / num_mas
FGR_IRR_month_mas = FGR_IRR_month_mas / num_mas

Rnet_NOI_month_mas = Rnet_NOI_month_mas / num_mas
Rnet_CTL_month_mas = Rnet_CTL_month_mas / num_mas
Rnet_IRR_month_mas = Rnet_IRR_month_mas / num_mas

x = range(1, 13)
x1 = np.array(x) - 0.15
x2 = np.array(x) + 0.15

ax1 = plt.subplot(3, 4, 9)
plot_with_obs_nolegend(ax1, x, x1, x2, le_month_mas, EFLX_LH_TOT_NOI_month_mas, EFLX_LH_TOT_CTL_month_mas, EFLX_LH_TOT_IRR_month_mas,
                  QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '($\mathregular{W/m^2}$)', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'LHF', 'h', 'MAS')

ax1 = plt.subplot(3, 4, 10)
plot_with_obs_nolegend(ax1, x, x1, x2, hs_month_mas, FSH_NOI_month_mas, FSH_CTL_month_mas, FSH_IRR_month_mas,
                  QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'SHF', 'i', 'MAS')

ax1 = plt.subplot(3, 4, 11)
plot_with_obs_nolegend(ax1, x, x1, x2, lw_out_month_mas, LWup_NOI_month_mas, LWup_CTL_month_mas, LWup_IRR_month_mas,
                  QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'LWup', 'j', 'MAS')

ax1 = plt.subplot(3, 4, 12)
plot_with_obs_nolegend(ax1, x, x1, x2, sw_out_month_mas, SWup_NOI_month_mas, SWup_CTL_month_mas, SWup_IRR_month_mas,
                  QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', 'Qirr (mm/day)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'SWup', 'k', 'MAS')

plt.subplots_adjust(left=0.1,
                    right=0.9,
                    top=0.9,
                    bottom=0.1,
                    wspace=0.35,
                    hspace=0.3)


#plt.savefig('C:\Research1\Figures\\Single_point_LH_SH_LW_SW.png')
#plt.show()

x = range(1, 13)
x1 = np.array(x) - 0.15
x2 = np.array(x) + 0.15

f = plt.figure(figsize = (16, 10), dpi=1000)
f.subplots_adjust(hspace=0.4, wspace=0.4, left = 0.07, right = 0.95, top = 0.95, bottom = 0.05)
set_plot_param()

ax1 = plt.subplot(3, 4, 1)

plot_only_obs(ax1, x, x1, x2, sw_in_month_neb,QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'OBS',
                  'Qirr_irr', 'Qirr_ctl','upper left', 'upper right', '($\mathregular{W/m^2}$)', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D']
                    , 'SWdown', 'a', 'NEB')

ax1 = plt.subplot(3, 4, 2)
plot_only_obs_nolegend(ax1, x, x1, x2, lw_in_month_neb,QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'OBS',
                  'Qirr_irr', 'Qirr_ctl','upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D']
                    , 'LWdown', 'b', 'NEB')

ax1 = plt.subplot(3, 4, 3)
plot_with_obs_without(ax1, x, x1, x2, FGR_NOI_month_neb, FGR_CTL_month_neb, FGR_IRR_month_neb,
                      QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl','upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                      'GH', 'c', 'NEB')

ax1 = plt.subplot(3, 4, 4)
plot_without_obs_nolegend(ax1, x, x1, x2, Rnet_NOI_month_neb, Rnet_CTL_month_neb, Rnet_IRR_month_neb,
                      QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', '', 'Qirr (mm/day)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                      'Rnet', 'd', 'NEB')


ax1 = plt.subplot(3, 4, 5)

plot_only_obs_nolegend(ax1, x, x1, x2, sw_in_month_cas,QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'OBS',
                  'Qirr_irr', 'Qirr_ctl','upper left', 'upper right', '($\mathregular{W/m^2}$)', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D']
                    , 'SWdown', 'e', 'CAS')

ax1 = plt.subplot(3, 4, 6)
plot_only_obs_nolegend(ax1, x, x1, x2, lw_in_month_cas,QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'OBS',
                  'Qirr_irr', 'Qirr_ctl','upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D']
                    , 'LWdown', 'f', 'CAS')

ax1 = plt.subplot(3, 4, 7)
plot_without_obs_nolegend(ax1, x, x1, x2, FGR_NOI_month_cas, FGR_CTL_month_cas, FGR_IRR_month_cas,
                      QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                      'GH', 'g', 'CAS')

ax1 = plt.subplot(3, 4, 8)
plot_without_obs_nolegend(ax1, x, x1, x2, Rnet_NOI_month_cas, Rnet_CTL_month_cas, Rnet_IRR_month_cas,
                      QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', '', 'Qirr (mm/day)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                      'Rnet', 'h', 'CAS')

ax1 = plt.subplot(3, 4, 9)

plot_only_obs_nolegend(ax1, x, x1, x2, sw_in_month_mas,QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'OBS',
                  'Qirr_irr', 'Qirr_ctl','upper left', 'upper right', '($\mathregular{W/m^2}$)', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D']
                    , 'SWdown', 'i', 'MAS')

ax1 = plt.subplot(3, 4, 10)
plot_only_obs_nolegend(ax1, x, x1, x2, lw_in_month_mas,QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'OBS',
                  'Qirr_irr', 'Qirr_ctl','upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D']
                    , 'LWdown', 'j', 'MAS')

ax1 = plt.subplot(3, 4, 11)
plot_without_obs_nolegend(ax1, x, x1, x2, FGR_NOI_month_mas, FGR_CTL_month_mas, FGR_IRR_month_mas,
                      QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                      'GH', 'k', 'MAS')

ax1 = plt.subplot(3, 4, 12)
plot_without_obs_nolegend(ax1, x, x1, x2, Rnet_NOI_month_mas, Rnet_CTL_month_mas, Rnet_IRR_month_mas,
                      QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', '', 'Qirr (mm/day)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                      'Rnet', 'l', 'MAS')
plt.subplots_adjust(left=0.1,
                    right=0.9,
                    top=0.9,
                    bottom=0.1,
                    wspace=0.35,
                    hspace=0.3)
FSH_IRR_CTL_mas = FSH_IRR_month_mas - FSH_CTL_month_mas
FSH_IRR_CTL_cas = FSH_IRR_month_cas - FSH_CTL_month_cas
FSH_IRR_CTL_neb = FSH_IRR_month_neb - FSH_CTL_month_neb
FSH_IRR_NOI_mas = FSH_IRR_month_mas - FSH_NOI_month_mas
FSH_IRR_NOI_cas = FSH_IRR_month_cas - FSH_NOI_month_cas
FSH_IRR_NOI_neb = FSH_IRR_month_neb - FSH_NOI_month_neb
plt.savefig('Single_point_SW_LW_GH_Rnet.png')
#plt.show()