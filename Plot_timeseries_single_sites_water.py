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

def plot_without_obs(ax1, x, x1, x2, v_IRR_NOI, v_IRR_CTL, v_QIRR_IRR, v_QIRR_CTL, label_IRR_NOI, label_IRR_CTL, label_QIRR_IRR,
                     label_QIRR_CTL, legloc_DIFF, legloc_QIRR, ylabel_DIFF, ylabel_QIRR, xtick_value, xtick_month, title, num, site):
    set_plot_param()
    plt.plot(x, v_IRR_NOI, 'green', marker='v', label=label_IRR_NOI,
             linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_IRR_CTL, 'blue', marker='>', label=label_IRR_CTL,
             linewidth=0.8, markersize=5, alpha=0.5)
    plt.legend(loc=legloc_DIFF, frameon=False, fontsize=10)
    plt.ylabel(ylabel_DIFF, fontsize=10)
    plt.xticks(xtick_value,
               xtick_month,
               )
    plt.tick_params(labelsize=10)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.3, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.3, label=label_QIRR_CTL)
    plt.legend(loc=legloc_QIRR, frameon=False, fontsize=10)
    plt.ylabel(ylabel_QIRR, fontsize=10)
    plt.tick_params(labelsize=10)
    plt.title(title, loc='right')
    plt.title(site, loc='left')
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=12, transform=ax1.transAxes, weight='bold')


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
    plt.tick_params(labelsize=14)
    plt.title(title, loc='right', fontsize=14)
    plt.title(site, loc='left', fontsize=14)
    plt.ylim(0,8)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.2, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.2, label=label_QIRR_CTL)
    plt.legend(loc=legloc_QIRR, frameon=False, fontsize=13)
    plt.ylabel(ylabel_QIRR, fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim(0, 4)
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
    plt.title(title, loc='right', fontsize=14)
    plt.title(site, loc='left', fontsize=14)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.2, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.2, label=label_QIRR_CTL)
    plt.legend(loc=legloc_QIRR, frameon=False, fontsize=10)
    plt.ylabel(ylabel_QIRR, fontsize=10)
    plt.tick_params(labelsize=10)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=14, transform=ax1.transAxes, weight='bold')

def plot_without_obs_nolegend(ax1, x, x1, x2, v_IRR_NOI, v_IRR_CTL, v_QIRR_IRR, v_QIRR_CTL, label_IRR_NOI, label_IRR_CTL, label_QIRR_IRR,
                     label_QIRR_CTL, legloc_DIFF, legloc_QIRR, ylabel_DIFF, ylabel_QIRR, xtick_value, xtick_month, title, num, site):
    set_plot_param()
    plt.plot(x, v_IRR_NOI, 'green', marker='v', label=label_IRR_NOI,
             linewidth=0.8, markersize=5, alpha=0.5)
    plt.plot(x, v_IRR_CTL, 'blue', marker='>', label=label_IRR_CTL,
             linewidth=0.8, markersize=5, alpha=0.5)

    plt.ylabel(ylabel_DIFF, fontsize=10)
    plt.xticks(xtick_value,
               xtick_month,
               )
    plt.tick_params(labelsize=10)
    ax2 = ax1.twinx()
    plt.bar(x1, v_QIRR_IRR, width=0.3, color='green', alpha=0.3, label=label_QIRR_IRR)
    plt.bar(x2, v_QIRR_CTL, width=0.3, color='blue', alpha=0.3, label=label_QIRR_CTL)

    plt.ylabel(ylabel_QIRR, fontsize=10)
    plt.tick_params(labelsize=10)
    plt.title(title, loc='right', fontsize=14)
    plt.title(site, loc='left', fontsize=14)
    ax1.text(-0.15, 1.05, num, color='dimgrey', fontsize=12, transform=ax1.transAxes, weight='bold')


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

def plot_with_obs_without_nolegend(ax1, x, x1, x2, v_NOI, v_CTL, v_IRR, v_QIRR_IRR, v_QIRR_CTL, label_NOI, label_CTL, label_IRR,
                  label_QIRR_IRR, label_QIRR_CTL,legloc_VAR, legloc_QIRR, ylabel_VAR, ylabel_QIRR, xtick_value, xtick_month, title, num, site):
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

str_infile_cas = 'D:\Forcing_tower\Castellaro\\Castellero_monthly.csv'
pd_reader_cas = pd.read_csv(str_infile_cas, header=None)
data_forc_cas = np.array(pd_reader_cas)
le_cas = data_forc_cas[:, 0]


le_month_cas = np.zeros(12)


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

data_NOI_cas = Data_from_nc('Castellaro_noirr_SpGs_QSOIL_monmean')
QSOIL_NOI_cas = data_NOI_cas.load_variable('QSOIL')
data_CTL_cas = Data_from_nc('\Castellaro_drip_SpGs_QSOIL_monmean')
QSOIL_CTL_cas = data_CTL_cas.load_variable('QSOIL')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_QSOIL_monmean')
QSOIL_IRR_cas = data_IRR_cas.load_variable('QSOIL')

QSOIL_NOI_month_cas = np.zeros(12)
QSOIL_CTL_month_cas = np.zeros(12)
QSOIL_IRR_month_cas = np.zeros(12)

data_NOI_cas = Data_from_nc('Castellaro_noirr_SpGs_QVEGE_monmean')
QVEGE_NOI_cas = data_NOI_cas.load_variable('QVEGE')
data_CTL_cas = Data_from_nc('Castellaro_drip_SpGs_QVEGE_monmean')
QVEGE_CTL_cas = data_CTL_cas.load_variable('QVEGE')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_QVEGE_monmean')
QVEGE_IRR_cas = data_IRR_cas.load_variable('QVEGE')

QVEGE_NOI_month_cas = np.zeros(12)
QVEGE_CTL_month_cas = np.zeros(12)
QVEGE_IRR_month_cas = np.zeros(12)

data_NOI_cas = Data_from_nc('Castellaro_noirr_SpGs_QVEGT_monmean')
QVEGT_NOI_cas = data_NOI_cas.load_variable('QVEGT')
data_CTL_cas = Data_from_nc('Castellaro_drip_SpGs_QVEGT_monmean')
QVEGT_CTL_cas = data_CTL_cas.load_variable('QVEGT')
data_IRR_cas = Data_from_nc('Castellaro_flood_SpGs_QVEGT_monmean')
QVEGT_IRR_cas = data_IRR_cas.load_variable('QVEGT')

QVEGT_NOI_month_cas = np.zeros(12)
QVEGT_CTL_month_cas = np.zeros(12)
QVEGT_IRR_month_cas = np.zeros(12)


for year in range(2):
    for month in range(12):
        #if le_cas[12*year + month] > -90:
        le_month_cas[month] = le_month_cas[month] + le_cas[12*year + month]
        EFLX_LH_TOT_NOI_month_cas[month] = EFLX_LH_TOT_NOI_month_cas[month] + EFLX_LH_TOT_NOI_cas[12 * year + month]
        EFLX_LH_TOT_CTL_month_cas[month] = EFLX_LH_TOT_CTL_month_cas[month] + EFLX_LH_TOT_CTL_cas[12 * year + month]
        EFLX_LH_TOT_IRR_month_cas[month] = EFLX_LH_TOT_IRR_month_cas[month] + EFLX_LH_TOT_IRR_cas[12 * year + month]

        QIRRIG_CTL_month_cas[month] = QIRRIG_CTL_month_cas[month] + QIRRIG_CTL_cas[12 * year + month]
        QIRRIG_IRR_month_cas[month] = QIRRIG_IRR_month_cas[month] + QIRRIG_IRR_cas[12 * year + month]



        QSOIL_NOI_month_cas[month] = QSOIL_NOI_month_cas[month] + QSOIL_NOI_cas[12 * year + month]
        QSOIL_CTL_month_cas[month] = QSOIL_CTL_month_cas[month] + QSOIL_CTL_cas[12 * year + month]
        QSOIL_IRR_month_cas[month] = QSOIL_IRR_month_cas[month] + QSOIL_IRR_cas[12 * year + month]



        QVEGE_NOI_month_cas[month] = QVEGE_NOI_month_cas[month] + QVEGE_NOI_cas[12 * year + month]
        QVEGE_CTL_month_cas[month] = QVEGE_CTL_month_cas[month] + QVEGE_CTL_cas[12 * year + month]
        QVEGE_IRR_month_cas[month] = QVEGE_IRR_month_cas[month] + QVEGE_IRR_cas[12 * year + month]



        QVEGT_NOI_month_cas[month] = QVEGT_NOI_month_cas[month] + QVEGT_NOI_cas[12 * year + month]
        QVEGT_CTL_month_cas[month] = QVEGT_CTL_month_cas[month] + QVEGT_CTL_cas[12 * year + month]
        QVEGT_IRR_month_cas[month] = QVEGT_IRR_month_cas[month] + QVEGT_IRR_cas[12 * year + month]
        num_cas[month] = num_cas[month] + 1

le_month_cas = le_month_cas / num_cas / 28.94
EFLX_LH_TOT_NOI_month_cas = EFLX_LH_TOT_NOI_month_cas / num_cas / 28.94
EFLX_LH_TOT_CTL_month_cas = EFLX_LH_TOT_CTL_month_cas / num_cas / 28.94
EFLX_LH_TOT_IRR_month_cas = EFLX_LH_TOT_IRR_month_cas / num_cas / 28.94

QIRRIG_CTL_month_cas = QIRRIG_CTL_month_cas / num_cas
QIRRIG_IRR_month_cas = QIRRIG_IRR_month_cas / num_cas

QSOIL_NOI_month_cas = QSOIL_NOI_month_cas / num_cas * 86400
QSOIL_CTL_month_cas = QSOIL_CTL_month_cas / num_cas * 86400
QSOIL_IRR_month_cas = QSOIL_IRR_month_cas / num_cas * 86400

QVEGE_NOI_month_cas = QVEGE_NOI_month_cas / num_cas * 86400
QVEGE_CTL_month_cas = QVEGE_CTL_month_cas / num_cas * 86400
QVEGE_IRR_month_cas = QVEGE_IRR_month_cas / num_cas * 86400

QVEGT_NOI_month_cas = QVEGT_NOI_month_cas / num_cas * 86400
QVEGT_CTL_month_cas = QVEGT_CTL_month_cas / num_cas * 86400
QVEGT_IRR_month_cas = QVEGT_IRR_month_cas / num_cas * 86400



x = range(1, 13)
x1 = np.array(x) - 0.15
x2 = np.array(x) + 0.15

f = plt.figure(figsize = (16, 10), dpi=1000)
f.subplots_adjust(hspace=0.4, wspace=0.4, left = 0.07, right = 0.95, top = 0.95, bottom = 0.05)
set_plot_param()

ax1 = plt.subplot(3, 4, 5)
plot_with_obs_nolegend(ax1, x, x1, x2, le_month_cas, EFLX_LH_TOT_NOI_month_cas, EFLX_LH_TOT_CTL_month_cas, EFLX_LH_TOT_IRR_month_cas,
                  QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '(mm/day)', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'ET', 'e', 'CAS')

ax1 = plt.subplot(3, 4, 6)
plot_with_obs_without_nolegend(ax1, x, x1, x2, QSOIL_NOI_month_cas, QSOIL_CTL_month_cas, QSOIL_IRR_month_cas,
                  QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'GE', 'f', 'CAS')

ax1 = plt.subplot(3, 4, 7)
plot_with_obs_without_nolegend(ax1, x, x1, x2, QVEGE_NOI_month_cas, QVEGE_CTL_month_cas, QVEGE_IRR_month_cas,
                  QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'CE', 'g', 'CAS')

ax1 = plt.subplot(3, 4, 8)
plot_with_obs_without_nolegend(ax1, x, x1, x2, QVEGT_NOI_month_cas, QVEGT_CTL_month_cas, QVEGT_IRR_month_cas,
                  QIRRIG_IRR_month_cas * 86400, QIRRIG_CTL_month_cas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', 'Qirr (mm/day)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'TR', 'h', 'CAS')

str_infile_neb = 'D:\Forcing_tower\\NE1\\Nebraska_monthly.csv'
pd_reader_neb = pd.read_csv(str_infile_neb, header=None)
data_forc_neb = np.array(pd_reader_neb)
le_neb = data_forc_neb[:, 0]


le_month_neb = np.zeros(12)


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

data_NOI_neb = Data_from_nc('Ne1_noirr_SpGs_QSOIL_monmean')
QSOIL_NOI_neb = data_NOI_neb.load_variable('QSOIL')
data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_QSOIL_monmean')
QSOIL_CTL_neb = data_CTL_neb.load_variable('QSOIL')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_QSOIL_monmean')
QSOIL_IRR_neb = data_IRR_neb.load_variable('QSOIL')

QSOIL_NOI_month_neb = np.zeros(12)
QSOIL_CTL_month_neb = np.zeros(12)
QSOIL_IRR_month_neb = np.zeros(12)

data_NOI_neb = Data_from_nc('Ne1_noirr_SpGs_QVEGE_monmean')
QVEGE_NOI_neb = data_NOI_neb.load_variable('QVEGE')
data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_QVEGE_monmean')
QVEGE_CTL_neb = data_CTL_neb.load_variable('QVEGE')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_QVEGE_monmean')
QVEGE_IRR_neb = data_IRR_neb.load_variable('QVEGE')

QVEGE_NOI_month_neb = np.zeros(12)
QVEGE_CTL_month_neb = np.zeros(12)
QVEGE_IRR_month_neb = np.zeros(12)

data_NOI_neb = Data_from_nc('Ne1_noirr_SpGs_QVEGT_monmean')
QVEGT_NOI_neb = data_NOI_neb.load_variable('QVEGT')
data_CTL_neb = Data_from_nc('Ne1_drip_SpGs_QVEGT_monmean')
QVEGT_CTL_neb = data_CTL_neb.load_variable('QVEGT')
data_IRR_neb = Data_from_nc('Ne1_sprinkler_SpGs_QVEGT_monmean')
QVEGT_IRR_neb = data_IRR_neb.load_variable('QVEGT')

QVEGT_NOI_month_neb = np.zeros(12)
QVEGT_CTL_month_neb = np.zeros(12)
QVEGT_IRR_month_neb = np.zeros(12)


for year in range(13):
    for month in range(12):
        #if le_neb[12*year + month] > -90:
        le_month_neb[month] = le_month_neb[month] + le_neb[12*year + month]
        EFLX_LH_TOT_NOI_month_neb[month] = EFLX_LH_TOT_NOI_month_neb[month] + EFLX_LH_TOT_NOI_neb[12 * year + month]
        EFLX_LH_TOT_CTL_month_neb[month] = EFLX_LH_TOT_CTL_month_neb[month] + EFLX_LH_TOT_CTL_neb[12 * year + month]
        EFLX_LH_TOT_IRR_month_neb[month] = EFLX_LH_TOT_IRR_month_neb[month] + EFLX_LH_TOT_IRR_neb[12 * year + month]

        QIRRIG_CTL_month_neb[month] = QIRRIG_CTL_month_neb[month] + QIRRIG_CTL_neb[12 * year + month]
        QIRRIG_IRR_month_neb[month] = QIRRIG_IRR_month_neb[month] + QIRRIG_IRR_neb[12 * year + month]

        num_neb[month] = num_neb[month] + 1


        QSOIL_NOI_month_neb[month] = QSOIL_NOI_month_neb[month] + QSOIL_NOI_neb[12 * year + month]
        QSOIL_CTL_month_neb[month] = QSOIL_CTL_month_neb[month] + QSOIL_CTL_neb[12 * year + month]
        QSOIL_IRR_month_neb[month] = QSOIL_IRR_month_neb[month] + QSOIL_IRR_neb[12 * year + month]




        QVEGE_NOI_month_neb[month] = QVEGE_NOI_month_neb[month] + QVEGE_NOI_neb[12 * year + month]
        QVEGE_CTL_month_neb[month] = QVEGE_CTL_month_neb[month] + QVEGE_CTL_neb[12 * year + month]
        QVEGE_IRR_month_neb[month] = QVEGE_IRR_month_neb[month] + QVEGE_IRR_neb[12 * year + month]




        QVEGT_NOI_month_neb[month] = QVEGT_NOI_month_neb[month] + QVEGT_NOI_neb[12 * year + month]
        QVEGT_CTL_month_neb[month] = QVEGT_CTL_month_neb[month] + QVEGT_CTL_neb[12 * year + month]
        QVEGT_IRR_month_neb[month] = QVEGT_IRR_month_neb[month] + QVEGT_IRR_neb[12 * year + month]

        num_neb[month] = num_neb[month] + 1

le_month_neb = le_month_neb / num_neb / 28.94
EFLX_LH_TOT_NOI_month_neb = EFLX_LH_TOT_NOI_month_neb / num_neb / 28.94
EFLX_LH_TOT_CTL_month_neb = EFLX_LH_TOT_CTL_month_neb / num_neb / 28.94
EFLX_LH_TOT_IRR_month_neb = EFLX_LH_TOT_IRR_month_neb / num_neb / 28.94


QIRRIG_CTL_month_neb = QIRRIG_CTL_month_neb / num_neb
QIRRIG_IRR_month_neb = QIRRIG_IRR_month_neb / num_neb



QSOIL_NOI_month_neb = QSOIL_NOI_month_neb / num_neb * 86400
QSOIL_CTL_month_neb = QSOIL_CTL_month_neb / num_neb * 86400
QSOIL_IRR_month_neb = QSOIL_IRR_month_neb / num_neb * 86400



QVEGE_NOI_month_neb = QVEGE_NOI_month_neb / num_neb * 86400
QVEGE_CTL_month_neb = QVEGE_CTL_month_neb / num_neb * 86400
QVEGE_IRR_month_neb = QVEGE_IRR_month_neb / num_neb * 86400



QVEGT_NOI_month_neb = QVEGT_NOI_month_neb / num_neb * 86400
QVEGT_CTL_month_neb = QVEGT_CTL_month_neb / num_neb * 86400
QVEGT_IRR_month_neb = QVEGT_IRR_month_neb / num_neb * 86400



x = range(1, 13)
x1 = np.array(x) - 0.15
x2 = np.array(x) + 0.15



ax1 = plt.subplot(3, 4, 1)
plot_with_obs(ax1, x, x1, x2, le_month_neb, EFLX_LH_TOT_NOI_month_neb, EFLX_LH_TOT_CTL_month_neb, EFLX_LH_TOT_IRR_month_neb,
                  QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '(mm/day)', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'ET', 'a', 'NEB')
plt.ylim(0, 6)

ax1 = plt.subplot(3, 4, 2)
plot_with_obs_without_nolegend(ax1, x, x1, x2, QSOIL_NOI_month_neb, QSOIL_CTL_month_neb, QSOIL_IRR_month_neb,
                  QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'GE', 'b', 'NEB')

ax1 = plt.subplot(3, 4, 3)
plot_with_obs_without_nolegend(ax1, x, x1, x2, QVEGE_NOI_month_neb, QVEGE_CTL_month_neb, QVEGE_IRR_month_neb,
                  QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'CE', 'c', 'NEB')

ax1 = plt.subplot(3, 4, 4)
plot_with_obs_without_nolegend(ax1, x, x1, x2, QVEGT_NOI_month_neb, QVEGT_CTL_month_neb, QVEGT_IRR_month_neb,
                  QIRRIG_IRR_month_neb * 86400, QIRRIG_CTL_month_neb * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', 'Qirr (mm/day)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'TR', 'd', 'NEB')

str_infile_mas = 'D:\Forcing_tower\Japan\\Japan_monthly.csv'
pd_reader_mas = pd.read_csv(str_infile_mas, header=None)
data_forc_mas = np.array(pd_reader_mas)
le_mas = data_forc_mas[:, 0]


le_month_mas = np.zeros(12)


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

data_NOI_mas = Data_from_nc('Japan_noirr_SpGs_QSOIL_monmean')
QSOIL_NOI_mas = data_NOI_mas.load_variable('QSOIL')
data_CTL_mas = Data_from_nc('Japan_drip_SpGs_QSOIL_monmean')
QSOIL_CTL_mas = data_CTL_mas.load_variable('QSOIL')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_QSOIL_monmean')
QSOIL_IRR_mas = data_IRR_mas.load_variable('QSOIL')

QSOIL_NOI_month_mas = np.zeros(12)
QSOIL_CTL_month_mas = np.zeros(12)
QSOIL_IRR_month_mas = np.zeros(12)

data_NOI_mas = Data_from_nc('Japan_noirr_SpGs_QVEGE_monmean')
QVEGE_NOI_mas = data_NOI_mas.load_variable('QVEGE')
data_CTL_mas = Data_from_nc('Japan_drip_SpGs_QVEGE_monmean')
QVEGE_CTL_mas = data_CTL_mas.load_variable('QVEGE')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_QVEGE_monmean')
QVEGE_IRR_mas = data_IRR_mas.load_variable('QVEGE')

QVEGE_NOI_month_mas = np.zeros(12)
QVEGE_CTL_month_mas = np.zeros(12)
QVEGE_IRR_month_mas = np.zeros(12)

data_NOI_mas = Data_from_nc('Japan_noirr_SpGs_QVEGT_monmean')
QVEGT_NOI_mas = data_NOI_mas.load_variable('QVEGT')
data_CTL_mas = Data_from_nc('Japan_drip_SpGs_QVEGT_monmean')
QVEGT_CTL_mas = data_CTL_mas.load_variable('QVEGT')
data_IRR_mas = Data_from_nc('Japan_flood_SpGs_QVEGT_monmean')
QVEGT_IRR_mas = data_IRR_mas.load_variable('QVEGT')

QVEGT_NOI_month_mas = np.zeros(12)
QVEGT_CTL_month_mas = np.zeros(12)
QVEGT_IRR_month_mas = np.zeros(12)


for year in range(1):
    for month in range(12):
        if le_mas[12*year + month] > -90:
            le_month_mas[month] = le_month_mas[month] + le_mas[12*year + month]
            EFLX_LH_TOT_NOI_month_mas[month] = EFLX_LH_TOT_NOI_month_mas[month] + EFLX_LH_TOT_NOI_mas[12 * year + month]
            EFLX_LH_TOT_CTL_month_mas[month] = EFLX_LH_TOT_CTL_month_mas[month] + EFLX_LH_TOT_CTL_mas[12 * year + month]
            EFLX_LH_TOT_IRR_month_mas[month] = EFLX_LH_TOT_IRR_month_mas[month] + EFLX_LH_TOT_IRR_mas[12 * year + month]

            QIRRIG_CTL_month_mas[month] = QIRRIG_CTL_month_mas[month] + QIRRIG_CTL_mas[12 * year + month]
            QIRRIG_IRR_month_mas[month] = QIRRIG_IRR_month_mas[month] + QIRRIG_IRR_mas[12 * year + month]


            QSOIL_NOI_month_mas[month] = QSOIL_NOI_month_mas[month] + QSOIL_NOI_mas[12 * year + month]
            QSOIL_CTL_month_mas[month] = QSOIL_CTL_month_mas[month] + QSOIL_CTL_mas[12 * year + month]
            QSOIL_IRR_month_mas[month] = QSOIL_IRR_month_mas[month] + QSOIL_IRR_mas[12 * year + month]


            QVEGE_NOI_month_mas[month] = QVEGE_NOI_month_mas[month] + QVEGE_NOI_mas[12 * year + month]
            QVEGE_CTL_month_mas[month] = QVEGE_CTL_month_mas[month] + QVEGE_CTL_mas[12 * year + month]
            QVEGE_IRR_month_mas[month] = QVEGE_IRR_month_mas[month] + QVEGE_IRR_mas[12 * year + month]


            QVEGT_NOI_month_mas[month] = QVEGT_NOI_month_mas[month] + QVEGT_NOI_mas[12 * year + month]
            QVEGT_CTL_month_mas[month] = QVEGT_CTL_month_mas[month] + QVEGT_CTL_mas[12 * year + month]
            QVEGT_IRR_month_mas[month] = QVEGT_IRR_month_mas[month] + QVEGT_IRR_mas[12 * year + month]

            num_mas[month] = num_mas[month] + 1

le_month_mas = le_month_mas / num_mas / 28.94
EFLX_LH_TOT_NOI_month_mas = EFLX_LH_TOT_NOI_month_mas / num_mas / 28.94
EFLX_LH_TOT_CTL_month_mas = EFLX_LH_TOT_CTL_month_mas / num_mas / 28.94
EFLX_LH_TOT_IRR_month_mas = EFLX_LH_TOT_IRR_month_mas / num_mas / 28.94


QIRRIG_CTL_month_mas = QIRRIG_CTL_month_mas / num_mas
QIRRIG_IRR_month_mas = QIRRIG_IRR_month_mas / num_mas


QSOIL_NOI_month_mas = QSOIL_NOI_month_mas / num_mas * 86400
QSOIL_CTL_month_mas = QSOIL_CTL_month_mas / num_mas * 86400
QSOIL_IRR_month_mas = QSOIL_IRR_month_mas / num_mas * 86400



QVEGE_NOI_month_mas = QVEGE_NOI_month_mas / num_mas * 86400
QVEGE_CTL_month_mas = QVEGE_CTL_month_mas / num_mas * 86400
QVEGE_IRR_month_mas = QVEGE_IRR_month_mas / num_mas * 86400



QVEGT_NOI_month_mas = QVEGT_NOI_month_mas / num_mas * 86400
QVEGT_CTL_month_mas = QVEGT_CTL_month_mas / num_mas * 86400
QVEGT_IRR_month_mas = QVEGT_IRR_month_mas / num_mas * 86400



x = range(1, 13)
x1 = np.array(x) - 0.15
x2 = np.array(x) + 0.15


ax1 = plt.subplot(3, 4, 9)
plot_with_obs_nolegend(ax1, x, x1, x2, le_month_mas, EFLX_LH_TOT_NOI_month_mas, EFLX_LH_TOT_CTL_month_mas, EFLX_LH_TOT_IRR_month_mas,
                  QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'OBS', 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '(mm/day)', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'ET', 'i', 'MAS')

ax1 = plt.subplot(3, 4, 10)
plot_with_obs_without_nolegend(ax1, x, x1, x2, QSOIL_NOI_month_mas, QSOIL_CTL_month_mas, QSOIL_IRR_month_mas,
                  QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'GE', 'j', 'MAS')


ax1 = plt.subplot(3, 4, 11)
plot_with_obs_without_nolegend(ax1, x, x1, x2, QVEGE_NOI_month_mas, QVEGE_CTL_month_mas, QVEGE_IRR_month_mas,
                  QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', '', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'CE', 'k', 'MAS')

ax1 = plt.subplot(3, 4, 12)
plot_with_obs_without_nolegend(ax1, x, x1, x2, QVEGT_NOI_month_mas, QVEGT_CTL_month_mas, QVEGT_IRR_month_mas,
                  QIRRIG_IRR_month_mas * 86400, QIRRIG_CTL_month_mas * 86400, 'NOI', 'CTL', 'IRR',
                  'Qirr_irr', 'Qirr_ctl', 'upper left', 'upper right', '', 'Qirr (mm/day)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [r'J', r'F', r'M', r'A', r'M', 'J', r'J', r'A', r'S', r'O', r'N', r'D'],
                    'TR', 'l', 'MAS')

plt.subplots_adjust(left=0.1,
                    right=0.9,
                    top=0.9,
                    bottom=0.1,
                    wspace=0.35,
                    hspace=0.3)
EFLX_LH_TOT_IRR_CTL_mas = EFLX_LH_TOT_IRR_month_mas - EFLX_LH_TOT_CTL_month_mas
EFLX_LH_TOT_IRR_CTL_cas = EFLX_LH_TOT_IRR_month_cas - EFLX_LH_TOT_CTL_month_cas
EFLX_LH_TOT_IRR_CTL_neb = EFLX_LH_TOT_IRR_month_neb - EFLX_LH_TOT_CTL_month_neb
QSOIL_IRR_CTL_mas = QSOIL_IRR_month_mas - QSOIL_CTL_month_mas
QVEGT_IRR_CTL_mas = QVEGT_IRR_month_mas - QVEGT_CTL_month_mas
QVEGE_IRR_CTL_mas = QVEGE_IRR_month_mas - QVEGE_CTL_month_mas
plt.savefig('Single_point_ET_GE_CE_TR.png')
plt.show()