import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 绘图库
import matplotlib as mpl

def set_plot_param():
    """Set my own customized plotting parameters"""

    mpl.rc('axes', edgecolor='black')
    mpl.rc('axes', labelcolor='dimgrey')
    mpl.rc('xtick', color='dimgrey')
    mpl.rc('xtick', labelsize=16)
    mpl.rc('ytick', color='dimgrey')
    mpl.rc('ytick', labelsize=16)
    mpl.rc('axes', titlesize=18)
    mpl.rc('axes', labelsize=16)
    mpl.rc('legend', fontsize='large')
    mpl.rc('text', color='dimgrey')

csvframe_noi = pd.read_csv("NOI_final.csv",header=None)
csvframe_ctl = pd.read_csv("CTL_final.csv",header=None)
csvframe_irr = pd.read_csv("IRR_final.csv",header=None)
array_noi = np.array(csvframe_noi)
array_ctl = np.array(csvframe_ctl)
array_irr = np.array(csvframe_irr)


LHF_noi = array_noi[:,0]
LHF_ctl = array_ctl[:,0]
LHF_irr = array_irr[:,0]
LHF_ctl_noi = LHF_ctl - LHF_noi
LHF_irr_noi = LHF_irr - LHF_noi
LHF_irr_ctl = LHF_irr - LHF_ctl

SHF_noi = array_noi[:,1]
SHF_ctl = array_ctl[:,1]
SHF_irr = array_irr[:,1]
SHF_ctl_noi = SHF_ctl - SHF_noi
SHF_irr_noi = SHF_irr - SHF_noi
SHF_irr_ctl = SHF_irr - SHF_ctl

LW_noi = array_noi[:,2]
LW_ctl = array_ctl[:,2]
LW_irr = array_irr[:,2]
LW_ctl_noi = LW_ctl - LW_noi
LW_irr_noi = LW_irr - LW_noi
LW_irr_ctl = LW_irr - LW_ctl

SW_noi = array_noi[:,3]
SW_ctl = array_ctl[:,3]
SW_irr = array_irr[:,3]
SW_ctl_noi = SW_ctl - SW_noi
SW_irr_noi = SW_irr - SW_noi
SW_irr_ctl = SW_irr - SW_ctl

E_noi = array_noi[:,4] / 28.94 * 365
E_ctl = array_ctl[:,4] / 28.94 * 365
E_irr = array_irr[:,4] / 28.94 * 365
E_ctl_noi = E_ctl - E_noi
E_irr_noi = E_irr - E_noi
E_irr_ctl = E_irr - E_ctl

T_noi = array_noi[:,5] * 31536000
T_ctl = array_ctl[:,5] * 31536000
T_irr = array_irr[:,5] * 31536000
T_ctl_noi = T_ctl - T_noi
T_irr_noi = T_irr - T_noi
T_irr_ctl = T_irr - T_ctl

R_noi = array_noi[:,6] * 31536000
R_ctl = array_ctl[:,6] * 31536000
R_irr = array_irr[:,6] * 31536000
R_ctl_noi = R_ctl - R_noi
R_irr_noi = R_irr - R_noi
R_irr_ctl = R_irr - R_ctl

SM_noi = array_noi[:,7]
SM_ctl = array_ctl[:,7]
SM_irr = array_irr[:,7]
SM_ctl_noi = SM_ctl - SM_noi
SM_irr_noi = SM_irr - SM_noi
SM_irr_ctl = SM_irr - SM_ctl

plot_LHF_ctl_noi = [LHF_ctl_noi[0], LHF_ctl_noi[7], LHF_ctl_noi[20], LHF_ctl_noi[36], LHF_ctl_noi[38], LHF_ctl_noi[39]]
plot_LHF_irr_noi = [LHF_irr_noi[0], LHF_irr_noi[7], LHF_irr_noi[20], LHF_irr_noi[36], LHF_irr_noi[38], LHF_irr_noi[39]]
plot_LHF_irr_ctl = [LHF_irr_ctl[0], LHF_irr_ctl[7], LHF_irr_ctl[20], LHF_irr_ctl[36], LHF_irr_ctl[38], LHF_irr_ctl[39]]

plot_SHF_ctl_noi = [SHF_ctl_noi[0], SHF_ctl_noi[7], SHF_ctl_noi[20], SHF_ctl_noi[36], SHF_ctl_noi[38], SHF_ctl_noi[39]]
plot_SHF_irr_noi = [SHF_irr_noi[0], SHF_irr_noi[7], SHF_irr_noi[20], SHF_irr_noi[36], SHF_irr_noi[38], SHF_irr_noi[39]]
plot_SHF_irr_ctl = [SHF_irr_ctl[0], SHF_irr_ctl[7], SHF_irr_ctl[20], SHF_irr_ctl[36], SHF_irr_ctl[38], SHF_irr_ctl[39]]

plot_LW_ctl_noi = [LW_ctl_noi[0], LW_ctl_noi[7], LW_ctl_noi[20], LW_ctl_noi[36], LW_ctl_noi[38], LW_ctl_noi[39]]
plot_LW_irr_noi = [LW_irr_noi[0], LW_irr_noi[7], LW_irr_noi[20], LW_irr_noi[36], LW_irr_noi[38], LW_irr_noi[39]]
plot_LW_irr_ctl = [LW_irr_ctl[0], LW_irr_ctl[7], LW_irr_ctl[20], LW_irr_ctl[36], LW_irr_ctl[38], LW_irr_ctl[39]]

plot_SW_ctl_noi = [SW_ctl_noi[0], SW_ctl_noi[7], SW_ctl_noi[20], SW_ctl_noi[36], SW_ctl_noi[38], SW_ctl_noi[39]]
plot_SW_irr_noi = [SW_irr_noi[0], SW_irr_noi[7], SW_irr_noi[20], SW_irr_noi[36], SW_irr_noi[38], SW_irr_noi[39]]
plot_SW_irr_ctl = [SW_irr_ctl[0], SW_irr_ctl[7], SW_irr_ctl[20], SW_irr_ctl[36], SW_irr_ctl[38], SW_irr_ctl[39]]

plot_E_ctl_noi = [E_ctl_noi[0], E_ctl_noi[7], E_ctl_noi[20], E_ctl_noi[36], E_ctl_noi[38], E_ctl_noi[39]]
plot_E_irr_noi = [E_irr_noi[0], E_irr_noi[7], E_irr_noi[20], E_irr_noi[36], E_irr_noi[38], E_irr_noi[39]]
plot_E_irr_ctl = [E_irr_ctl[0], E_irr_ctl[7], E_irr_ctl[20], E_irr_ctl[36], E_irr_ctl[38], E_irr_ctl[39]]

plot_T_ctl_noi = [T_ctl_noi[0], T_ctl_noi[7], T_ctl_noi[20], T_ctl_noi[36], T_ctl_noi[38], T_ctl_noi[39]]
plot_T_irr_noi = [T_irr_noi[0], T_irr_noi[7], T_irr_noi[20], T_irr_noi[36], T_irr_noi[38], T_irr_noi[39]]
plot_T_irr_ctl = [T_irr_ctl[0], T_irr_ctl[7], T_irr_ctl[20], T_irr_ctl[36], T_irr_ctl[38], T_irr_ctl[39]]

plot_R_ctl_noi = [R_ctl_noi[0], R_ctl_noi[7], R_ctl_noi[20], R_ctl_noi[36], R_ctl_noi[38], R_ctl_noi[39]]
plot_R_irr_noi = [R_irr_noi[0], R_irr_noi[7], R_irr_noi[20], R_irr_noi[36], R_irr_noi[38], R_irr_noi[39]]
plot_R_irr_ctl = [R_irr_ctl[0], R_irr_ctl[7], R_irr_ctl[20], R_irr_ctl[36], R_irr_ctl[38], R_irr_ctl[39]]

plot_SM_ctl_noi = [SM_ctl_noi[0], SM_ctl_noi[7], SM_ctl_noi[20], SM_ctl_noi[36], SM_ctl_noi[38], SM_ctl_noi[39]]
plot_SM_irr_noi = [SM_irr_noi[0], SM_irr_noi[7], SM_irr_noi[20], SM_irr_noi[36], SM_irr_noi[38], SM_irr_noi[39]]
plot_SM_irr_ctl = [SM_irr_ctl[0], SM_irr_ctl[7], SM_irr_ctl[20], SM_irr_ctl[36], SM_irr_ctl[38], SM_irr_ctl[39]]



fig = plt.figure(figsize=[18,22], dpi=1000)
fig.subplots_adjust(hspace=0.2, wspace=0.2, left = 0.1, right = 0.95, top = 0.95, bottom = 0.05)
set_plot_param()
ax1 = fig.add_subplot(3,2,2)
ax1.text(0.01,1.02,'b',color='dimgrey',fontsize=20, transform=ax1.transAxes, weight = 'bold')
total_width, n = 0.8, 4
width = total_width / n
x = list(range(6))
for i in range(len(x)):
    x[i] = x[i] -0.4
plt.bar(x, plot_LHF_ctl_noi, width=width, label='LHF',fc = 'tomato')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_SHF_ctl_noi, width=width, label='SHF',fc = 'dodgerblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_SW_ctl_noi, width=width, label='SWup',fc = 'chartreuse')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_LW_ctl_noi, width=width, label='LWup',fc = 'orange')
plt.xticks([0, 1, 2, 3, 4, 5 ], [r'GLO', r'NCA', r'MED', r'EAS', r'SAS', r'SEA'], fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((-4, 7))
plt.legend(loc='upper left', fontsize=20,frameon=False)
ax1.set_title('CTL - NOI',loc='right',fontsize = 22)
plt.ylabel(r"Changes ($\mathregular{W/m^2}$)", fontsize=20)

ax1 = fig.add_subplot(3,2,1)
ax1.text(0.01,1.02,'a',color='dimgrey',fontsize=20, transform=ax1.transAxes, weight = 'bold')
total_width, n = 0.8, 4
width = total_width / n
x = list(range(6))
for i in range(len(x)):
    x[i] = x[i] -0.4
plt.bar(x, plot_E_ctl_noi, width=width, label='E',fc = 'tomato')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_T_ctl_noi, width=width, label='TR',fc = 'dodgerblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_R_ctl_noi, width=width, label='R',fc = 'chartreuse')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_SM_ctl_noi, width=width, label='TSW',fc = 'orange')
plt.xticks([0, 1, 2, 3, 4, 5 ], [r'GLO', r'NCA', r'MED', r'EAS', r'SAS', r'SEA'], fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((-90, 100))
plt.legend(loc='upper left', fontsize=20,frameon=False)
ax1.set_title('CTL - NOI',loc='right',fontsize = 22)
plt.ylabel(r"Changes (mm/year) / ($\mathregular{kg/m^2}$)", fontsize=20)

ax1 = fig.add_subplot(3,2,4)
ax1.text(0.01,1.02,'d',color='dimgrey',fontsize=20, transform=ax1.transAxes, weight = 'bold')
total_width, n = 0.8, 4
width = total_width / n
x = list(range(6))
for i in range(len(x)):
    x[i] = x[i] -0.4
plt.bar(x, plot_LHF_irr_noi, width=width, label='LHF',fc = 'tomato')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_SHF_irr_noi, width=width, label='SHF',fc = 'dodgerblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_SW_irr_noi, width=width, label='SWup',fc = 'chartreuse')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_LW_irr_noi, width=width, label='LWup',fc = 'orange')
plt.xticks([0, 1, 2, 3, 4, 5 ], [r'GLO', r'NCA', r'MED', r'EAS', r'SAS', r'SEA'], fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((-4, 7))
plt.legend(loc='upper left', fontsize=20,frameon=False)
ax1.set_title('IRR - NOI',loc='right',fontsize = 22)
plt.ylabel(r"Changes ($\mathregular{W/m^2}$)", fontsize=20)

ax1 = fig.add_subplot(3,2,6)
ax1.text(0.01,1.02,'f',color='dimgrey',fontsize=20, transform=ax1.transAxes, weight = 'bold')
total_width, n = 0.8, 4
width = total_width / n
x = list(range(6))
for i in range(len(x)):
    x[i] = x[i] -0.4
plt.bar(x, plot_LHF_irr_ctl, width=width, label='LHF',fc = 'tomato')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_SHF_irr_ctl, width=width, label='SHF',fc = 'dodgerblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_SW_irr_ctl, width=width, label='SWup',fc = 'chartreuse')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_LW_irr_ctl, width=width, label='LWup',fc = 'orange')
plt.xticks([0, 1, 2, 3, 4, 5 ], [r'GLO', r'NCA', r'MED', r'EAS', r'SAS', r'SEA'], fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((-4, 7))
plt.legend(loc='upper left', fontsize=20,frameon=False)
ax1.set_title('IRR - CTL',loc='right',fontsize = 22)
plt.ylabel(r"Changes ($\mathregular{W/m^2}$)", fontsize=20)

ax1 = fig.add_subplot(3,2,3)
ax1.text(0.01,1.02,'c',color='dimgrey',fontsize=20, transform=ax1.transAxes, weight = 'bold')
total_width, n = 0.8, 4
width = total_width / n
x = list(range(6))
for i in range(len(x)):
    x[i] = x[i] -0.4
plt.bar(x, plot_E_irr_noi, width=width, label='E',fc = 'tomato')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_T_irr_noi, width=width, label='TR',fc = 'dodgerblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_R_irr_noi, width=width, label='R',fc = 'chartreuse')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_SM_irr_noi, width=width, label='TSW',fc = 'orange')
plt.xticks([0, 1, 2, 3, 4, 5 ], [r'GLO', r'NCA', r'MED', r'EAS', r'SAS', r'SEA'], fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((-90, 100))
plt.legend(loc='upper left', fontsize=20,frameon=False)
ax1.set_title('IRR - NOI',loc='right',fontsize = 22)
plt.ylabel(r"Changes (mm/year) / ($\mathregular{kg/m^2}$)", fontsize=20)

ax1 = fig.add_subplot(3,2,5)
ax1.text(0.01,1.02,'e',color='dimgrey',fontsize=20, transform=ax1.transAxes, weight = 'bold')
total_width, n = 0.8, 4
width = total_width / n
x = list(range(6))
for i in range(len(x)):
    x[i] = x[i] -0.4
plt.bar(x, plot_E_irr_ctl, width=width, label='E',fc = 'tomato')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_T_irr_ctl, width=width, label='TR',fc = 'dodgerblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_R_irr_ctl, width=width, label='R',fc = 'chartreuse')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, plot_SM_irr_ctl, width=width, label='TSW',fc = 'orange')
plt.xticks([0, 1, 2, 3, 4, 5 ], [r'GLO', r'NCA', r'MED', r'EAS', r'SAS', r'SEA'], fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((-90, 100))
plt.legend(loc='upper left', fontsize=20,frameon=False)
ax1.set_title('IRR - CTL',loc='right',fontsize = 22)
plt.ylabel(r"Changes (mm/year) / ($\mathregular{kg/m^2}$)", fontsize=20)

plt.savefig('Region-impacts.png')
#plt.show()
print('test')