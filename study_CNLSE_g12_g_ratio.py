#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:02:56 2025

@author: egliott
"""

import numpy as np
import scipy
from scipy.fft import fft2, ifft2,ifftshift, rfft2,irfft2, rfftfreq, fftfreq
import matplotlib.pyplot as plt
import time
import scipy.signal
from scipy import signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import netCDF4 as nc
from netCDF4 import Dataset
plt.rcParams ['figure.dpi'] = 300
import shutil
import xarray as xr
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from photutils.profiles import RadialProfile
import numba
from scipy.signal import find_peaks

##Varying g12/g

# fig1, axs1 = plt.subplots(1,1) 
#phi2
# fig2, axs2 = plt.subplots(1,1)
#L(t)
# fig3, axs3 = plt.subplots(1,1) 
#ecart_rho
fig4, axs4 = plt.subplots(1,1)
#ecart_v 

nb_step = 500
L_max = 100
NX = 1024
waist = 1e-3
window = 4*waist
puiss = 2
intens = puiss/window**2
dx = window/1024
k0 = 2*np.pi/(780e-9)


path = '/users/jussieu/egliott/Documents/Coarsening/codes/Data_coarsening/'
name = 'param_test'

def alg_f(t_loc,alpha_loc,C_loc):
    return C_loc*(t_loc**alpha_loc)

def ind_premiere_annulation(tab):
    ind = 0
    while(ind < len(tab) and tab[ind] > 0):
        ind += 1
    return ind if ind < len(tab) else -1

def annulation_g1(tab,r0_loc):
    t_annul = []
    for i in range(len(tab)):
        ind_annulation = ind_premiere_annulation(tab[i])
        if ind_annulation == -1:
            t_annul.append(np.nan)  
        else:
            t_annul.append(r0_loc[ind_annulation])
    return np.array(t_annul)

# colors = ['red','pink','purple','blue','cyan','green']
cmap = plt.cm.plasma

sigma_base = 1.5625e-05 #Correspond à 4*dx

##Results for sigma/xi_s = 1 fixed, varying xi_d/xi_s


# n2_list = [-5e-10,-1e-9,-1.26e-9,-1.68e-9,-2.5e-9,-5e-9,-1e-8, -5e-8]
# g_ratio_list = ['3p02','2p01','1p8','1p6','1p4','1p2','1p1','1p02']

n2_list = [-5.05e-10,-1.01e-9,-1.26e-9,-1.68e-9,-2.5e-9,-5e-9,-1e-8]
g_ratio_list = ['3p0','2p0','1p8','1p6','1p4','1p2','1p1']

# n2_list = [-5e-10,-1.26e-9,-5e-9]
# g_ratio_list = ['3p02','1p8','1p2']
#Lmax=1000, rep = 10

# n2_list = [-5e-10,-1.5e-9,-5e-9]
# g_ratio_list = ['3p02','1p67','1p2']
# L_max=1000, rep = 1

# n2_list = [-5e-10]
# g_ratio_list = ['3p02']

xi_fix = sigma_base/1

lines = []

for i in range(len(n2_list)):
    # if g_ratio_list[i] == '3p02' or g_ratio_list[i] == '1p8' or g_ratio_list[i] == '1p2':
    #     para_name = f'n2_{n2_list[i]}_gRatio{g_ratio_list[i]}_power{puiss}_window{window}_NX{NX}_sig{sigma_base}_steps{nb_step}_L{1e3}_rep={10}.npy'
    para_name = f'n2_{n2_list[i]}_gRatio{g_ratio_list[i]}_power{puiss}_window{window}_NX{NX}_sig{sigma_base}_steps{500}_L{100}_rep={8}.npy'

    color = cmap(i / len(n2_list))

    path_var = path + name + 'varm_' + para_name
    path_L = path + name + 'L_' + para_name
    path_dens = path + name + 'dens_' + para_name
    path_dens0 = path + name + 'dens0_' + para_name
    path_s_speed = path + name + 's_speed_' + para_name
    path_d_speed = path + name + 'd_speed_' + para_name

    phi2 = np.load(path_var,allow_pickle='True')
    L = np.load(path_L,allow_pickle='True')
    dens = np.load(path_dens, allow_pickle='True')
    dens0 = np.load(path_dens0, allow_pickle='True')
    s_speed = np.load(path_s_speed,allow_pickle='True')
    d_speed = np.load(path_d_speed,allow_pickle='True')

    
    popt, pcov = curve_fit(alg_f, L[0][50:], L[1][50:])
    
    ratio_g12_g11 = 1 - 2/((k0**2)*n2_list[i]*intens*(xi_fix**2))

    
    line, = axs1.plot(phi2[0],phi2[1], label = rf'$g12/g$ = {ratio_g12_g11:1.2f}',color = color)
    lines.append(line)
    axs2.plot(L[0],L[1], label = rf'$g12/g$ = {ratio_g12_g11:1.2f}, alpha = {popt[0]:.2f}, c0 = {popt[1]:.2f}',color = color)
    
    ecart_v = []
    ecart_rho = []
    
    rh0 = dens0[1][0]
    z_steps = dens[0]
    
    for j in range(len(dens[0])):
        ecart_rho.append(dens[1][j]/rh0)
        ecart_v.append(d_speed[1][j]/s_speed[1][j])
    axs3.plot(z_steps,ecart_rho, label = rf'$g12/g$ = {ratio_g12_g11:1.2f}', color = color)
    axs4.plot(z_steps,ecart_v, label = rf'$g12/g$ = {ratio_g12_g11:1.2f}',color = color)


axs2.set_xlim(0.5,100)
# axs2.set_ylim(1,10)
axs2.set_xscale('log')
axs2.set_yscale('log')
axs2.set_xlabel('t/t_NL', fontsize = 'large')
axs2.set_ylabel(r'$L(t)/\xi_s$', fontsize = 'large')
axs2.legend()
# axs3.set_yscale('log')
axs3.set_xlabel('t/t_NL', fontsize = 'large')
axs3.set_ylabel(r'$\sigma(\rho_1 + \rho_2)/\rho_0$', fontsize = 'large')
axs3.legend()

# axs4.axvline(x = 7, color = 'k', ls = '--')

# axs4.set_xscale('log')
axs4.set_xlim(0,35)
axs4.set_ylim(0.1,75)
axs4.set_yscale('log')
axs4.set_xlabel('t/t_NL', fontsize = 'large')
axs4.set_ylabel(r'$\sigma(v_1 - v_2)/\sigma(v_1 + v_2)$', fontsize = 'large')
axs4.legend()

cnlse, = axs1.plot(0,0,color = 'k', label = 'CNLSE')
    
##Results hydro : Nx=1800
    
# file_0 = f'Coarsening_dt=4.9999999999999996e-06_Tmax=30_Nx=1800_Xmax=75_Sig={1}_Imb=0.0_Eps=0.01_C=-1.0_Nexp=110_usave=5_g1save=2_seed=2.nc'
# path_save_0_copy = path + file_0 
    
# try:
#     with nc.Dataset(path_save_0_copy,'r') as ds0:
#         t_data_0 = ds0['t_arr'][:]
#         varphi_data_0 = ds0['varphi'][:]
#         g1_0 = ds0['g1'][:]
#         phi2D_0 = ds0['phi_2D'][:]
# except Exception as e:
#     print('Error',e)  

# for i in range(len(g1_0)):
#     g1_0[i] = g1_0[i]/g1_0[i][0]
    
# axs1.plot(t_data_0,varphi_data_0, color = 'k', ls = '--',label = 'Effective theory')
    

##Results hydro : Nx=1400, 8 real

name = 'Coarsening'
seed_list = [0,2,4,6,8,10,12,14]
# seed_list = [0,2,4]
varphi_real = np.zeros((len(seed_list),int(100/0.05) + 1))
g1_real = np.zeros((len(seed_list),101,1400))

for i_s in range(len(seed_list)): 
    u_save = 5
    version = ''
    old = ''
    if seed_list[i_s] == 2:
        version = '_v2'
        old = f'_C={-1.0}'
    if seed_list[i_s] == 6 or seed_list[i_s] == 8 or seed_list[i_s] == 10:
        u_save = 1
        version = '_v4'
    
    file_name = f"_dt={4.9999999999999996e-06}_Tmax={100}_Nx={1400}_Xmax={75}_Sig={1}_Imb={0.0}_Eps={0.01}" + old + f"_Nexp={110}_usave={u_save}_g1save={1}_seed={seed_list[i_s]}"
    path_save = path + name + file_name + '_copy' + version + '.nc'

    try:
        with nc.Dataset(path_save,'r') as ds:
            t_data= ds['t_arr'][:]
            varphi= ds['varphi'][:]
            g1 = ds['g1'][:]
            phi2D= ds['phi_2D'][:]
    except Exception as e:
        print('Error',e) 
        
    varphi_real[i_s] = varphi
    
    for i in range(len(g1)):
        g1[i] = g1[i]/g1[i][0]
        
    g1_real[i_s] = g1

varphi_av = np.mean(varphi_real, axis = 0)
varphi_err = np.std(varphi_real, axis = 0)
g1_av = np.mean(g1_real,axis = 0)

x = np.linspace(0,75,1400)
y = np.linspace(0,75,1400)
r0 = np.sqrt(x**2 + y**2)
N_prec = 10000
g1_prec = np.zeros((len(g1_av),N_prec))
r0_prec = np.linspace(0,xmax/2,N_prec)

for i in range(len(g1_av)):
    g1_func = interpolate.interp1d(r0,g1_av[i], fill_value = 'extrapolate')
    g1_prec[i] = g1_func(r0_prec)
    
L_t = annulation_g1(g1_prec,r0_prec)
t_g = np.arange(0,101,1)
       
eff, = axs1.plot(t_data,varphi_av,color = 'k', ls = '--',label = 'Effective theory')
error_band = axs1.fill_between(t_data, varphi_av - varphi_err, varphi_av + varphi_err, color='silver', alpha=0.3, label='Error Band')
axs2.plot(t_g,L_t,color = 'k', ls = '--',label = 'Effective theory')

# axs2.plot(t_g,0.55*t_g**(2/3), color = 'g')
tt = np.arange(0,1e3,0.1)
# axs2.plot(tt,0.35*tt**(2/3),color = 'g')

##Analytical solution
sig_sur_ksi = 1
k_arr = np.linspace(0,2,2**10)
k_arr_inf = np.linspace(2,10**1,2**10)
eps = 0.01
t_test = np.arange(0,10,0.1)
tt = np.arange(0,6,0.1)

def sol_analytique0(t_loc,sig_sur_ksi):
    integrand1 = k_arr*np.exp(-(k_arr*sig_sur_ksi)**2)*(np.cosh(k_arr*np.sqrt(1 - k_arr**2/4)*t_loc)**2)
    integral1 = integrate.simpson(integrand1,k_arr)
    
    integrand2 = k_arr_inf*np.exp(-(k_arr_inf*sig_sur_ksi)**2)*(np.cos(k_arr_inf*np.sqrt(k_arr_inf**2/4-1)*t_loc)**2)
    integral2 = integrate.simpson(integrand2,k_arr_inf)
    
    
    integral = integral1 + integral2
    return (eps**2)*2*integral*(sig_sur_ksi**2)


sol0 = []
for t_i in tt:
    sol0.append(sol_analytique0(t_i,sig_sur_ksi))

sol0 = np.array(sol0)

lin, = axs1.plot(tt,sol0,ls = 'dashdot', color = '#d62728', zorder = 2, label = 'Linearized solution')

legend1 = axs1.legend(handles=lines, loc='lower right')
legend2 = axs1.legend(handles=[cnlse, eff,error_band,lin], loc=(0.27,0.025))

# Add the legends to the plot
axs1.add_artist(legend1)
axs1.add_artist(legend2)

axs1.set_xlim(0,35)
axs1.set_ylim(-1e-2,0.75)
# axs1.set_yscale('log')
axs1.set_xlabel(r'$t/t_{NL}$', fontsize = 'large')
axs1.set_ylabel(r'$<\phi^2(t)>$', fontsize = 'large')
axs1.set_title(r'$\sigma/\xi_s = 1$')
# axs1.legend(loc = 'lower right')
