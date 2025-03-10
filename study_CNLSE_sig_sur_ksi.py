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

##Varying sigma/xi_s

fig1, axs1 = plt.subplots(1,1) 
#phi2
# fig2, axs2 = plt.subplots(1,1)
#L(t)
# fig3, axs3 = plt.subplots(1,1) 
#ecart_rho
# fig4, axs4 = plt.subplots(1,1)
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
sig_list = [sigma_base/8,sigma_base/4,sigma_base/2,sigma_base, sigma_base*2, sigma_base*4]

#Faire pour g12/g = 3.02, 1.02
#Faire pour tmax=100 et tmax=1000 avec 10 rep

for i in range(len(n2_list)):
    # if g_ratio_list[i] == '1p02':
    #     L_max=30
    #     nb_step = 100
    color = cmap(i / len(n2_list))
    
    para_name = f'n2_{n2_list[i]}_gRatio{g_ratio_list[i]}_power{puiss}_window{window}_NX{NX}_sig{sigma_base}_steps{nb_step}_L{L_max}.npy'
    path_var = path + name + 'varm_' + para_name
    path_L = path + name + 'L_' + para_name
    phi2 = np.load(path_var,allow_pickle='True')
    L = np.load(path_L,allow_pickle='True')
    # path_save_E = path + name + 'E_field_' + para_name
    # all_E = np.load(path_save_E)
    
    popt, pcov = curve_fit(alg_f, L[0][50:], L[1][50:])
    
    ratio_g12_g11 = 1 - 2/((k0**2)*n2_list[i]*intens*(xi_fix**2))

    Dn_spin = -n2*(ratio_g12_g11-1)*intens/2
    Dn_dens = -n2*(ratio_g12_g11+1)*intens/2
    
    xi_spin = 1/(k0*np.sqrt(Dn_spin))
    xi_dens = 1/(k0*np.sqrt(Dn_dens))
    
    axs1.plot(phi2[0],phi2[1], label = rf'$g12/g$ = {ratio_g12_g11:1.2f}',color = color)
    axs2.plot(L[0],L[1], label = rf'$g12/g$ = {ratio_g12_g11:1.2f}, alpha = {popt[0]:.2f}',color = color)
    
    ecart_v = []
    ecart_rho = []
    
    # time_step = 1
    # z_steps = np.linspace(0,7,nb_step//time_step + 1)
    
    # rho_1 = np.abs(all_E[j,0,:])**2
    # rho_2 = np.abs(all_E[j,1,:])**2
    # rh0 = np.mean(rho_1 + rho_2)
    
    # for j in range(len(z_steps)):
        
    #     rho_1 = np.abs(all_E[j,0,:])**2
    #     rho_2 = np.abs(all_E[j,1,:])**2
    #     # dev = np.mean(np.abs(rho_1 + rho_2 - rho_0)/rho_0)
    #     dev = np.sqrt(np.var(rho_1 + rho_2))/rho_0
    #     ecart_rho.append(dev)
       
    #     speed_x, speed_y = np.array(np.gradient(np.angle(all_E[j]),dx, axis=(-2,-1)))/k0
     
    #     diff_speed_x, diff_speed_y = speed_x[0]-speed_x[1], speed_y[0]-speed_y[1]
    #     diff_speed = diff_speed_x**2 + diff_speed_y**2
    #     sum_speed_x, sum_speed_y = speed_x[0]+speed_x[1], speed_y[0]+speed_y[1]
    #     sum_speed = sum_speed_x**2 + sum_speed_y**2
        
    #     vit_tot = np.mean(sum_speed)
    #     vit_diff = np.mean((diff_speed))
       
    #     ecart_v.append(vit_diff/vit_tot)
    
    # axs3.plot(z_steps,ecart_rho, marker = '.', label = rf'$g12/g$ = {ratio_g12_g11:1.1f}')
    # axs4.plot(z_steps,ecart_v, marker = '.', label = rf'$g12/g$ = {ratio_g12_g11:1.1f}')


axs2.set_xlim(5,100)
# axs2.set_ylim(1,10)
axs2.set_xscale('log')
axs2.set_yscale('log')
axs2.set_xlabel('t/t_NL', fontsize = 'large')
axs2.set_ylabel(r'$L(t)/\xi_s$', fontsize = 'large')
axs2.legend()
# axs3.set_yscale('log')
axs3.set_xlabel('t/t_NL')
axs3.set_ylabel(r'$<|\rho_1 + \rho_2 -\rho_0|>/\rho_0$')
axs3.legend()

axs4.axvline(x = 7, color = 'k', ls = '--')

# axs4.set_xscale('log')
axs4.set_yscale('log')
axs4.set_xlabel('t/t_NL')
axs4.set_ylabel(r'$\sigma(\Delta v)/\sigma(v)$')
axs4.legend()

axs1.plot(0,0,color = 'k', label = 'CNLSE')
    
##Results hydro : Nx=1400 and Nx=1800
    
file_0 = f'Coarsening_dt=4.9999999999999996e-06_Tmax=30_Nx=1800_Xmax=75_Sig={1}_Imb=0.0_Eps=0.01_C=-1.0_Nexp=110_usave=5_g1save=2_seed=2.nc'
path_save_0_copy = path + file_0 
    
try:
    with nc.Dataset(path_save_0_copy,'r') as ds0:
        t_data_0 = ds0['t_arr'][:]
        varphi_data_0 = ds0['varphi'][:]
        g1_0 = ds0['g1'][:]
        phi2D_0 = ds0['phi_2D'][:]
except Exception as e:
    print('Error',e)  

for i in range(len(g1_0)):
    g1_0[i] = g1_0[i]/g1_0[i][0]
    
# axs1.plot(t_data_0,varphi_data_0, color = 'k', ls = '--',label = 'Effective theory')
    

file_1 = f'Coarsening_dt=4.9999999999999996e-06_Tmax=100_Nx=1400_Xmax=75_Sig={1}_Imb=0.0_Eps=0.01_C=-1.0_Nexp=110_usave=5_g1save=1_seed=2_copy_v2.nc'
path_save_1_copy = path + file_1 
    
try:
    with nc.Dataset(path_save_1_copy,'r') as ds1:
        t_data_1 = ds1['t_arr'][:]
        varphi_data_1 = ds1['varphi'][:]
        g1_1 = ds1['g1'][:]
        phi2D_1 = ds1['phi_2D'][:]
except Exception as e:
    print('Error',e)  
    
for i in range(len(g1_1)):
    g1_1[i] = g1_1[i]/g1_1[i][0]
    
# axs1.plot(t_data_1,varphi_data_1, color = 'k', ls = '--',label = 'Effective theory')

name = 'Coarsening'
seed_list = [0,2,4]
varphi_real = np.zeros((len(seed_list),int(100/0.05) + 1))

for i_s in range(len(seed_list)): 

    if seed_list[i_s] == 2:
        file_name = f"_dt={4.9999999999999996e-06}_Tmax={100}_Nx={1400}_Xmax={75}_Sig={1}_Imb={0.0}_Eps={0.01}_C={-1.0}_Nexp={110}_usave={5}_g1save={1}_seed={seed_list[i_s]}"
        path_save = path + name + file_name + '_copy_v2.nc'
    else :
        file_name = f"_dt={4.9999999999999996e-06}_Tmax={100}_Nx={1400}_Xmax={75}_Sig={1}_Imb={0.0}_Eps={0.01}_Nexp={110}_usave={5}_g1save={1}_seed={seed_list[i_s]}"
        path_save = path + name + file_name + '_copy.nc'

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
       
axs1.plot(t_data,varphi_av,color = 'k', ls = '--',label = 'Effective theory')
axs2.plot(t_g,L_t,color = 'k', ls = '--',label = 'Effective theory')

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

axs1.plot(tt,sol0,ls = 'dashdot', color = 'silver', zorder = 2, label = 'Linearized solution')



axs1.set_xlim(0,25)
axs1.set_ylim(-1e-2,0.71)
# axs1.set_yscale('log')
axs1.set_xlabel(r'$t/t_{NL}$', fontsize = 'large')
axs1.set_ylabel(r'$<\phi^2(t)>$', fontsize = 'large')
axs1.set_title(r'$\sigma/\xi_s = 1$')
axs1.legend(loc = 'lower right')
