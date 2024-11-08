#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:30:05 2024

@author: egliott
"""

import numpy as np
import scipy
from scipy.fft import fft2, ifft2,ifftshift
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

fig1, axs1 = plt.subplots(1,1)
fig2, axs2 = plt.subplots(1,1)

tmax = 100 #maximum time
dt = 10**-5 # time step
dt2 = dt**2

imbalance = 0.0 #-0.4 # initial imbalance between the two species (0 = balanced mixture)
C2 = -2.   #  2m(g-g12)rho
eps = 0.01   # Amplitude of the initial random field


Nexp = 110   # Regularization parameter of the potential. Should be integer
        #A higher value  increases accuracy but makes dynamics less stable
        
      

Nt = int(round(tmax/float(dt)))  # Number of time points
t = np.linspace(0,tmax,Nt)

phi2 = np.linspace(0,Nt*dt,Nt) # second moment of order parameter vs time
# phi = np.empty((tmax + 1,Nx,Ny), dtype= np.float32) # registered solution phi(x,y,t) at integer times t

xmax = 75
ymax = 75
Nx = 2**10  #number of grid points along x
Ny = 2**10    #number of grid points along y. In the present code, Nx should equal Ny

x = np.linspace(0,xmax,Nx)
y = np.linspace(0,ymax,Ny)

dx = x[1] - x[0]
dy = y[1] - y[0]



sig = 1
correlation_scale = sig/dx
print('sigma_pix = ', correlation_scale)

lamb_pref = (eps/(sig*xmax))*np.sqrt(3/np.pi)

print('sigma_phys = ', sig)


ini = 'speckle'


print('dx = ', dx)
r0 = np.sqrt(x**2 + y**2)
X, Y = np.meshgrid(x, y, indexing='ij')

kx = ky = np.fft.fftfreq(Nx, d = dx)*2.*np.pi
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

path = '/users/jussieu/egliott/Documents/Coarsening/2D_code/Data_coarsening/'
# path = 'C:\\Users\\elyse\\Documents\\Data_coarsening\\'


file_long = 'IC_test_opti_dt=1e-05_Tmax=100_Nx=1024_Xmax=75_Initial=speckle_Sig=13.653333333333334_Imb=0.0_Eps=0.01_C=-2.0_Nexp=110'
# path_save_long = path + file_long + '.nc'
path_save_long_copy = path + file_long + '_copy'+ '.nc'

try:
    with nc.Dataset(path_save_long_copy,'r') as ds_long:
        t_data_long = ds_long['t_arr'][:]
        varphi_data_long = ds_long['varphi'][:]
        g1 = ds_long['g1'][:]
        phi2D = ds_long['phi_2D'][:]
except Exception as e:
    print('Error',e)   
    
axs1.plot(t_data_long,varphi_data_long)

# axs2.pcolormesh(X,Y,phi2D[0])