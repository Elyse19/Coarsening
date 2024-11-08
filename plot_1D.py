#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:49:45 2024

@author: egliott
"""

import numpy as np
import scipy
from scipy.fft import fft2, ifft2, fft, ifft
import matplotlib.pyplot as plt
import time
import scipy.signal
import netCDF4 as nc
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from photutils.profiles import RadialProfile
from numba import jit

fig1, axs1 = plt.subplots(1,1)
# fig2, axs2 = plt.subplots(1,1)

n_c = 1

tmax = 100 #maximum time
dt = 10**-4 # time step
dt2 = dt**2

imbalance = 0.0 #-0.4 # initial imbalance between the two species (0 = balanced mixture)
C2 = -2.   #  2m(g-g12)rho
eps = 0.01   # Amplitude of the initial random field

sig = 1
Nexp = 110   # Regularization parameter of the potential. Should be integer
        #A higher value  increases accuracy but makes dynamics less stable
        

Nt = int(round(tmax/float(dt)))  # Number of time points
t = np.linspace(0,tmax,Nt)

phi2 = np.linspace(0,Nt*dt,Nt) # second moment of order parameter vs time
# phi = np.empty((tmax + 1,Nx,Ny), dtype= np.float32) # registered solution phi(x,y,t) at integer times t

xmax = 75

Nx = 512  #number of grid points along x

x = np.linspace(0,xmax*(1-1/Nx),Nx)

dx = x[1] - x[0]


k = np.fft.fftfreq(Nx, d = dx)*2.*np.pi
k_2 = k**2
dt2_k_2 = k_2*dt2

n_real = 1
ini = 'speckle'

path = '/users/jussieu/egliott/Documents/Coarsening/2D_code/Data_coarsening_1D/'

name = 'testing_1d'
file_name = f"_dt={dt}_Tmax={tmax}_Nx={Nx}_Xmax={xmax}_Initial={ini}_Imb={imbalance}_Eps={eps}_sig={sig}_C={C2}_Nexp={Nexp}_nreal={n_real}_copy"
path_save = path + name + file_name + ".nc"

try:
    with nc.Dataset(path_save,'r') as ds:
        t_data = ds['t_arr'][:]
        varphi_data = ds['varphi'][:]
except Exception as e:
    print('Error',e)        
        
# ds = nc.Dataset(path_save)

# t_data = ds['t_arr'][:]
# varphi_data = ds['varphi'][:]
# phi_x = ds['phi_x'][:]
# g1 = ds['g1'][:]

axs1.plot(t_data,varphi_data)

# axs2.plot(x,phi_x[2])