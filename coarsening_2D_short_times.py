#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:11 2023

@author: Nicolas Cherroret
"""

import numpy as np
import scipy
from scipy.fft import fft2, ifft2,ifftshift
import matplotlib.pyplot as plt
import time
import scipy.signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
# from joblib import Parallel, delayed
from photutils.profiles import RadialProfile
from scipy import interpolate


start = time.time()
n_c = 0

tmax = 10 #maximum time
dt = 10**-4 # time step
dt2 = dt**2

imbalance = 0.0 #-0.4 # initial imbalance between the two species (0 = balanced mixture)
# C2 = -2.   #  2m(g-g12)rho
eps = 0.01   # Amplitude of the initial random field


Nexp = 110   # Regularization parameter of the potential. Should be integer
        #A higher value  increases accuracy but makes dynamics less stable
        

Nt = int(round(tmax/float(dt)))  # Number of time points
t = np.linspace(0,tmax,Nt)

phi2 = np.linspace(0,Nt*dt,Nt) # second moment of order parameter vs time
# phi = np.empty((tmax + 1,Nx,Ny), dtype= np.float32) # registered solution phi(x,y,t) at integer times t

xmax = 75
ymax = 75
Nx = 256  #number of grid points along x
Ny = 256   #number of grid points along y. In the present code, Nx should equal Ny

x = np.linspace(0,xmax,Nx)
y = np.linspace(0,ymax,Ny)
r0 = np.sqrt(x**2 + y**2)
X, Y = np.meshgrid(x, y, indexing='ij')

dx = x[1] - x[0]
dy = y[1] - y[0]



kx = ky = np.fft.fftfreq(Nx, d = dx)*2.*np.pi
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')



def advance_vectorized(u, u_1, u_2, u_hat, u_1_hat, u_2_hat, dt2, step1=False):
    
    if step1:
        dt2 = 0.5*dt2  # redefine for the first time step
        D1 = 1.
        D2 = 0
    else:
        D1 = 2.
        D2 = 1.
    
    u_1_hat[:] = fft2(u_1)
    u_2_hat[:] = fft2(u_2)
        
    q_2 = Kx**2 + Ky**2
    
    u_hat[:] = D1*u_1_hat - D2*u_2_hat + dt2*q_2*(u_1_hat - (1/4)*q_2*u_1_hat)
    # u_hat[:] = D1*u_1_hat - D2*u_2_hat + dt2*q_2*(C2*u_1 - q_2*u_1)
    
    u[:] = ifft2(u_hat).real

    return u

def g_1_avant_moy(phi_loc):
    fourier = fft2(phi_loc)*np.conjugate(fft2(phi_loc))
    res = ifft2(fourier).real
    return res

## Generation of the initial random noise
sig = 1
correlation_scale = sig/dx
xx = np.arange(0,r0[-1]/2,1)

n_realization = n_c

int_save = int(1/dt)*5
int_g1_save = int(1/dt)*5
saving = int(0.1/dt)

u = np.zeros((Nx,Ny), dtype=np.float64) # solution array
u_1 = np.zeros((Nx,Ny), dtype=np.float64) # solution at t-dt
u_2 = np.zeros((Nx,Ny), dtype=np.float64) # solution at t-2*dt

u_hat = np.zeros((Nx,Ny), dtype=np.complex128)
u_1_hat = np.zeros((Nx,Ny), dtype=np.complex128)
u_2_hat = np.zeros((Nx,Ny), dtype=np.complex128)

seed = 2
np.random.seed(seed)

lamb_pref = (eps/(sig*xmax))*np.sqrt(3/np.pi) #IC prefactor
# lamb_pref = (eps/(xmax))*np.sqrt(3/np.pi)

noise = np.random.rand(Nx, Ny)*2 -1
noise = lamb_pref*gaussian_filter(noise,correlation_scale, mode = 'wrap',truncate = 8.0)*Nx*2*np.pi*sig**2


## Definition of initial state

ini = 'speckle'
u_1_in = imbalance + noise


u_1 = u_1_in

u_1_in_hat = fft2(u_1_in)


n = 0 # Special formula for first time step
u = advance_vectorized(u, u_1, u_2, u_hat, u_1_hat, u_2_hat, dt2, step1=True)
u_2, u_1, u = u_1, u, u_2

u_arr = np.zeros((int(Nt/int_save)+1,Nx,Ny))
# phi2 = [np.sum(u_1**2)/(Nx**2)]
phi2 = [np.var(u_1)]

Cons_N0 = np.sum(u_1)/(Nx)**2
print(Cons_N0)


path = '/users/jussieu/egliott/Documents/Coarsening/codes/Data_coarsening/'
#path = 'C:\\Users\\elyse\\Documents\\Data_coarsening\\'

name = 'short_times'
file_name = f"_dt={dt}_Tmax={tmax}_Nx={Nx}_Xmax={xmax}_Initial={ini}_Sig={sig}_Imb={imbalance}_Eps={eps}"
path_save = path + name + file_name + ".npz"

i = 0
t_i = 0
t_data = []

# g1_save = np.zeros((int(Nt/int_g1_save)+1,Nx))


j = 0
k = 0

for i in range(Nt):
    t_i = t_i + dt
    t_data.append(t_i)
    
    if i% int_save == 0:
        print('t = ' + str(round(t_i,3)))
        u_arr[j] = u_1
        j += 1
    # if i% int_g1_save == 0:
    #     g1_t_avant = ifftshift(g_1_avant_moy(u_1))
    #     g1_t_prof = RadialProfile(g1_t_avant,(Nx//2,Nx//2),xx)
    #     g1_t_apres = g1_t_prof.profile
    #     g1_func = interpolate.interp1d(g1_t_prof.radius,g1_t_apres, fill_value = 'extrapolate')
    #     g1_save[k] = g1_func(r0)
    #     k += 1
    
    # print(i)
    Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2)/Cons_N0
    # print(Cons_N)
    phi2.append(np.var(u_1))
    # phi2.append(np.sum(u_1**2)/(Nx**2))   
    if i%saving == 0:
        np.savez(path_save,u_arr,phi2,t_data)
        print(i)
        print("Cons_N = " + str(Cons_N))
    if np.isnan(Cons_N):
        print("nan found")
        break
    # if np.max(u_1)>1:
    #     print("max")
    #     break
    u = advance_vectorized(u, u_1, u_2, u_hat, u_1_hat, u_2_hat, dt2)
    u_2, u_1, u = u_1, u, u_2
      


end = time.time()
print('Time = ' + str(end -start))
