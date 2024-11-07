#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:11 2023

@author: egliott
"""

import numpy as np
import scipy
from scipy.fft import rfft, irfft
import matplotlib.pyplot as plt
import time
import scipy.signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from photutils.profiles import RadialProfile
from numba import jit

n_c = 1

tmax = 30 #maximum time
dt = 10**-4 # time step
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

xmax = 150

Nx = 800  #number of grid points along x

x = np.linspace(0,xmax*(1-1/Nx),Nx)

dx = x[1] - x[0]


k = np.fft.rfftfreq(Nx, d = dx)*2.*np.pi
k_2 = k**2
dt2_k_2 = k_2*dt2

@jit(nopython=True, parallel=True)
def compute_F(u_1_loc):
    u1_loc_2 = u_1_loc**2
    u1_loc_3 = u1_loc_2*u_1_loc
    u1_loc_4 = u1_loc_2**2
    F = (np.pi/2 - u_1_loc - 81*np.pi*u1_loc_2/238 + 367*u1_loc_3/714 + 183*np.pi*u1_loc_4/9520)/(1 - 81*u1_loc_2/119 + 183*u1_loc_4/4760)
    return F
#Pade approximation for arccos(phi)
#Acceleration with jit (power calculations) and parallelization

@jit(nopython=True, parallel=True)
def sqrt_NL(u_1_loc):
    u1_loc_2 = u_1_loc**2 
    return np.sqrt((1 - u_1_loc**(2 + 2*Nexp))/(1 - u1_loc_2))
#Sqrt part of the nonlinear part
#Acceleration with jit (power calculations) and parallelization

def advance_vectorized_step1(u_1_loc):
    dt2_prime = 0.5*dt2
    
    u_1_hat = rfft(u_1_loc)
    
    F = compute_F(u_1_loc)
    F_hat = rfft(F)
    
    NL_part = sqrt_NL(u_1_loc)*irfft(k_2*F_hat)
    NL_part_hat = rfft(NL_part)
    
    
    u_hat = (1*u_1_hat + dt2_prime*k_2*NL_part_hat)/(1 + C2*dt2_prime*k_2) 
    u = irfft(u_hat).real

    return u

def advance_vectorized(u_1_loc, u_2_loc):
    
    u_1_hat = rfft(u_1_loc)
    u_2_hat = rfft(u_2_loc)
    
    F = compute_F(u_1_loc)
    F_hat = rfft(F)
    
    NL_part = sqrt_NL(u_1_loc)*irfft(k_2*F_hat)
    NL_part_hat = rfft(NL_part)
    
    
    u_hat = (2*u_1_hat - u_2_hat + dt2_k_2*NL_part_hat)/(1 + C2*dt2_k_2) 
    u = irfft(u_hat).real

    return u

def g_1_avant_moy(phi_loc):
    phi_hat = rfft(phi_loc)
    fourier = phi_hat*np.conjugate(phi_hat)
    res = irfft(fourier).real
    return res

## Generation of the initial random noise
sig = 1
correlation_scale = sig/dx


n_realization = n_c

int_save = int(1/dt)*10
int_g1_save = int(0.1/dt)
int_phi2_save = int(0.05/dt)
saving = int(100/dt)


u = np.zeros((Nx), dtype=np.float64) # solution array
u_1 = np.zeros((Nx), dtype=np.float64) # solution at t-dt
u_2 = np.zeros((Nx), dtype=np.float64) # solution at t-2*dt

def ini_u(seed):   
    np.random.seed(seed)
    lamb_pref = eps*(np.sqrt(6)/np.pi**(1/4))/np.sqrt(sig*xmax)
    
    noise = np.random.rand(Nx)*2 -1
    noise = lamb_pref*gaussian_filter(noise,correlation_scale, mode = 'wrap',truncate = 8.0)*np.sqrt(Nx*2*np.pi*sig**2)
    
    return noise






ini = 'speckle'

n_real = 20


u_arr = np.zeros((int(Nt/int_save)+1,Nx))
# phi2 = [np.sum(u_1**2)/(Nx)]
Cons_N0 = np.sum(u_1)/(Nx)**2 #Initial conservation number

path = '/users/jussieu/egliott/Documents/Coarsening/2D_code/Data_coarsening_1D/'

name = 'testing_1d'
file_name = f"_dt={dt}_Tmax={tmax}_Nx={Nx}_Xmax={xmax}_Initial={ini}_Imb={imbalance}_Eps={eps}_C={C2}_Nexp={Nexp}_seed={n_i}"
path_save = path + name + file_name + ".npz"

i = 0
t_i = 0
t_data = []

g1_save = np.zeros((int(Nt/int_g1_save)+1,Nx))


j = 0
k = 0

start = time.time()

phi_2 = np.zeros((int(Nt/int_phi2_save) + 1))

def real(n_real_i):
    t_i = 0

    u_2 = np.zeros((Nx), dtype=np.float64)
    u_1 = ini_u(n_real_i)
    
    u = advance_vectorized_step1(u_1)
    u_2, u_1, u = u_1, u, u_2
    # t_i = t_i + dt
    
    for i in range(Nt):
        t_i = t_i + dt
        
        # if i% int_save == 0:
        #     print('t = ' + str(round(t_i,3)))
        #     u_arr[j] = u_1
        #     j += 1
        # if i% int_g1_save == 0:
        #     g1_t = ifftshift(g_1_avant_moy(u_1))
        #     # g1_func = interpolate.interp1d(x,g1_t, fill_value = 'extrapolate')
        #     g1_save[k] = g1_t
        #     k += 1
        if i%int_phi2_save == 0:
    
            Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2)/Cons_N0
            # phi2.append(np.sum(u_1**2)/(Nx)) 
            # t_data.append(t_i)
            i_save = i//int_phi2_save
            phi_2[i_save] += np.var(u_1)/(n_real + 1)
            if n_real_i == 0:
                t_data.append(t_i)
        
        if i%int_save == 0:
            # np.savez(path_save,u_arr,phi2,g1_save,t_data)
            print(i//int_save)
            
        u = advance_vectorized(u_1, u_2)
        u_2, u_1, u = u_1, u, u_2
        
for n_r_i in range(n_real):
    print('real=', n_r_i)
    real(n_r_i)
        
# t_data = np.arange(0,Nt,dt)    
t_data = np.array(t_data)
np.savez(path_save,u_arr,phi_2,g1_save,t_data)  


end = time.time()
print('Time = ' + str(end -start))

