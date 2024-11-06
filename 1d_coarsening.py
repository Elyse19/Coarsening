#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:11 2023

@author: Nicolas Cherroret
"""

# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import numpy as np
import scipy
from scipy.fft import fft2, ifft2, fft, ifft
import matplotlib.pyplot as plt
import time
import scipy.signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from photutils.profiles import RadialProfile

fig1, axs1 = plt.subplots(1,1)

start = time.time()
n_c = 1

tmax = 500 #maximum time
dt = 0.0001 # time step
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

Nx = 400  #number of grid points along x


x = np.linspace(0,xmax,Nx)


dx = x[1] - x[0]


kx = np.fft.fftfreq(Nx, d = dx)*2.*np.pi



def advance_vectorized(u, u_1, u_2, u_hat, u_1_hat, u_2_hat, dt2, step1=False):
    
    if step1:
        dt2 = 0.5*dt2  # redefine for the first time step
        D1 = 1.
        D2 = 0
    else:
        D1 = 2.
        D2 = 1.
    
    u_1_hat[:] = fft(u_1)
    u_2_hat[:] = fft(u_2)
    
    F = (np.pi/2 - u_1 - 81*np.pi*u_1**2/238 + 367*u_1**3/714 + 183*np.pi*u_1**4/9520)/(1 - 81*u_1**2/119 + 183*u_1**4/4760)
    
    q_2 = kx**2
    
    NLpart_hat = fft(np.sqrt((1 - u_1**(2 + 2*Nexp))/(1 - u_1**2))*ifft(q_2*fft(F)))
    u_hat[:] = (D1*u_1_hat - D2*u_2_hat + dt2*q_2*NLpart_hat)/(1 + (C2)*dt2*q_2) 
    
    u[:] = ifft(u_hat).real

    return u

def g_1_avant_moy(phi_loc):
    fourier = fft(phi_loc)*np.conjugate(fft(phi_loc))
    res = ifft(fourier).real
    return res

## Generation of the initial random noise

correlation_scale = 3


n_realization = n_c

int_save = int(1/dt)*10
int_g1_save = int(0.1/dt)
saving = int(100/dt)


u = np.zeros((Nx), dtype=np.float64) # solution array
u_1 = np.zeros((Nx), dtype=np.float64) # solution at t-dt
u_2 = np.zeros((Nx), dtype=np.float64) # solution at t-2*dt

u_hat = np.zeros((Nx), dtype=np.complex128)
u_1_hat = np.zeros((Nx), dtype=np.complex128)
u_2_hat = np.zeros((Nx), dtype=np.complex128)

n_i = 6

np.random.seed(n_i)

noise = np.random.rand(Nx)*2 -1
noise = eps*gaussian_filter(noise,correlation_scale)

# axs1.plot(x,noise)
## Definition of initial state

ini = 'speckle'
u_1_in = imbalance + noise


u_1 = u_1_in

u_1_in_hat = fft(u_1_in)


n_i = 6

n = 0 # Special formula for first time step
u = advance_vectorized(u, u_1, u_2, u_hat, u_1_hat, u_2_hat, dt2, step1=True)
u_2, u_1, u = u_1, u, u_2

u_arr = np.zeros((int(Nt/int_save)+1,Nx))
phi2 = [np.sum(u_1**2)/(Nx)]


path = '/users/jussieu/egliott/Documents/Coarsening/2D_code/Data_coarsening/'

name = 'testing_1d'
file_name = f"_dt={dt}_Tmax={tmax}_Nx={Nx}_Xmax={xmax}_Initial={ini}_Imb={imbalance}_Eps={eps}_C={C2}_Nexp={Nexp}_seed={n_i}"
path_save = path + name + file_name + ".npz"

i = 0
t_i = 0
t_data = []

g1_save = np.zeros((int(Nt/int_g1_save)+1,Nx))


j = 0
k = 0

for i in range(Nt):
    t_i = t_i + dt
    t_data.append(t_i)
    
    if i% int_save == 0:
        print('t = ' + str(round(t_i,3)))
        u_arr[j] = u_1
        j += 1
    if i% int_g1_save == 0:
        g1_t = ifftshift(g_1_avant_moy(u_1))
        # g1_func = interpolate.interp1d(x,g1_t, fill_value = 'extrapolate')
        g1_save[k] = g1_t
        k += 1
    
    # print(i)
    # Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2)/Cons_N0
  
    phi2.append(np.sum(u_1**2)/(Nx))   
    if i%saving == 0:
        np.savez(path_save,u_arr,phi2,g1_save,t_data)
        print(i)
        # plt.plot(x,u_1)
        plt.show()
    u = advance_vectorized(u, u_1, u_2, u_hat, u_1_hat, u_2_hat, dt2)
    u_2, u_1, u = u_1, u, u_2
    
np.savez(path_save,u_arr,phi2,g1_save,t_data)  
# axs1.plot(x,u_1)

end = time.time()
print('Time = ' + str(end -start))

