#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:11 2023

@author: Nicolas Cherroret
"""

from numba import jit
import numpy as np
import scipy
from scipy.fft import fft2, ifft2,ifftshift
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import time
import scipy.signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from photutils.profiles import RadialProfile
from scipy import interpolate



n_c = 0

tmax = 5*10**-4 #maximum time
dt = 5*10**-4 # time step
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
ymax = 150
Nx = 2**9  #number of grid points along x
Ny = 2**9   #number of grid points along y. In the present code, Nx should equal Ny

x = np.linspace(0,xmax,Nx)
y = np.linspace(0,ymax,Ny)

dx = x[1] - x[0]
dy = y[1] - y[0]




sig = 1
correlation_scale = sig/dx
print('sigma_pix = ', correlation_scale)

lamb_pref = (eps/(sig*xmax))*np.sqrt(3/np.pi)

print('sigma_phys = ', sig)



print('dx = ', dx)
r0 = np.sqrt(x**2 + y**2)
X, Y = np.meshgrid(x, y, indexing='ij')

kx = ky = np.fft.fftfreq(Nx, d = dx)*2.*np.pi
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

q_2 = Kx**2 + Ky**2


def advance_vectorized(u, u_1, u_2, dt2, step1=False):
    
    if step1:
        dt2 = 0.5*dt2  # redefine for the first time step
        D1 = 1.
        D2 = 0
    else:
        D1 = 2.
        D2 = 1.
    
    u_1_hat = fft(u_1)
    u_2_hat = fft(u_2)
    
    F = (np.pi/2 - u_1 - 81*np.pi*u_1**2/238 + 367*u_1**3/714 + 183*np.pi*u_1**4/9520)/(1 - 81*u_1**2/119 + 183*u_1**4/4760)
    
    
    NLpart_hat =  fft2(np.sqrt((1 - u_1**(2 + 2*Nexp))/(1 - u_1**2))*ifft2(q_2*fft2(F)))
    u_hat = (D1*u_1_hat - D2*u_2_hat + dt2*q_2*NLpart_hat)/(1 + C2*dt2*q_2) 
    
    u[:] = ifft2(u_hat).real

    return u


def g_1_avant_moy(phi_loc):
    fourier = fft2(phi_loc)*np.conjugate(fft2(phi_loc))
    res = ifft2(fourier).real
    return res

## Generation of the initial random noise


xx = np.arange(0,Nx//2,1)

n_realization = n_c

int_save = int(1/dt)*5
int_g1_save = int(1/dt)*1
saving = int(0.1/dt)
int_phi2_save = int(0.05/dt)


u = np.zeros((Nx,Ny), dtype=np.float64) # solution array
u_1 = np.zeros((Nx,Ny), dtype=np.float64) # solution at t-dt
u_2 = np.zeros((Nx,Ny), dtype=np.float64) # solution at t-2*dt


np.random.seed(n_c)

noise = np.random.rand(Nx, Ny)*2 -1
noise = lamb_pref*gaussian_filter(noise,correlation_scale, mode = 'wrap',truncate = 8.0)*Nx*2*np.pi*sig**2

print('var= ',np.var(noise))

## Definition of initial state

ini = 'speckle'
u_1_in = imbalance + noise


u_1 = u_1_in

u_1_in_hat = fft2(u_1_in)


n = 0 # Special formula for first time step

u = advance_vectorized(u, u_1, u_2, dt2,step1 = True)
u_2, u_1, u = u_1, u, u_2

u_arr = np.zeros((int(Nt/int_save)+1,Nx,Ny))
# phi2 = [np.sum(u_1**2)/(Nx**2)]
phi2 = [np.var(u_1)]

Cons_N0 = np.sum(u_1)/(Nx)**2
print('Cons = ', Cons_N0)
Cons_N_list = [Cons_N0]

path = '/users/jussieu/egliott/Documents/Coarsening/2D_code/Data_coarsening/'
#path = 'C:\\Users\\elyse\\Documents\\Data_coarsening\\'

name = 'IC_test'
file_name = f"_dt={dt}_Tmax={tmax}_Nx={Nx}_Xmax={xmax}_Initial={ini}_Sig={correlation_scale}_Imb={imbalance}_Eps={eps}_C={C2}_Nexp={Nexp}"
path_save = path + name + file_name + ".npz"

i = 0
t_i = 0
t_data = []

g1_save = np.zeros((int(Nt/int_g1_save)+1,Nx))


j = 0
k = 0

start = time.time()

for i in range(Nt):
    t_i = t_i + dt
    
    if i% int_save == 0:
        print('t = ' + str(round(t_i,3)))
        u_arr[j] = u_1
        j += 1
    if i% int_g1_save == 0:
        g1_t_avant = ifftshift(g_1_avant_moy(u_1))
        g1_t_prof = RadialProfile(g1_t_avant,(Nx//2,Nx//2),xx)
        g1_t_apres = g1_t_prof.profile
        g1_func = interpolate.interp1d(g1_t_prof.radius*dx,g1_t_apres, fill_value = 'extrapolate')
        g1_save[k] = g1_func(r0)
        k += 1
    
    # print(i)
    Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2)/Cons_N0
    # print(Cons_N)
    if i%int_phi2_save == 0:
        Cons_N_list.append(Cons_N)
        phi2.append(np.var(u_1))
        t_data.append(t_i)
    # phi2.append(np.sum(u_1**2)/(Nx**2))   
    if i%saving == 0:
        np.savez(path_save,u_arr,phi2,g1_save,t_data,Cons_N_list)
        print(i)
    if np.isnan(Cons_N):
        print("nan found")
        break
    u = advance_vectorized(u, u_1, u_2, dt2)
    u_2, u_1, u = u_1, u, u_2



end = time.time()
print('Time = ' + str(end -start))
