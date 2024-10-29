#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:11 2023

@author: Nicolas Cherroret
"""

from numba import jit
import numpy as np
import scipy
# from scipy.fft import fft2, ifft2,ifftshift
from numpy.fft import fft2, ifft2,ifftshift,fftfreq
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import time
import scipy.signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from photutils.profiles import RadialProfile
from scipy import interpolate
import pyfftw


n_c = 0

tmax = 5 # maximum time
dt = 5*10**-4 # time step
dt2 = dt**2

imbalance = 0.0 #-0.4 # initial imbalance between the two species (0 = balanced mixture)
C2 = -2.   #  2m(g-g12)rho
eps = 0.01   # Amplitude of the initial random field


Nexp = 110   # Regularization parameter of the potential. Should be integer
        #A higher value  increases accuracy but makes dynamics less stable

sig = 1
      

Nt = int(round(tmax/float(dt)))  # Number of time points
t = np.linspace(0,tmax,Nt)

phi2 = np.linspace(0,Nt*dt,Nt) # second moment of order parameter vs time
# phi = np.empty((tmax + 1,Nx,Ny), dtype= np.float32) # registered solution phi(x,y,t) at integer times t

xmax = 150
ymax = 150
Nx = 2**9  #number of grid points along x
Ny = 2**9   #number of grid points along y. In the present code, Nx should equal Ny

x = np.linspace(0,xmax*(1-1/Nx),Nx)
y = np.linspace(0,ymax*(1-1/Nx),Ny)

dx = x[1] - x[0]
dy = y[1] - y[0]


sig_phys = 1
correlation_scale = sig_phys/dx



print('sigma_pix = ', correlation_scale)

lamb_pref = (eps/(sig*xmax))*np.sqrt(3/np.pi)

print('sigma_phys = ', sig)



print('dx = ', dx)
r0 = np.sqrt(x**2 + y**2)
X, Y = np.meshgrid(x, y, indexing='ij')

kx = ky = fftfreq(Nx, d = dx)*2.*np.pi
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

q_2 = Kx**2 + Ky**2

dt2q2 = q_2*dt2


@jit(nopython=True, parallel=True)
def compute_F(u_1_loc):
    u1_loc_2 = u_1_loc**2
    u1_loc_3 = u1_loc_2*u_1_loc
    u1_loc_4 = u1_loc_2**2
    F = (np.pi/2 - u_1_loc - 81*np.pi*u1_loc_2/238 + 367*u1_loc_3/714 + 183*np.pi*u1_loc_4/9520)/(1 - 81*u1_loc_2/119 + 183*u1_loc_4/4760)
    #Pade approximation for arccos(phi)
    return F

@jit(nopython=True, parallel=True)
def sqrt_NL(u_1_loc):
    u1_loc_2 = u_1_loc**2 
    return np.sqrt((1 - u_1_loc**(2 + 2*Nexp))/(1 - u1_loc_2))

# def fftw(data,threads=1):
#     input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
#     output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
#     input_array[:] = data
    
#     fft2_object = pyfftw.FFTW(input_array,output_array, axes = (0,1), direction = 'FFTW_FORWARD', threads = threads)
#     fft2_object()
#     fft_result = output_array.copy()
#     return fft_result

# def ifftw(data,threads=1):
#     input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
#     output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
#     input_array[:] = data
    
#     ifft2_object = pyfftw.FFTW(input_array,output_array, axes = (0,1), direction = 'FFTW_BACKWARD', threads = threads)
#     ifft2_object()
#     ifft_result = output_array.copy()/np.prod(data.shape)
#     return ifft_result
    

def advance_vectorized_step1(u_1_loc):
    D1 = 1
    dt2_prime = 0.5*dt2
    
    u_1_hat = fft2(u_1_loc)
    
    
    F = compute_F(u_1_loc)
    F_hat = fft2(F)
    
    NL_part = sqrt_NL(u_1_loc)*ifft2(q_2*F_hat)
    NLpart_hat =  fft2(NL_part)
    u_hat = (D1*u_1_hat + dt2_prime*q_2*NLpart_hat)/(1 + C2*dt2_prime*q_2) 
    
    u = ifft2(u_hat).real
    
    return u

# def advance_vectorized_step1(u_1_loc):
#     D1 = 1
#     dt2_prime = 0.5*dt2
    
#     u_1_hat = fftw(u_1_loc)
    
    
#     F = compute_F(u_1_loc)
#     F_hat = fftw(F)
    
#     NL_part = sqrt_NL(u_1_loc)*ifftw(q_2*F_hat)
#     NLpart_hat =  fftw(NL_part)
#     u_hat = (D1*u_1_hat + dt2_prime*q_2*NLpart_hat)/(1 + C2*dt2_prime*q_2) 
    
#     u = ifftw(u_hat).real
    
#     return u

def advance_vectorized(u_1_loc, u_2_loc):
    D1 = 2.
    D2 = 1.
    
    u_1_hat = fft2(u_1_loc)
    u_2_hat = fft2(u_2_loc)
     
    F = compute_F(u_1_loc)
    F_hat = fft2(F)
    
    NL_part = sqrt_NL(u_1_loc)*ifft2(q_2*F_hat)
    NLpart_hat =  fft2(NL_part)
    
    u_hat = (D1*u_1_hat - D2*u_2_hat + dt2q2*NLpart_hat)/(1 + C2*dt2q2) 
    
    u = ifft2(u_hat).real
    
    return u

# def advance_vectorized(u_1_loc, u_2_loc):
#     D1 = 2.
#     D2 = 1.
    
#     u_1_hat = fftw(u_1_loc)
#     u_2_hat = fftw(u_2_loc)
     
#     F = compute_F(u_1_loc)
#     F_hat = fftw(F)
    
#     NL_part = sqrt_NL(u_1_loc)*ifftw(q_2*F_hat)
#     NLpart_hat =  fftw(NL_part)
    
#     u_hat = (D1*u_1_hat - D2*u_2_hat + dt2q2*NLpart_hat)/(1 + C2*dt2q2) 
    
#     u = ifftw(u_hat).real
    
#     return u


def g_1_avant_moy(phi_loc):
    phi_hat = fft2(phi_loc)
    fourier = phi_hat*np.conjugate(phi_hat)
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


n = 0 # Special formula for first time step

u = advance_vectorized_step1(u_1)
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
# path_save = path + name + ".hdf5"


i = 0
t_i = 0
t_data = []

g1_save = np.zeros((int(Nt/int_g1_save)+1,Nx))


j = 0
k = 0

start = time.time()

try: ncfile.close()  # just to be safe, make sure dataset is not already open.
except: pass
ncfile = Dataset(path + name + file_name + '.nc',mode='w',format='NETCDF4_CLASSIC') 
time_dim = ncfile.createDimension('t_arr', Nt/int_phi2_save)     
phi2_dim = ncfile.createDimension('varphi', Nt/int_phi2_save)   
t_arr = ncfile.createVariable('t_arr', np.float32, ('t_arr',))
varphi = ncfile.createVariable('varphi', np.float32, ('varphi',))



for i in range(Nt):
    t_i = t_i + dt
    
    if i% int_save == 0:
        print('t = ' + str(round(t_i,3)))
        u_arr[j] = u_1
        j += 1
    # if i% int_g1_save == 0:
    #     g1_t_avant = ifftshift(g_1_avant_moy(u_1))
    #     g1_t_prof = RadialProfile(g1_t_avant,(Nx//2,Nx//2),xx)
    #     g1_t_apres = g1_t_prof.profile
    #     g1_func = interpolate.interp1d(g1_t_prof.radius*dx,g1_t_apres, fill_value = 'extrapolate')
    #     g1_save[k] = g1_func(r0)
    #     k += 1
    
    # print(i)
    # Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2)/Cons_N0
    # print(Cons_N)
    if i%int_phi2_save == 0:
        # Cons_N_list.append(Cons_N)
        # phi2.append(np.var(u_1))
        # t_data.append(t_i)
        Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2)/Cons_N0
        i_save = i//int_phi2_save
        t_arr[i_save] = t_i
        varphi[i_save] = np.var(u_1)
    # phi2.append(np.sum(u_1**2)/(Nx**2))   
    if i%saving == 0:
        # np.savez(path_save,u_arr,phi2,g1_save,t_data,Cons_N_list)
        # np.savez(path_save,phi2,t_data,Cons_N_list)
        print(i)
    if np.isnan(Cons_N):
        print("nan found")
        break
    u = advance_vectorized(u_1, u_2)
    u_2, u_1, u = u_1, u, u_2

end = time.time()
print('Time = ' + str(end -start))

ncfile.close(); print('Dataset is closed!')


