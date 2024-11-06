#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:11 2023

@author: egliott
"""

from numba import jit
import numpy as np
import scipy
from scipy.fft import fft2, ifft2,ifftshift, rfft2,irfft2, rfftfreq, fftfreq
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import time
import scipy.signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from photutils.profiles import RadialProfile
from scipy import interpolate


n_c = 8 #number of cores for parallelization

tmax = 30 # maximum time
dt = 10**-5 # time step
dt2 = dt**2

imbalance = 0.0 #-0.4 # initial imbalance between the two species (0 = balanced mixture)
C2 = -2.   #  2m(g-g12)rho
eps = 0.01   # Amplitude of the initial random field


Nexp = 110   # Regularization parameter of the potential. Should be integer
        #A higher value  increases accuracy but makes dynamics less stable

sig = 1 #Physical correlation scale
      
Nt = int(round(tmax/float(dt)))  # Number of time points

xmax = 150 #Box size along x
ymax = 150 #Box size along y
Nx = 2048  #number of grid points along x
Ny = 2048   #number of grid points along y. In the present code, Nx should equal Ny.

x = np.linspace(0,xmax*(1-1/Nx),Nx) #x grid
y = np.linspace(0,ymax*(1-1/Nx),Ny) #y grid

dx = x[1] - x[0]
dy = y[1] - y[0]

correlation_scale = sig/dx #Correlation scale in pixel units (needed for gaussian_filter function)

print('sigma_phys = ', sig)
print('sigma_pix = ', correlation_scale)
print('dx = ', dx)


r0 = np.sqrt(x**2 + y**2)
X, Y = np.meshgrid(x, y, indexing='ij') #2D mesgrid for coordinates x,y

kx = fftfreq(Nx, d = dx)*2.*np.pi 
ky = rfftfreq(Nx, d = dx)*2.*np.pi
#1D kx,ky grids in Fourier space
Kx, Ky = np.meshgrid(kx, ky, indexing='ij') #2D meshgrid in Fourier space

q_2 = Kx**2 + Ky**2

dt2q2 = q_2*dt2


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
    D1 = 1
    dt2_prime = 0.5*dt2 #For the first time step, we take 1/2 of dt2
    
    u_1_hat = rfft2(u_1_loc, workers = n_c)
    
    F = compute_F(u_1_loc)
    F_hat = rfft2(F, workers = n_c)
    
    NL_part = sqrt_NL(u_1_loc)*irfft2(q_2*F_hat, workers = n_c)
    NLpart_hat =  rfft2(NL_part, workers = n_c)
    
    u_hat = (D1*u_1_hat + dt2_prime*q_2*NLpart_hat)/(1 + C2*dt2_prime*q_2) 
    u = irfft2(u_hat, workers = n_c)
    
    return u
#Function used to calculate the the first step of the ode (see notes)
#Takes u(0) and gives u(dt)

def advance_vectorized(u_1_loc, u_2_loc):
    D1 = 2.
    D2 = 1.
    
    u_1_hat = rfft2(u_1_loc, workers = n_c)
    u_2_hat = rfft2(u_2_loc, workers = n_c)
     
    F = compute_F(u_1_loc)
    F_hat = rfft2(F, workers = n_c)
    
    NL_part = sqrt_NL(u_1_loc)*irfft2(q_2*F_hat, workers = n_c)
    NLpart_hat =  rfft2(NL_part, workers = n_c)
    
    u_hat = (D1*u_1_hat - D2*u_2_hat + dt2q2*NLpart_hat)/(1 + C2*dt2q2)  
    u = irfft2(u_hat, workers = n_c)
    
    return u
#Function used to solve the ode(see notes)
#Takes u(t - dt) and u(t - 2dt) and returns u(t)
#Pseudo spectral method with an implicit time integration


xx = np.arange(0,Nx//2,1) #Pixel grid used to calculate g1

int_save = int(1/dt)*5
int_g1_save = int(1/dt)*1
saving = int(0.1/dt)
int_phi2_save = int(0.05/dt)
#Choice of when to save certain variables

u = np.zeros((Nx,Ny), dtype=np.float64) # solution array
u_1 = np.zeros((Nx,Ny), dtype=np.float64) # solution at t-dt
u_2 = np.zeros((Nx,Ny), dtype=np.float64) # solution at t-2*dt

## Generation of the initial random noise

np.random.seed(n_c)

lamb_pref = (eps/(sig*xmax))*np.sqrt(3/np.pi) #IC prefactor

noise = np.random.rand(Nx, Ny)*2 -1
noise = lamb_pref*gaussian_filter(noise,correlation_scale, mode = 'wrap',truncate = 8.0)*Nx*2*np.pi*sig**2
#IC normalized such that <\phi^2(0)> = eps^2

print('var= ',np.var(noise))

## Definition of initial state

ini = 'speckle'
u_1_in = imbalance + noise


u_1 = u_1_in

## Special formula for first time step
u = advance_vectorized_step1(u_1)
u_2, u_1, u = u_1, u, u_2


Cons_N0 = np.sum(u_1)/(Nx)**2 #Initial conservation number
print('Cons = ', Cons_N0)


path = '/users/jussieu/egliott/Documents/Coarsening/2D_code/Data_coarsening/'
#path = 'C:\\Users\\elyse\\Documents\\Data_coarsening\\'
# path = ''

name = 'IC_test_opti'
file_name = f"_dt={dt}_Tmax={tmax}_Nx={Nx}_Xmax={xmax}_Initial={ini}_Sig={correlation_scale}_Imb={imbalance}_Eps={eps}_C={C2}_Nexp={Nexp}"


i = 0
t_i = 0
j = 0
k = 0

start = time.time()

try: ncfile.close()  # just to be safe, make sure dataset is not already open.
except: pass
ncfile = Dataset(path + name + file_name + '.nc',mode='w',format='NETCDF4_CLASSIC') #file creation
#This saving system has the advantage of not having to rewrite the whole file each time it is updated (time consuming)

time_dim = ncfile.createDimension('t_arr', Nt/int_phi2_save)     
phi2_dim = ncfile.createDimension('varphi', Nt/int_phi2_save)   
g1_dim = ncfile.createDimension('g1', (Nt/int_g1_save,Nx))   
phi_2D = ncfile.createDimension('phi_2D', (Nt/int_save,Nx,Ny) )  
#Set file dimensions
t_arr = ncfile.createVariable('t_arr', np.float32, ('t_arr',))
varphi = ncfile.createVariable('varphi', np.float32, ('varphi',))
g1 = ncfile.createVariable('g1',np.float32,('g1',))
phi_2D = ncfile.createVariable('phi_2D',np.float32,('phi_2D',))
#Set variables

for i in range(Nt):
    t_i = t_i + dt
    
    if i% int_save == 0:
        print('t = ' + str(round(t_i,3)))
        phi_2D[j] = u_1 #Save 2D phi
        j += 1
        
    if i% int_g1_save == 0:
        g1_t_avant = ifftshift(g_1_avant_moy(u_1)) #g1 before radial average
        g1_t_prof = RadialProfile(g1_t_avant,(Nx//2,Nx//2),xx) 
        g1_t_apres = g1_t_prof.profile #g1 after radial average for pixel values of xx
        g1_func = interpolate.interp1d(g1_t_prof.radius*dx,g1_t_apres, fill_value = 'extrapolate') #g1 function for real values
        g1[k] = g1_func(r0) #Save g1
        k += 1
    
    if i%int_phi2_save == 0:
        Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2)/Cons_N0 #Check conservation of N
        i_save = i//int_phi2_save
        t_arr[i_save] = t_i #Save time
        varphi[i_save] = np.var(u_1) #Save <\phi^2>
     
    if i%saving == 0:
        ncfile.sync() #Sync values to nc file
        print(i)
        
    if np.isnan(Cons_N):
        print("nan found") #Usually this means that the code is unstable, try with smaller timestep
        break
    
    u = advance_vectorized(u_1, u_2)
    u_2, u_1, u = u_1, u, u_2
    #Advance in ODE
    
ncfile.sync()
end = time.time()
print('Time = ' + str(end -start))

ncfile.close(); print('Dataset is closed!')


