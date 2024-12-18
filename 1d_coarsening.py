#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:11 2023

@author: egliott
"""

import numpy as np
import scipy
from scipy.fft import rfft, irfft, ifftshift
import matplotlib.pyplot as plt
import time
import scipy.signal
import netCDF4 as nc
from netCDF4 import Dataset
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from numba import jit
import shutil
import os
from scipy import interpolate

tmax = 500 # Maximum time
dt = 10**-6 #Time step
dt2 = dt**2
Nt = int(round(tmax/float(dt)))  # Number of time points

imbalance = 0.0 # Initial imbalance between the two species (0 = balanced mixture)
C2 = -2.   #  2m(g-g12)rho
eps = 0.01   # Amplitude of the initial random field


Nexp = 110   # Regularization parameter of the potential. Should be integer
        #A higher value  increases accuracy but makes dynamics less stable
        
n_real = 64 # Number of disorder realizations


xmax = 75 # Box size
Nx = 2**12  #number of grid points along x
x = np.linspace(0,xmax*(1-1/Nx),Nx) #x grid
dx = x[1] - x[0] 


k = np.fft.rfftfreq(Nx, d = dx)*2.*np.pi 
k_2 = k**2
dt2_k_2 = k_2*dt2

sig = 1
correlation_scale = sig/dx

t_u_save = 5 #Save phi(x,t) every t=t_u_save
t_g1_save = 0.1 #Save g1(x,t) every t=t_g1_save
int_u_save = int(1/dt)*t_u_save
int_g1_save = int(t_g1_save/dt)
int_phi2_save = int(0.05/dt)



@jit(nopython=True)
def compute_F(u_1_loc):
    '''
    Gives the Pade approximant for arccos(phi).
    Acceleration with jit (power calculations).

    Parameters
    ----------
    u_1_loc : 1D array

    Returns
    -------
    F : 1D array

    '''
    u1_loc_2 = u_1_loc**2
    u1_loc_3 = u1_loc_2*u_1_loc
    u1_loc_4 = u1_loc_2**2
    F = (np.pi/2 - u_1_loc - 81*np.pi*u1_loc_2/238 + 367*u1_loc_3/714 + 183*np.pi*u1_loc_4/9520)/(1 - 81*u1_loc_2/119 + 183*u1_loc_4/4760)
    return F



@jit(nopython=True)
def sqrt_NL(u_1_loc):
    '''
    Returns the regularization of 1/\sqrt{1 - phi^2}.
    Regularized using Nexp.
    Acceleration with jit (power calculations).

    Parameters
    ----------
    u_1_loc : 1D array

    Returns
    -------
    1D array

    '''
    u1_loc_2 = u_1_loc**2 
    return np.sqrt((1 - u_1_loc**(2 + 2*Nexp))/(1 - u1_loc_2))


def advance_vectorized_step1(u_1_loc):
    '''
    Calculate the first step of the ODE (half time step) - see notes

    Parameters
    ----------
    u_1_loc : 1D array (phi(t=0))

    Returns
    -------
    u : 1D array (phi(dt))

    '''
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
    '''
    Advance one step in ODE
    Pseudo spectral method with an implicit time integration

    Parameters
    ----------
    u_1_loc : 1D array (phi(t-dt))
    u_2_loc : 1D array (phi(t-2dt))

    Returns
    -------
    u : 1D array (phi(t))

    '''
    
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
    '''
    Calculates g1 function (before averaging)

    Parameters
    ----------
    phi_loc : 1D array 

    Returns
    -------
    res : 1D array

    '''
    phi_hat = rfft(phi_loc)
    fourier = phi_hat*np.conjugate(phi_hat)
    res = irfft(fourier).real
    return res

## Generation of the initial random noise


u = np.zeros((Nx), dtype=np.float64) # solution array
u_1 = np.zeros((Nx), dtype=np.float64) # solution at t-dt
u_2 = np.zeros((Nx), dtype=np.float64) # solution at t-2*dt

def ini_u(seed):   
    '''
    Calculates initial condition for a given seed.
    Gaussian blur using a uniform distribution.
    Fix with \sqrt{Nx}.
    Prefactor such that <\phi^2(0)> = \epsilon^2 (see notes).
    

    Parameters
    ----------
    seed : int

    Returns
    -------
    noise : 1D array (initial condition phi(0))

    '''
    np.random.seed(seed)
    lamb_pref = eps*(np.sqrt(3)/np.pi**(1/4))/np.sqrt(sig*xmax)
    
    noise = np.random.rand(Nx)*2 -1
    noise = lamb_pref*gaussian_filter(noise,correlation_scale, mode = 'wrap',truncate = 8.0)*np.sqrt(Nx*2*np.pi*sig**2)
    
    return noise


path = '/users/jussieu/egliott/Documents/Coarsening/2D_code/Data_coarsening_1D/'

name = 'testing_1d'
file_name = f"_dt={dt}_Tmax={tmax}_Nx={Nx}_Xmax={xmax}_Imb={imbalance}_Eps={eps}_sig={sig}_C={C2}_usave={t_u_save}_g1save={t_g1_save}_Nexp={Nexp}_nreal={n_real}"
file_copy = file_name + '_copy'
file_temp = file_name + '_temp'



def real(real_i):
    '''
    Simulation for 1 disorder realization.

    Parameters
    ----------
    real_i : int (corresponds to seed of realization)

    Returns
    -------
    data : Dictionary with phi_2,g1_save,u_save at different times

    '''
    data = { 'phi_2' : np.zeros((int(Nt/int_phi2_save) + 1)),
            'g1_save' : np.zeros((int(Nt/int_g1_save)+1,Nx)),
            'u_save' : np.zeros((int(Nt/int_u_save)+1,Nx)),
            'u_1_save': np.zeros((int(Nt/int_u_save)+1,Nx))
            }
   
    
    t_i = 0
    
    u_2 = np.zeros((Nx), dtype=np.float64)
    u_1 = ini_u(real_i)
    Cons_N0 = np.sum(u_1)/(Nx)**2
    u = advance_vectorized_step1(u_1)
    u_2, u_1, u = u_1, u, u_2
    
    j = 0
    k = 0
    
    for i in range(Nt):
        t_i = t_i + dt
    
        if i% int_u_save == 0:
            print('t = ' + str(round(t_i,3)))
            data['u_save'][j] = u_1
            # data['u_1_save'][j] = u_2
            j += 1
        if i% int_g1_save == 0:
            g1_t = g_1_avant_moy(u_1)
            g1_func = interpolate.interp1d(x,g1_t, fill_value = 'extrapolate')
            data['g1_save'][k] = g1_t
            k += 1
        if i%int_phi2_save == 0:
            
            # print(t_i)
        
            Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2)/Cons_N0
           
            i_save = i//int_phi2_save
            data['phi_2'][i_save] = np.var(u_1)
            
        if np.isnan(Cons_N):
                print("nan found") #Usually this means that the code is unstable, try with smaller timestep
                break
         
            
        u = advance_vectorized(u_1, u_2)
        u_2, u_1, u = u_1, u, u_2
        
    return data


#This saving system has the advantage of not having to rewrite the whole file each time it is updated (time consuming)
try :
    if os.path.exists(path + name + file_name + '_dis2'+ '.nc'):
        os.remove(path + name + file_name + '_dis2'+'.nc')
    with nc.Dataset(path + name + file_name + '_dis2'+'.nc', 'w',format = 'NETCDF4_CLASSIC') as ds:
    
        ds.createDimension('t_arr', int(Nt/int_phi2_save)+1)     
        ds.createDimension('varphi', int(Nt/int_phi2_save)+1)   
        g1_time_dim = ds.createDimension('g1_time', int(Nt/int_g1_save)+1)   
        g1_value_dim = ds.createDimension('g1_value', Nx)   
        phi_x_time_dim = ds.createDimension('phi_x_time', int(Nt/int_u_save) + 1 )  
        phi_x_value_dim = ds.createDimension('phi_x_value', Nx )  
        #Set file dimensions
        t_arr = ds.createVariable('t_arr', np.float32, ('t_arr',))
        varphi = ds.createVariable('varphi', np.float32, ('varphi',))
        g1 = ds.createVariable('g1',np.float32,('g1_time','g1_value'))
        phi_x = ds.createVariable('phi_x',np.float32,('phi_x_time','phi_x_value'))
        #Set variables
        
        start = time.time()  
                        
        
        res = Parallel(n_jobs=n_real)(delayed (real)(i+25) for i in range(n_real))
      
        
        t_arr[:] = np.arange(0,tmax+0.05,0.05)
        
        
        res_phi2 = np.zeros((n_real,int(Nt/int_phi2_save)+1))
        res_g1 = np.zeros((n_real,int(Nt/int_g1_save)+1,Nx))
        res_u = np.zeros((n_real,int(Nt/int_u_save)+1,Nx))
                         
        for i in range(len(res)):
            res_phi2[i] = res[i]['phi_2']
            res_g1[i] = res[i]['g1_save']
            res_u[i] = res[i]['u_save']
       
        
        var_phi_av = np.sum(res_phi2, axis=0)/n_real
        g1_av = np.sum(res_g1, axis = 0)/n_real
        u_av = np.sum(res_u, axis = 0)/n_real
        
        varphi[:len(var_phi_av)] = var_phi_av
        g1[:] = g1_av
        phi_x[:] = u_av
        
        end = time.time()
        print('Time = ', end-start)
   
except Exception as e:
    print('Error',e)

