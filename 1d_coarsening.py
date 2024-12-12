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
from photutils.profiles import RadialProfile
from numba import jit
import shutil
import os
from scipy import interpolate

n_c = 1

tmax = 100 #maximum time
dt = 10**-6 # time step
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

Nx = 4096  #number of grid points along x

x = np.linspace(0,xmax*(1-1/Nx),Nx)

dx = x[1] - x[0]


k = np.fft.rfftfreq(Nx, d = dx)*2.*np.pi
k_2 = k**2
dt2_k_2 = k_2*dt2

x_test = np.linspace(-0.99,0.99,100)

# def arccos_lim(x_loc):
#     if np.abs(x_loc - 1) < 10**-2:
#         return np.sqrt(2*(1-x_loc))
#     if np.abs(x_loc + 1) < 10**-2:
#         return np.sqrt(2*(1+x_loc))
#     else :
#         return np.arccos(x_loc)
    
# res_arc = np.array([arccos_lim(x_i) for x_i in x_test])
# Func_arc = interpolate.interp1d(x_test, res_arc, kind='cubic', fill_value = 'extrapolate')

@jit(nopython=True)
def compute_F(u_1_loc):
    u1_loc_2 = u_1_loc**2
    u1_loc_3 = u1_loc_2*u_1_loc
    u1_loc_4 = u1_loc_2**2
    F = (np.pi/2 - u_1_loc - 81*np.pi*u1_loc_2/238 + 367*u1_loc_3/714 + 183*np.pi*u1_loc_4/9520)/(1 - 81*u1_loc_2/119 + 183*u1_loc_4/4760)
    return F
#Pade approximation for arccos(phi)
#Acceleration with jit (power calculations) and parallelization

@jit(nopython=True)
def sqrt_NL(u_1_loc):
    u1_loc_2 = u_1_loc**2 
    return np.sqrt((1 - u_1_loc**(2 + 2*Nexp))/(1 - u1_loc_2))
#Sqrt part of the nonlinear part
#Acceleration with jit (power calculations) and parallelization

def advance_vectorized_step1(u_1_loc):
    dt2_prime = 0.5*dt2
    
    u_1_hat = rfft(u_1_loc)
    
    F = compute_F(u_1_loc)
    # F = Func_arc(u_1_loc)
    # F = np.arccos(u_1_loc)
    F_hat = rfft(F)
    
    # s_NL = np.sqrt(1 - u_1_loc**2)
    NL_part = sqrt_NL(u_1_loc)*irfft(k_2*F_hat)
    # NL_part = irfft(k_2*F_hat)/s_NL
    NL_part_hat = rfft(NL_part)
    
    
    u_hat = (1*u_1_hat + dt2_prime*k_2*NL_part_hat)/(1 + C2*dt2_prime*k_2) 
    u = irfft(u_hat).real

    return u

def advance_vectorized(u_1_loc, u_2_loc):
    
    u_1_hat = rfft(u_1_loc)
    u_2_hat = rfft(u_2_loc)
    
    # if np.var(u_1_loc)<0.1:
    #     F = np.arccos(u_1_loc)
    #     F_hat = rfft(F)
        
    #     s_NL = np.sqrt(1 - u_1_loc**2)
    #     NL_part = irfft(k_2*F_hat)/s_NL
    # else :
    # F = Func_arc(u_1_loc)
    F = compute_F(u_1_loc)
    # F = np.arccos(u_1_loc)
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

int_u_save = int(1/dt)*5
int_g1_save = int(0.1/dt)
int_phi2_save = int(0.05/dt)
int_save = int(1/dt)*10

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

n_real = 24


# u_arr = np.zeros((int(Nt/int_save)+1,Nx))
# phi2 = [np.sum(u_1**2)/(Nx)]
# Cons_N0 = np.sum(u_1)/(Nx)**2 #Initial conservation number

path = '/users/jussieu/egliott/Documents/Coarsening/2D_code/Data_coarsening_1D/'

name = 'testing_1d'
file_name = f"_dt={dt}_Tmax={tmax}_Nx={Nx}_Xmax={xmax}_Initial={ini}_Imb={imbalance}_Eps={eps}_sig={sig}_C={C2}_Nexp={Nexp}_nreal={n_real}"
file_copy = file_name + '_copy'
file_temp = file_name + '_temp'


i = 0
t_i = 0
t_data = []

# g1_save = np.zeros((int(Nt/int_g1_save)+1,Nx))


j = 0
k = 0

start = time.time()


# phi_2 = np.zeros(n_real,(int(Nt/int_phi2_save) + 1))

def real(real_i):
    data = { 'phi_2' : np.zeros((int(Nt/int_phi2_save) + 1)),
            'g1_save' : np.zeros((int(Nt/int_g1_save)+1,Nx)),
            'u_save' : np.zeros((int(Nt/int_u_save)+1,Nx))
            }
    # phi_2 = np.zeros((int(Nt/int_phi2_save) + 1))
    
    t_i = 0
    
    u_2 = np.zeros((Nx), dtype=np.float64)
    u_1 = ini_u(real_i)
    Cons_N0 = np.sum(u_1)/(Nx)**2
    u = advance_vectorized_step1(u_1)
    u_2, u_1, u = u_1, u, u_2
    # t_i = t_i + dt
    j = 0
    k = 0
    
    for i in range(Nt):
        t_i = t_i + dt
    
        if i% int_u_save == 0:
            print('t = ' + str(round(t_i,3)))
            data['u_save'][j] = u_1
            j += 1
        if i% int_g1_save == 0:
            g1_t = ifftshift(g_1_avant_moy(u_1))
            g1_func = interpolate.interp1d(x,g1_t, fill_value = 'extrapolate')
            data['g1_save'][k] = g1_t
            k += 1
        if i%int_phi2_save == 0:
            
            print(t_i)
        
            Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2)/Cons_N0
            # phi2.append(np.sum(u_1**2)/(Nx)) 
            # t_data.append(t_i)
            i_save = i//int_phi2_save
            data['phi_2'][i_save] = np.var(u_1)
            # varphi[i_save] += np.var(u_1)/n_real
            # t_arr[i_save] = t_i
            # if n_real_i == 0:
            #     # t_data.append(t_i)
            #     t_arr[i_save] = t_i
        
        if i%int_save == 0:
            print(i//int_save)
        #       # np.savez(path_save,u_arr,phi2,g1_save,t_data)
        #       # print(i//int_save)
            # ds.sync()
            # try: 
            #     shutil.copy(path + name + file_name + '.nc', path + name + file_copy + '.nc' )
            #     print('copied')
            # except IOError as e:
            #     print(f'{e}')
            
        u = advance_vectorized(u_1, u_2)
        u_2, u_1, u = u_1, u, u_2
        
    return data


#This saving system has the advantage of not having to rewrite the whole file each time it is updated (time consuming)
try :
    if os.path.exists(path + name + file_name  + '.nc'):
        os.remove(path + name + file_name + '.nc')
    with nc.Dataset(path + name + file_name + '.nc', 'w',format = 'NETCDF4_CLASSIC') as ds:
    
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
        
           
                        
        # for n_r_i in range(n_real):
        #      print('real=', n_r_i)
        #      real(n_r_i)
        # real(0,t_arr,varphi)
        # ds.sync()
        # t_arr[:] = np.arange(int(Nt/int_phi2_save)+1)
        # varphi[:] = np.arange(int(Nt/int_phi2_save)+1)
            
        # os.replace(path + name + file_temp + '.nc',path+ name + file_name + '.nc')
        # res_phi2 = np.array(Parallel(n_jobs=n_real)(delayed (real)(i) for i in range(n_real)))
        # print(res.shape)
        res = Parallel(n_jobs=n_real)(delayed (real)(i) for i in range(n_real))
        # print(res[0])
        
        # res_phi2 = res[:]['phi_2']
        # res_g1 = res[:][1]
        # res_u = res[:][2]
        
        t_arr[:] = np.arange(0,tmax+0.05,0.05)
        
        
        # var_phi_av = 0
        # g1_av = 0
        # u_av = 0
        res_phi2 = np.zeros((n_real,int(Nt/int_phi2_save)+1))
        res_g1 = np.zeros((n_real,int(Nt/int_g1_save)+1,Nx))
        res_u = np.zeros((n_real,int(Nt/int_u_save)+1,Nx))
                         
        for i in range(len(res)):
            res_phi2[i] = res[i]['phi_2']
            res_g1[i] = res[i]['g1_save']
            res_u[i] = res[i]['u_save']
        # res_phi2 = res
        
        var_phi_av = np.sum(res_phi2, axis=0)/n_real
        g1_av = np.sum(res_g1, axis = 0)/n_real
        u_av = np.sum(res_u, axis = 0)/n_real
        
        varphi[:len(var_phi_av)] = var_phi_av
        g1[:] = g1_av
        phi_x[:] = u_av
        # print(ds['phi_x'][0])
        
except Exception as e:
    print('Error',e)

