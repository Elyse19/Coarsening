#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:11 2023

@author: egliott
"""

from numba import jit
import numpy as np
import scipy
from scipy.fft import fft2, ifft2, ifftshift, rfft2, irfft2, rfftfreq, fftfreq
import netCDF4 as nc
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import time
import scipy.signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from photutils.profiles import RadialProfile
from scipy import interpolate
import os
import shutil
import sys
from mpmath import *

#sys.path.append(
 #   '/users/jussieu/egliott/Documents/Coarsening/codes/Data_coarsening')

n_c = 8  # number of cores for parallelization

tmax = 100  # maximum time
dt = 5*10**-6  # time step
dt2 = dt**2

# -0.4 # initial imbalance between the two species (0 = balanced mixture)
imbalance = 0.0
C2 = -1.  # 2m(g-g12)rho
eps = 0.01   # Amplitude of the initial random field


Nexp = 110   # Regularization parameter of the potential. Should be integer
# A higher value  increases accuracy but makes dynamics less stable


Nt = int(round(tmax/float(dt)))  # Number of time points

xmax = 75  # Box size along x
ymax = 75  # Box size along y
Nx = 1400  # number of grid points along x
# number of grid points along y. In the present code, Nx should equal Ny.
Ny = 1400

x = np.linspace(0, xmax*(1-1/Nx), Nx)  # x grid
y = np.linspace(0, ymax*(1-1/Nx), Ny)  # y grid

dx = x[1] - x[0]
dy = y[1] - y[0]

sig = 1.5
# Correlation scale in pixel units (needed for gaussian_filter function)
correlation_scale = sig/dx

print('sigma_phys = ', sig)
print('sigma_pix = ', correlation_scale)
print('dx = ', dx)


r0 = np.sqrt(x**2 + y**2)
X, Y = np.meshgrid(x, y, indexing='ij')  # 2D mesgrid for coordinates x,y

kx = fftfreq(Nx, d=dx)*2.*np.pi
ky = rfftfreq(Nx, d=dx)*2.*np.pi
# 1D kx,ky grids in Fourier space
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')  # 2D meshgrid in Fourier space

q_2 = Kx**2 + Ky**2

dt2q2 = q_2*dt2

precision = 15
mp.dps = precision
mp.pretty = True
one = mpf(1)


def f_precis(x_loc):
    return acos(x_loc)


order_prec = 8  # Order of the Padé approximation for arccos(phi)
taylor_exp = taylor(f_precis, 0, order_prec*2+1)
p_pa1, q_pa1 = pade(taylor_exp, order_prec, order_prec)
p_pa = [float(x) for x in p_pa1]
p_pa = np.array(p_pa)  # Padé coefficients for the numerator
q_pa = [float(x) for x in q_pa1]
q_pa = np.array(q_pa)  # Padé coefficients for the denominator


@jit(nopython=True, parallel=True)
def compute_F(u_1_loc):
    '''
    Parameters
    ----------
    u_1_loc : 2D array

    Returns
    -------
    F : 2D array
    Pade approximation for arccos(phi)

    '''
    u1_loc_2 = u_1_loc**2
    u1_loc_3 = u1_loc_2*u_1_loc
    u1_loc_4 = u1_loc_2**2
    u1_loc_5 = u1_loc_4*u_1_loc
    u1_loc_6 = u1_loc_4*u1_loc_2
    u1_loc_7 = u1_loc_6*u_1_loc
    u1_loc_8 = u1_loc_6*u1_loc_2
    F_num = (p_pa[0] + p_pa[1]*u_1_loc + p_pa[2]*u1_loc_2 + p_pa[3]*u1_loc_3 + p_pa[4] *
             u1_loc_4 + p_pa[5]*u1_loc_5 + p_pa[6]*u1_loc_6 + p_pa[7]*u1_loc_7 + p_pa[8]*u1_loc_8)
    F_denom = (q_pa[0] + q_pa[1]*u_1_loc + q_pa[2]*u1_loc_2 + q_pa[3]*u1_loc_3 + q_pa[4]
               * u1_loc_4 + q_pa[5]*u1_loc_5 + q_pa[6]*u1_loc_6 + q_pa[7]*u1_loc_7 + q_pa[8]*u1_loc_8)
    # F_num = (p_pa[0] + p_pa[1]*u_1_loc + p_pa[2]*u1_loc_2 + p_pa[3]*u1_loc_3 + p_pa[4]*u1_loc_4 )
    # F_denom = (q_pa[0] + q_pa[1]*u_1_loc + q_pa[2]*u1_loc_2 + q_pa[3]*u1_loc_3 + q_pa[4]*u1_loc_4 )
    F = F_num/F_denom
    return F


# @jit(nopython=True, parallel=True)
# def compute_F(u_1_loc):
#     u1_loc_2 = u_1_loc**2
#     u1_loc_3 = u1_loc_2*u_1_loc
#     u1_loc_4 = u1_loc_2**2
#     F = (np.pi/2 - u_1_loc - 81*np.pi*u1_loc_2/238 + 367*u1_loc_3/714 + 183*np.pi*u1_loc_4/9520)/(1 - 81*u1_loc_2/119 + 183*u1_loc_4/4760)
#     return F
# #Pade approximation for arccos(phi)
# #Acceleration with jit (power calculations) and parallelization

@jit(nopython=True, parallel=True)
def sqrt_NL(u_1_loc):
    u1_loc_2 = u_1_loc**2
    return np.sqrt((1 - u_1_loc**(2 + 2*Nexp))/(1 - u1_loc_2))
# Sqrt part of the nonlinear part
# Acceleration with jit (power calculations) and parallelization


def advance_vectorized_step1(u_1_loc):
    D1 = 1
    dt2_prime = 0.5*dt2  # For the first time step, we take 1/2 of dt2

    u_1_hat = rfft2(u_1_loc, workers=n_c)

    F = compute_F(u_1_loc)
    F_hat = rfft2(F, workers=n_c)

    NL_part = (1/4)*sqrt_NL(u_1_loc)*irfft2(q_2*F_hat, workers=n_c)
    NLpart_hat = rfft2(NL_part, workers=n_c)

    u_hat = (D1*u_1_hat + dt2_prime*q_2*NLpart_hat)/(1 + C2*dt2_prime*q_2)
    u = irfft2(u_hat, workers=n_c)

    return u
    #return (u, NL_part)
# Function used to calculate the the first step of the ode (see notes)
# Takes u(0) and gives u(dt)


def advance_vectorized(u_1_loc, u_2_loc):
    D1 = 2.
    D2 = 1.

    u_1_hat = rfft2(u_1_loc, workers=n_c)
    u_2_hat = rfft2(u_2_loc, workers=n_c)

    F = compute_F(u_1_loc)
    F_hat = rfft2(F, workers=n_c)

    NL_part = (1/4)*sqrt_NL(u_1_loc)*irfft2(q_2*F_hat, workers=n_c)
    NLpart_hat = rfft2(NL_part, workers=n_c)
   # N_loc_hat = rfft2(N_loc, workers=n_c)

    u_hat = (D1*u_1_hat - D2*u_2_hat + dt2q2*NLpart_hat)/(1 + C2*dt2q2)
   # u_hat = (D1*u_1_hat - D2*u_2_hat + dt2q2 *
    #         (2*NLpart_hat - N_loc_hat))/(1 + C2*dt2q2)

    u = irfft2(u_hat, workers=n_c)

    return u
    #return (u, NL_part)
# Function used to solve the ode(see notes)
# Takes u(t - dt) and u(t - 2dt) and returns u(t)
# Pseudo spectral method with an implicit time integration


xx = np.arange(0, Nx//2, 1)  # Pixel grid used to calculate g1


def g_1_avant_moy(phi_loc):
    phi_hat = rfft2(phi_loc)
    fourier = phi_hat*np.conjugate(phi_hat)
    res = irfft2(fourier).real
    return res


t_u_save = 5
t_g1_save = 1
int_u_save = int(1/dt)*t_u_save
int_g1_save = int(1/dt)*t_g1_save
saving = int(1/dt)*t_u_save
int_phi2_save = int(0.05/dt)
# Choice of when to save certain variables

u = np.zeros((Nx, Ny), dtype=np.float64)  # solution array
u_1 = np.zeros((Nx, Ny), dtype=np.float64)  # solution at t-dt
u_2 = np.zeros((Nx, Ny), dtype=np.float64)  # solution at t-2*dt

# Generation of the initial random noise

np.random.seed(n_c)

lamb_pref = (eps/(sig*xmax))*np.sqrt(3/np.pi)  # IC prefactor

noise = np.random.rand(Nx, Ny)*2 - 1
noise = lamb_pref*gaussian_filter(noise, correlation_scale,
                                  mode='wrap', truncate=8.0)*Nx*2*np.pi*sig**2
# IC normalized such that <\phi^2(0)> = eps^2

print('var= ', np.var(noise))


path = '/users/jussieu/egliott/Documents/Coarsening/codes/Data_coarsening/'
# path = 'C:\\Users\\elyse\\Documents\\Data_coarsening\\'
# path = ''
seed = 16
name = 'Coarsening'
file_name = f"_dt={dt}_Tmax={tmax}_Nx={Nx}_Xmax={xmax}_Sig={sig}_Imb={imbalance}_Eps={eps}_Nexp={Nexp}_usave={t_u_save}_g1save={t_g1_save}_seed={seed}"
file_copy = file_name + '_copy_v3'
file_copy1 = file_name + '_copy'
i = 0
t_i = 0
j = 0
k = 0

start = time.time()

try:
    with nc.Dataset(path + name + file_copy + '.nc', 'r') as ds_last:
        phi2D = ds_last['phi_2D'][:]
        phi2D_1 = ds_last['phi_2D_1'][:]
        g1_tab = ds_last['g1'][:]
        varphi_tab = ds_last['varphi'][:]
        t_arr_tab = ds_last['t_arr'][:]
        last_index = np.where(~phi2D.mask)[0][-1]
        print(last_index)
        last_phi_2D = phi2D[last_index]
        last_phi_2D_1 = phi2D_1[last_index]
        i_last = last_index*int_u_save
        j_last = last_index
        k_last = last_index*int_u_save
        t_last = i_last*dt
        print('t_last =', t_last)

except Exception as e:
    print('Error', e)

u_2 = last_phi_2D_1.data
u_1 = last_phi_2D.data

#(u_temp,  NL_2) = advance_vectorized_step1(u_2)

# print(t_arr_tab)

Cons_N0 = np.sum(phi2D[0])/(Nx)**2  # Initial conservation number
print('Cons = ', Cons_N0)
Cons_N = Cons_N0
# Need to modify this

try:
    if os.path.exists(path + name + file_name + '_v4' + '.nc'):
        os.remove(path + name + file_name + '_v4' + '.nc')
        os.remove(path + name + file_copy1 + '_v4' + '.nc')

    with nc.Dataset(path + name + file_name + '_v4' + '.nc', mode='w', format='NETCDF4_CLASSIC') as ds:

        time_dim = ds.createDimension('t_arr', int(Nt/int_phi2_save)+1)
        phi2_dim = ds.createDimension('varphi', int(Nt/int_phi2_save)+1)
        g1_time_dim = ds.createDimension('g1_time', int(Nt/int_g1_save)+1)
        g1_value_dim = ds.createDimension('g1_value', Nx)
        phi_time_dim = ds.createDimension('phi_time', int(Nt/int_u_save) + 1)
        phi_x_value_dim = ds.createDimension('phi_x_value', Nx)
        phi_y_value_dim = ds.createDimension('phi_y_value', Ny)
        # Set file dimensions
        t_arr = ds.createVariable('t_arr', np.float32, ('t_arr',))
        varphi = ds.createVariable('varphi', np.float32, ('varphi',))
        g1 = ds.createVariable('g1', np.float32, ('g1_time', 'g1_value'))
        phi_2D = ds.createVariable(
            'phi_2D', np.float32, ('phi_time', 'phi_x_value', 'phi_y_value'))
        phi_2D_1 = ds.createVariable(
            'phi_2D_1', np.float32, ('phi_time', 'phi_x_value', 'phi_y_value'))

        t_arr[:len(t_arr_tab)] = t_arr_tab
        varphi[:len(varphi_tab)] = varphi_tab
        g1[:len(g1_tab)] = g1_tab
        phi_2D[:len(phi2D)] = phi2D
        phi_2D_1[:len(phi2D_1)] = phi2D_1
        # Set variables
        t_i = t_last
        j = j_last
        k = k_last

        # print(varphi)
        ds.sync()
        try:
            shutil.copy(path + name + file_name + '_v4' + '.nc',
                        path + name + file_copy1 + '_v4.nc')
            print('copied')
        except IOError as e:
            print(f'{e}')

        for i in range(i_last+1, Nt):
            t_i = t_i + dt

            if i % int_u_save == 0:
                j = i//int_u_save
                print('t = ' + str(round(t_i, 3)))
                phi_2D[j] = u_1  # Save 2D phi
                phi_2D_1[j] = u_2
                j += 1

            if i % int_g1_save == 0:
                k = i//int_g1_save
                # g1 before radial average
                g1_t_avant = ifftshift(g_1_avant_moy(u_1))
                g1_t_prof = RadialProfile(g1_t_avant, (Nx//2, Nx//2), xx)
                #g1_t_prof = RadialProfile(g1_t_avant,xycen = (Nx//2,Nx//2),min_radius = 0, max_radius = Nx//2, radius_step = 1)
                g1_t_apres = g1_t_prof.profile  # g1 after radial average for pixel values of xx
                # g1 function for real values
                g1_func = interpolate.interp1d(
                    g1_t_prof.radius*dx, g1_t_apres, fill_value='extrapolate')
                g1[k] = g1_func(r0)  # Save g1
                #k += 1

            if i % int_phi2_save == 0:
                Cons_N = (Cons_N0 - np.sum(u_1)/(Nx)**2) / \
                    Cons_N0  # Check conservation of N
                i_save = i//int_phi2_save
                t_arr[i_save] = t_i  # Save time
                varphi[i_save] = np.var(u_1)  # Save <\phi^2>

            if i % saving == 0:
                ds.sync()  # Sync values to nc file
                print(i)
                try:
                    shutil.copy(path + name + file_name + '_v4' +
                                '.nc', path + name + file_copy1 + '_v4.nc')
                    print('copied')
                except IOError as e:
                    print(f'{e}')

            if np.isnan(Cons_N):
                # Usually this means that the code is unstable, try with smaller timestep
                print("nan found")
                ds.sync()
                break

            #(u, NL_1) = advance_vectorized(u_1, u_2, NL_2)
            u = advance_vectorized(u_1, u_2)
            u_2, u_1, u = u_1, u, u_2
            #NL_2 = NL_1
            # Advance in ODE
except Exception as e:
    print('Error', e)

end = time.time()
print('Time = ' + str(end - start))
