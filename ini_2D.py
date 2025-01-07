#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:25:46 2024

@author: egliott
"""

import numpy as np
import scipy
from scipy.fft import fft2, ifft2,ifftshift,rfft2,irfft2
import matplotlib.pyplot as plt
import time
import scipy.signal
from scipy import signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from photutils.profiles import RadialProfile

# fig2, axs2 = plt.subplots(1,1)
# fig3, axs3 = plt.subplots(1,1)
fig4, axs4 = plt.subplots(1,1)

np.random.seed(19)


xmax = 75
ymax = 75
Nx = 2**10
Ny = 2**10
eps = 0.01

x = np.linspace(0,xmax,Nx)
y = np.linspace(0,ymax,Ny)
r0 = np.sqrt(x**2 + y**2)
X, Y = np.meshgrid(x, y, indexing='ij')

dx = x[1] - x[0]
dy = y[1] - y[0]


kx = ky = np.fft.fftfreq(Nx, d = dx)*2.*np.pi
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
dk = kx[1] - kx[0]

q_2 = Kx**2 + Ky**2
    

correlation_scale = 2 #sigma in pixel units
sig = correlation_scale*dx #sigma physique

noise1 = np.random.rand(Nx, Ny)*2 -1 #uniform random noise
lamb_pref = (eps/(correlation_scale*dx*xmax))*np.sqrt(3/np.pi) #Prefactor such that <\phi^2(0)> = \epsilon^2


noise = lamb_pref*gaussian_filter(noise1,correlation_scale, mode = 'wrap', truncate = 8.0)*(2*np.pi*((correlation_scale*dx)**2))*Nx
#Noise generation v1 (like in Clara's code with modif)

kernel_tf = np.exp(-q_2*((correlation_scale*dx)**2)/2)*(2*np.pi*((correlation_scale*dx)**2))
noise_tf = lamb_pref*kernel_tf*fft2(noise1)*Nx
noise_after = ifft2(noise_tf)
#Noise generation v2 (sans boîte noire)

print('var = ' + str(np.var(noise)))

def g_1_avant_moy(phi_loc):
    phi_hat = rfft2(phi_loc)
    fourier = phi_hat*np.conjugate(phi_hat)
    res = irfft2(fourier)
    return res

axs2.axhline(y=eps, color = 'g')
axs2.axhline(y=-eps, color = 'g')

axs2.plot(x,noise[:,int(Ny/2)],color = 'b')
# axs2.plot(x,noise_after[:,int(Ny/2)], color ='k', ls = '--')

axs2.set_xlabel('x', fontsize = 'large')
axs2.set_ylabel(r'$\phi(x,y=0)$', fontsize = 'large')
#Coupes phi(x,y=0) + comparison of two versions

sigma_test = np.sqrt(eps**2)
print('var_test = ', sigma_test**2)

z = np.linspace(-4*sigma_test, 4*sigma_test, 1000)
normal_distr = (1.5*np.sqrt(Nx))*np.exp(-0.5*z**2/sigma_test**2)

axs3.hist(noise_after[:,int(Ny/2)], bins = 40)
axs3.plot(z,normal_distr, label = r'$e^{-x^2/4\sigma^2}$' + ' with' + r' $\sigma = \epsilon$')
axs3.set_xlabel(r'$\phi(x,y=0)$')
axs3.set_xlim(-4*sigma_test,4*sigma_test)
axs3.set_title('Distribution of IC')
axs3.legend()

xx = np.arange(0,Nx//2,1) #Pixel grid used to calculate g1

g1_t_avant = ifftshift(g_1_avant_moy(noise)) #g1 before radial average
g1_t_prof = RadialProfile(g1_t_avant,(Nx//2,Nx//2),xx) 
g1_t_apres = g1_t_prof.profile #g1 after radial average for pixel values of xx
g1_func = interpolate.interp1d(g1_t_prof.radius*dx,g1_t_apres, fill_value = 'extrapolate') 

axs4.plot(r0,g1_func(r0)/(Nx**2), color = 'b', label = 'Numerical IC')
axs4.plot(r0,(eps**2)*np.exp(-(r0**2)/(4*sig**2)), color = 'k', ls = '--', label = 'Analytical IC : \n' + r'$g_1(r) = \epsilon^2 e^{-r^2/4\sigma^2} $')

axs4.set_xlim(0,3)
axs4.set_xlabel('r', fontsize = 'large')
axs4.set_ylabel(r'$g_1(r)$', fontsize = 'large')

axs4.set_title('Benchmark of IC with ' + rf'$\epsilon = 0.01$')
axs4.legend()