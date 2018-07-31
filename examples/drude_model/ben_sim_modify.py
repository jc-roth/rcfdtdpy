#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:07:50 2017

@author: benofori
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% For this code, c has been set to 1
#%% First, define constants and sample parameters
c0 = 1;                                             # (um/300 ps) speed of light
R = 1/1;
eps_air = 1;                                        # dielectric costant of air
eps_mat = 4;                                       # high frequency dielectric constant of material

L = 5;                                           # (um) thickness of sample
nu_j = 0;                                           # (THz) resonant frequency of oscillator
gamma = 1;                                       # (THz) linewidth
gamma_j = gamma * 2 * np.pi;

beta_j = np.sqrt((gamma_j)**2 - (2*np.pi*nu_j)**2);      # (THz) effective oscillation frequency
A = 10;                                            # (a.u.) oscillator amplitude


#%% Next, define the simulation parameters
dz = 0.03;                                             # (um) size discretization
dt = R*dz/c0;                                           # (ps) timestep

t = np.arange(-1000,3000,0.1);                                # (ps) simulation time window 
z_pos = np.arange(c0/R*-5,10*c0/R,dz);                   # (um) spatial box size

# N = np.size(t);
N = 2**16;

z1 = np.min(np.abs(z_pos));                             # finding zero for setting the position of the sample
z_int_1 = np.argmin(np.abs(np.subtract(0, z_pos)))

z_int_2 = np.argmin(np.abs(np.subtract(5, z_pos)))              # Finding the far edge of the sample

z_end = np.size(z_pos);                                 # Finding the size of the overall simulation box
z_source = np.argmin(np.abs(np.subtract(-2.5, z_pos)))   # Setting the location of the source to be 0.5 um from the sample

#%% Next, define the source of the THz 
FWHM_THz = 0.6;                                     # (ps) FWHM of the envelope of the THz pulse
tau = FWHM_THz/(2*np.sqrt(2*np.log(2)));            # (ps) time constant for THz envelope


J_t = np.diff(np.diff(np.append(np.exp(-(t)**2/(2e4)), np.array([0, 0]))));
J_t = 100*J_t/(np.max(np.abs(J_t)));
# J_t = 0*exp(-(t+1).^2/(2*0.004));

chi_t = np.zeros((np.size(t),));

Sig_trc = np.zeros((np.size(t),));
Ref_trc = np.zeros((np.size(t),));

#%% Setting up the absorbing boundary conditions

Ez_first = np.zeros((np.size(t),));
Ez_first_ref = np.zeros((np.size(t),));

Ez_last = np.zeros((np.size(t),));
Ez_last_ref = np.zeros((np.size(t),));

#%% Defining initial conditions

E_z = np.zeros((np.size(z_pos),));
H_z = np.zeros((np.size(z_pos),));

E_z_ref = np.zeros((np.size(z_pos),));
H_z_ref = np.zeros((np.size(z_pos),));

A_1j = A*1/(2*beta_j);                           # Oscillator amplitude
A_2j = A*-1/(2*beta_j);                          # Oscillator amplitude

if np.abs(beta_j-gamma_j) < 1E-5:
    chi_0_1 = 0;
    chi_0_2 = A_2j/(beta_j+gamma_j)* (np.exp(dt*(-gamma_j-beta_j))-1);
else:
    chi_0_1 = A_1j/(beta_j-gamma_j)* (np.exp(dt*(-gamma_j+beta_j))-1);
    chi_0_2 = -A_2j/(beta_j+gamma_j)* (np.exp(dt*(-gamma_j-beta_j))-1);


d_chi_0_1 = chi_0_1*(1-np.exp(dt*(-gamma_j+beta_j)));
d_chi_0_2 = chi_0_2*(1-np.exp(dt*(-gamma_j-beta_j)));

psi_n = 0;
chi_0 = np.real(chi_0_1+chi_0_2);
chi_n = np.real(chi_0_1*np.exp(dt*(-gamma_j + beta_j)) + chi_0_2*np.exp(dt*(-gamma_j - beta_j)));

psi_n_1 = 0;
psi_n_2 = 0;


plot_on = 0;
for t_val in tqdm(np.arange(1/R,np.size(t)-1)):
    t_i = np.int(t_val)
    
    t_ii = t[t_i];
    
    J_t_i = J_t[t_i];
    
    if np.abs(beta_j-gamma_j) < 1E-5:
        chi_n_1 = 0;
        chi_n_2 = A_2j/(beta_j+gamma_j)*np.exp((t_i-1)*dt*(-gamma_j-beta_j))*(np.exp(dt*(-gamma_j-beta_j))-1);
    else:
        chi_n_1 = A_1j/(beta_j-gamma_j)*np.exp((t_i-1)*dt*(-gamma_j+beta_j)) * (np.exp(dt*(-gamma_j+beta_j))-1);
        chi_n_2 = -A_2j/(beta_j+gamma_j)*np.exp((t_i-1)*dt*(-gamma_j-beta_j)) * (np.exp(dt*(-gamma_j-beta_j))-1);
    

    chi_n = np.real(chi_n_1* np.exp(dt*(-gamma_j+beta_j)) + chi_n_2 * np.exp(dt*(-gamma_j-beta_j)));
    
    

    psi_n_1 = E_z*d_chi_0_1 + np.exp(dt*(-gamma_j+beta_j))*psi_n_1;
    psi_n_2 = E_z*d_chi_0_2 + np.exp(dt*(-gamma_j-beta_j))*psi_n_2;
    
    psi_n = np.real(psi_n_1 + psi_n_2);
    
    
    chi_t[t_i] = chi_n;
    

    H_z[0:-2] = H_z[0:-2] - dt/dz*(E_z[1:-1]-E_z[0:-2]);
    H_z_ref[0:-2] = H_z_ref[0:-2] - dt/dz*(E_z_ref[1:-1]-E_z_ref[0:-2]);
        
    for z_val1 in np.arange(1,z_int_1+1):
        z = np.int(z_val1)
        
        if z == np.round(z_source):
            J_t_i = J_t[t_i];
        else:
            J_t_i = 0;
             
        E_z[z] = E_z[z] + 1/(eps_air+chi_0)*0 - dt/dz*(1/eps_air) * (H_z[z] - H_z[z-1]) - J_t_i*dt;
        E_z_ref[z] = E_z_ref[z] + 1/(eps_air+chi_0)*0 - dt/dz*(1/eps_air) * (H_z_ref[z] - H_z_ref[z-1]) - J_t_i*dt;
    
    for z_val2 in np.arange(z_int_1+1,z_int_2+1):
        z = np.int(z_val2)
        
        E_z[z] = (eps_mat)/(eps_mat+chi_0)*E_z[z] + 1/(eps_mat+chi_0)*psi_n[z] - dt/dz*(1/(eps_mat+chi_0)) * (H_z[z] - H_z[z-1]) - J_t_i*dt;
        E_z_ref[z] = E_z_ref[z] + 1/(eps_air+chi_0)*0 - dt/dz*(1/eps_air) * (H_z_ref[z] - H_z_ref[z-1]) - 0;
    
    for z_val3 in np.arange(z_int_2+1,z_end):
        z = np.int(z_val3)
        
        E_z[z] = E_z[z] + 1/(eps_air+chi_0)*0 - dt/dz*(1/eps_air) * (H_z[z] - H_z[z-1]) - J_t_i*dt;
        E_z_ref[z] = E_z_ref[z] + 1/(eps_air+chi_0)*0 - dt/dz*(1/eps_air) * (H_z_ref[z] - H_z_ref[z-1]) - J_t_i*dt;
    
    
    Ez_first[t_i] = E_z[1];
    E_z[0] = Ez_first[t_i-1];
    
    Ez_first_ref[t_i] = E_z_ref[1];
    E_z_ref[0] = Ez_first_ref[t_i-1];
    
    Ez_last[t_i] = E_z[-3];
    E_z[-2] = Ez_last[t_i-1];
    
    Ez_last_ref[t_i] = E_z_ref[-3];
    E_z_ref[-2] = Ez_last_ref[t_i-1];
    
    Sig_trc[t_i] = E_z[-3];
    Ref_trc[t_i] = E_z_ref[-3];
    
    if plot_on:
        if np.mod(t_i, 10) == 0: 
            fig99 = plt.figure(99); 
            plt.clf()
            plt.axis([np.min(z_pos), np.max(z_pos), -0.2, 0.2])
#        %         caxis([0 25])
#        %         colorbar
        
#        hold on
            plt.plot(z_pos,E_z)
            plt.plot(z_pos,E_z_ref)
        #                     plot(z_pos-dz/2,H_z)
#        ylim([-0.3 0.3])
#        plt.plot(z_pos[z_int_1]*[1, 1],[-1, 1],'k--')
#        plt.plot(z_pos[z_int_2]*[1, 1],[-1, 1],'k--')
        #             pause
            fig99.suptitle(str(t_ii) + ' ps')
            plt.pause(0.02)
            plt.show()

#%%

chi_f = np.fft.fft(chi_t,N)
fig101 = plt.figure(101)
plt.clf()
f101_ax1 = fig101.add_subplot(4,2,(1,3))
f101_ax1.plot(t,Sig_trc)
f101_ax1.plot(t,Ref_trc)
#
freq = np.fft.fftfreq(N,dt)
omega = 2*np.pi*freq
#
sig_spec = np.fft.fft(Sig_trc-np.mean(Sig_trc),N)
ref_spec = np.fft.fft(Ref_trc-np.mean(Ref_trc),N)
#
t_spec = np.divide(sig_spec,ref_spec)
phi = -np.unwrap(np.angle(t_spec))
Abs_spec = np.abs(t_spec)
#
n = c0/(omega*L)*phi + 1
#
k = -c0/(omega*L)*np.log(Abs_spec*(n+1)**2/(4*n))
#
#n = np.fft.fftshift(n)
#k = np.fft.fftshift(k)
#
f1_ax2 = fig101.add_subplot(4,2,2)
f1_ax2.plot(np.fft.fftshift(freq),np.fft.fftshift(n))
f1_ax2.set_ylabel('n')
f1_ax2.set_xlim([0.2, 2.5])
f1_ax2.set_ylim([3, 5])
#
f1_ax4 = fig101.add_subplot(4,2,4)
f1_ax4.plot(np.fft.fftshift(freq),np.fft.fftshift(k))
f1_ax4.set_xlim([0.2, 2.5])
f1_ax4.set_ylabel(r'$\kappa$')
f1_ax4.set_xlabel('frequency (THz)')

f101_ax5 = fig101.add_subplot(2,2,3) 
f101_ax5.plot(t,chi_t)
f101_ax5.set_xlabel('t (ps)')
f101_ax5.set_ylabel(r'$\chi$(t) (a.u.)')

f101_ax6 = fig101.add_subplot(2,2,4) 
f101_ax6.plot(freq,np.real(chi_f),freq,-np.imag(chi_f))
f101_ax6.set_xlabel('frequency (THz)')
f101_ax6.set_ylabel(r'$\chi(\omega)$ (a.u.)')
f101_ax6.set_xlim([0.2, 2.5])

plt.tight_layout()
plt.show()