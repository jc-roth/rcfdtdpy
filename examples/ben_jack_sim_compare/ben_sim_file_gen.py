#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:07:50 2017

@author: benofori
"""
from tqdm import tqdm
from scipy.fftpack import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

#%% For this code, c has been set to 1
#%% First, define constants and sample parameters
c0 = 1;                                             # (um/300 ps) speed of light
R = 1/1;
eps_air = 1;                                        # dielectric costant of air
eps_mat = 16;                                       # high frequency dielectric constant of material

L = 0.09;                                           # (um) thickness of sample
nu_j = 1;                                           # (THz) resonant frequency of oscillator
gamma = 0.01;                                       # (THz) linewidth
gamma_j = gamma * 2 * np.pi;

beta_j = np.sqrt((gamma_j)**2 - (2*np.pi*nu_j)**2, dtype=np.complex64);      # (THz) effective oscillation frequency
A = 1;                                            # (a.u.) oscillator amplitude


#%% Next, define the simulation parameters
dz = 0.003;                                             # (um) size discretization
dt = R*dz/c0;                                           # (ps) timestep: (0.003 um) / (300 um/ps) = 0.00001 ps = 0.01 fs

t = np.arange(-90,210,dt) * (10/3);                   # (ps) simulation time window 
z_pos = np.arange(c0/R*-5,c0/R*5,dz);                   # (um) spatial box size

print(len(t), len(z_pos))

# N = np.size(t);
N = 2**16;

z1 = np.min(np.abs(z_pos));                             # finding zero for setting the position of the sample
z_int_1 = np.argwhere(z_pos==z1)[0][0]


z_int_2 = np.int(np.round(z_int_1+L/dz));               # Finding the far edge of the sample

z_end = np.size(z_pos);                                 # Finding the size of the overall simulation box
z_source = np.argmin(np.abs(z_pos-(-3)));    # Setting the location of the source to be at -3 um

#%% Next, define the source of the THz 
FWHM_THz = 0.6;                                     # (ps) FWHM of the envelope of the THz pulse
tau = FWHM_THz/(2*np.sqrt(2*np.log(2)));            # (ps) time constant for THz envelope


J_t = np.diff(np.diff(np.append(np.exp(-(t)**2/(6e3)), np.array([0, 0]))));

# UNCOMMENT FOLLOWING LINE, SEE WHAT HAPPENS
#J_t = 100*J_t/(np.max(np.abs(J_t)));


# J_t = 0*exp(-(t+1).^2/(2*0.004));

chi_t = np.zeros((np.size(t),));

Sig_trc = np.zeros((np.size(t),));
Ref_trc = np.zeros((np.size(t),));
Sig_back_trc = np.zeros((np.size(t),));
Ref_back_trc = np.zeros((np.size(t),));

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
    
    Sig_trc[t_i] = E_z[z_int_2+50];
    Ref_trc[t_i] = E_z_ref[z_int_2+50];

    # Jack's additions in the two lines below
    Sig_back_trc[t_i] = E_z[z_int_1-50];
    Ref_back_trc[t_i] = E_z_ref[z_int_1-50];
    
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


np.savez('bdata.npz', t=t, chi_t=chi_t, Sig_back_trc=Sig_back_trc, Ref_back_trc=Ref_back_trc, Sig_trc=Sig_trc, Ref_trc=Ref_trc)

# CHI STUFF
fig2 = plt.figure(2)
f2_ax1 = fig2.add_subplot(1,1,1); 
f2_ax1.plot(t,chi_t)
plt.show()

# Time plot
plt.plot(t, Sig_back_trc)
plt.show()

# FFT STUFF
incf = fft(Ref_trc)
freq = fftfreq(len(t), dt)
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.plot(freq, np.absolute(incf))
ax1.plot(freq, np.angle(incf))
plt.show()

trans = fft(Sig_trc)
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.plot(freq, np.absolute(trans))
ax1.plot(freq, np.angle(trans))
plt.show()


