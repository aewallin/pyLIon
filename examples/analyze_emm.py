#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:12:00 2023

@author: anders
"""
import pylion as pl
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from scipy import stats


trap = {'z0': 0.86e-3/2,  'frequency': 14.4e6,
        'voltageRF': 300, 'etaRF': 0.97, 'eps':5e-2,
        'voltageDC': -0.0, 'etaDC': 0.97, }
amu = 1.66053906660e-27
mIon = 88*amu
charge=1.60217663e-19
h = 6.62607015e-34
dalpha0 = -4.7938e-40 # J m^2 V^-2
srs = 444779044095486.3 # Hz

steps , data = pl.readdump('positions.txt')
steps2 , vels = pl.readdump('secv.txt')
steps3 , fs = pl.readdump('forces.txt')

data *= 1e9 # to nanometers
# F = qE
fox, foy, foz = fs[:, idx, 0], fs[:, idx, 1], fs[:, idx, 2]
ex,ey,ez = fox/charge, foy/charge, foz/charge # E in V/m

#%%
plt.figure()
import scipy.signal
import scipy.optimize
[fx, px] = scipy.signal.welch(x , fs=1.0/(10*timestep), nperseg=100000)
[fy, py] = scipy.signal.welch(y , fs=1.0/(10*timestep), nperseg=100000)
[fz, pz] = scipy.signal.welch(z , fs=1.0/(10*timestep), nperseg=100000)
[fEx, pEx] = scipy.signal.welch(ex , fs=1.0/(10*timestep), nperseg=100000)
[fEy, pEy] = scipy.signal.welch(ey , fs=1.0/(10*timestep), nperseg=100000)
[fEz, pEz] = scipy.signal.welch(ez , fs=1.0/(10*timestep), nperseg=100000)
plt.semilogy(fx/1e6, px, label='X, T_x = %.3f mK'%(1e3*Tx))
plt.semilogy(fEx/1e6, pEx, label='Ex')
plt.semilogy(fy/1e6, py, label='Y, T_y = %.3f mK'%(1e3*Ty))
plt.semilogy(fEy/1e6, pEy, label='Ey')

plt.semilogy(fz/1e6, pz, label='Z, T_z = %.3f mK'%(1e3*Tz))
plt.semilogy(fEz/1e6, pEz, label='Ez')

def lorentzian( f, f0, a, gam ):
    return a * gam**2 / ( gam**2 + ( f - f0 )**2)


def q_trap(etaDC, vDC, etaRF, vRF, fRF, eps, charge, m, z0):
    """
    Mathieu stability parameters ai and qi
    q proportional to applied RF-voltage
    a proportional to applied DC-voltage
    
    sum(ai) = 0 required by Laplace equation
    
    secular frequency
        w = beta * Omega/2 
        beta ~= sqrt( a + q^2/2 )
    
    stability:
        0 < beta < 1.0
    """
    
    qz = 2.0*etaRF*vRF*charge / (m *pow(2*np.pi*fRF*z0, 2))
    az = -1.0*etaDC*vDC*charge*4.0 / (m *pow(2*np.pi*fRF*z0, 2))

    qx = -1.0*(1.0 - eps)*qz/2.0
    ax = (1.0-eps)*etaDC*vDC*charge*2.0 / (m *pow(2*np.pi*fRF*z0, 2))

    qy = -1.0*(1.0 + eps)*qz/2.0
    ay = (1.0+eps)*etaDC*vDC*charge*2.0 / (m *pow(2*np.pi*fRF*z0, 2))

    return (ax, ay, az), (qx, qy, qz)

def secular2(a,q,fRF):
    """
    omega = beta * Omega / 2
    beta ~= sqrt( a + q**2 / 2)
    """
    beta_sq = a
    beta_sq += (1.0/2.0+a/2.0)*pow(q,2)
    beta_sq += (25.0/128.0+273.0*a/512.0)*pow(q,4)
    beta_sq += (317.0/2304.0+59525.0*a/82944.0)*pow(q,6)

    return np.sqrt(beta_sq)*fRF/2

def sfreqs(trap):
    a, q = q_trap( trap['etaDC'], trap['voltageDC'], 
                  trap['etaRF'], trap['voltageRF'], trap['frequency'], 
                  trap['eps'], charge, mIon, trap['z0'])
    return secular2(a[0],q[0], trap['frequency']), secular2(a[1],q[1], trap['frequency']), secular2(a[2],q[2], trap['frequency'])

a,q=q_trap( trap['etaDC'], trap['voltageDC'], 
                  trap['etaRF'], trap['voltageRF'], trap['frequency'], 
                  trap['eps'], charge, mIon, trap['z0'])

sx, sy, sz = sfreqs(trap) 
print(sfreqs(trap))
# fit to frequencies below fRF/2
fitidx=min(np.argwhere(fz>trap['frequency']/2))[0]
dfz = np.mean(np.diff(fEz))

e2s = []
# EMM
for frf in [trap['frequency'], 2*trap['frequency'], 3*trap['frequency']]:
    rf1_start=min(np.argwhere(fz>(frf-15e3)))[0]
    rf1_stop=min(np.argwhere(fz>(frf+15e3)))[0]
    rf1_e = np.trapz(pEz[rf1_start:rf1_stop], x=fEz[rf1_start:rf1_stop])
    plt.plot(frf/1e6,rf1_e/dfz,'o')
    e2s.append( (frf, rf1_e/dfz) )

# Secular
for frf in [sz, trap['frequency']-sz, trap['frequency']+sz, 2*trap['frequency']-sz,2*trap['frequency']+sz, 3*trap['frequency']-sz,3*trap['frequency']+sz ]:
    rf1_start=min(np.argwhere(fz>(frf-150e3)))[0]
    rf1_stop=min(np.argwhere(fz>(frf+150e3)))[0]
    rf1_e = np.trapz(pEz[rf1_start:rf1_stop], x=fEz[rf1_start:rf1_stop])
    plt.plot(frf/1e6,rf1_e/dfz,'o')
    e2s.append( (frf, rf1_e/dfz) )

def stark(E2, fractional=False):
    # Dube2014 PRL (3)
    # Dube2013 PRA (7)
    t1 = -1/2.0
    t2 = dalpha0/(h*1.0)
    return t1*t2*E2

x_f0, x_a, x_gamma = scipy.optimize.curve_fit( lorentzian, xdata = fx[0:fitidx], ydata = px[0:fitidx], 
                                              p0=(fz[np.argmax(px)],1e-4,10e3),
                                              bounds=((0.5e6, 1e-6, 1e3), (2e6, 7e-1, 2e5))  , method='dogbox')[0]
y_f0, y_a, y_gamma = scipy.optimize.curve_fit( lorentzian, xdata = fy[0:fitidx], ydata = py[0:fitidx], 
                                              p0=(fz[np.argmax(py)],1e-4,10e3),
                                              bounds=((0.5e6, 1e-6, 1e3), (2e6, 7e-1, 2e5))  , method='dogbox')[0]
z_f0, z_a, z_gamma = scipy.optimize.curve_fit( lorentzian, xdata = fz[0:fitidx], ydata = pz[0:fitidx], 
                                              p0=(fz[np.argmax(pz[1:fitidx])],1e-4,10e3),
                                              bounds=((2e6, 1e-6, 1e3), (4e6, 7e-1, 2e4)) , method='dogbox')[0]

print('Lorentzian fit to x PSD:', x_f0, x_a, x_gamma)
print('Lorentzian fit to y PSD:', y_f0, y_a, y_gamma)
print('Lorentzian fit to z PSD:', z_f0, z_a, z_gamma)
fplot=np.linspace(100,10e6,5000)
plt.semilogy(fplot/1e6, lorentzian(fplot, x_f0, x_a, x_gamma), label='f_x = %.1f Hz, (fit/predicted -1) = %.3g'%(x_f0,(x_f0-sx)/sx))
plt.semilogy(fplot/1e6, lorentzian(fplot, y_f0, y_a, y_gamma), label='f_y = %.1f Hz, (fit/predicted -1) = %.3g'%(y_f0,(y_f0-sy)/sy))
plt.semilogy(fplot/1e6, lorentzian(fplot, z_f0, z_a, z_gamma), label='f_z = %.1f Hz, (fit/predicted -1) = %.3g'%(z_f0,(z_f0-sz)/sz))


plt.semilogy([sx/1e6, sx/1e6], [1e-7, 2e-1],'--', label='predicted f_x = %.1f Hz'%sx)
plt.semilogy([sy/1e6, sy/1e6], [1e-7, 2e-1],'--', label='predicted f_y = %.1f Hz'%sy)
plt.semilogy([sz/1e6, sz/1e6], [1e-7, 2e-1],'--', label='predicted f_z = %.1f Hz'%sz)


plt.grid()
plt.legend()
plt.xlim((0,50))
plt.xlabel('Frequency / MHz')
plt.ylabel('Ion position PSD / nm^2/Hz')
plt.title('Trap: f_RF = %.3f MHz, V_RF = %.1f V, eta_RF = %.3f, eps=%.3f'%(trap['frequency']/1e6,trap['voltageRF'],trap['etaRF'],trap['eps'] ))

#%%
plt.figure()
for e in e2s:
    df=stark(pow(e[1],2)/2)
    plt.semilogy(e[0]/1e6,abs(df),'o')
    print(e,df)
plt.grid()
