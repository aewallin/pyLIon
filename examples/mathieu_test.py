#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:28:44 2022

@author: anders
"""

import numpy as np
import scipy as sp
import scipy.special as spfun
from matplotlib import pyplot as plt

#plt.rcParams['text.usetex'] = True


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

def secular(a,q,fRF):
    """
    omega = beta * Omega / 2
    beta ~= sqrt( a + q**2 / 2)
    """
    beta = np.sqrt( a+ pow(q,2)/2.0)
    return     

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


trap = {'z0': 0.86e-3/2,  'frequency': 14.4e6,
        'voltageRF': 300, 'etaRF': 0.97, 'eps':5e-2,
        'voltageDC': -0.0, 'etaDC': 0.97, }
amu = 1.66053906660e-27
#mIon = 87.62*amu
mIon = 88*amu

a, q = q_trap( trap['etaDC'], trap['voltageDC'], 
              trap['etaRF'], trap['voltageRF'], trap['frequency'], 
              trap['eps'], 1.6e-19, mIon, trap['z0'])
print(a,q)
# simulation gives 1.1, 1.2, 2.15 MHz
print('secular x; ',secular(a[0],q[0], trap['frequency']))
print('secular y; ',secular(a[1],q[1], trap['frequency']))
print('secular z; ',secular(a[2],q[2], trap['frequency']))
print('secular2 x; ',secular2(a[0],q[0], trap['frequency']))
print('secular2 y; ',secular2(a[1],q[1], trap['frequency']))
print('secular2 z; ',secular2(a[2],q[2], trap['frequency']))
print('sim z: ', 2235399.570749957)

wx, wy, wz = 2*np.pi*secular2(a[0],q[0], trap['frequency']), 2*np.pi*secular2(a[1],q[1], trap['frequency']), 2*np.pi*secular2(a[2],q[2], trap['frequency'])
kx, ky, kz = wx**2 * mIon, wy**2 * mIon, wz**2 * mIon
print('kx ', kx)
print('ky ', ky)
print('kz ', kz)

uplim =12#E_rec
Npts =1000 
Nstates =1
qplot = np.linspace(0, uplim/4.0, Npts)
EA = np.zeros([Npts,Nstates])
EB = np.zeros([Npts,Nstates])
#U = 4*q 
plt.figure()
print(np.shape(EA))   #plt.fill_between(U, EA[:,i], EB[:,i]) #plt.plot(U,Ea,U,Eb) 
for i in range(Nstates):
    az = spfun.mathieu_a(i,qplot)   # Characteristic value of even Mathieu functions
    bz = spfun.mathieu_b(i+1,qplot)  # Characteristic value of odd Mathieu functions
    plt.fill_between(qplot, az, bz,color='b',alpha=0.2,label='Z stable') # (x, y1, y2)
    ar = -2*spfun.mathieu_a(i,-qplot/2)   # Characteristic value of even Mathieu functions
    br = -2*spfun.mathieu_a(i+1,-qplot/2)  # Characteristic value of odd Mathieu functions
    plt.fill_between(qplot, ar, br,color='r',alpha=0.2,label='X, Y stable') 
plt.xlim((0,1.4))
plt.ylim((-0.8,0.2))
plt.xlabel(r'$q_z$ / $-q_{x,y}/2$')
plt.ylabel(r'$a_z$ / $-2a_{x,y}$')
plt.plot(q[2], a[2],'o',label='sim')
plt.legend()
print(np.shape(EA))    #plt.fill_between(U, EA[:,i], EB[:,i]) #plt.plot(U,Ea,U,Eb) 
plt.show()