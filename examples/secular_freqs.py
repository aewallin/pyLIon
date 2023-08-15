#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:05:26 2022

@author: anders
"""

Vrf = [300, 305, 310, 315, 320]
fs = [ [1026155.0620276262, 1138906.9471141472, 2228631.381373111 ],
      [1044920.1334952484, 1155569.71648396, 2271296.606731964],
      [1064200.502046731,  1176837.969035943, 2313842.654490645],
      [ 1082738.9787930197 , 1194333.9133578052 , 2351479.036971614 ],
      [1095099.7968006048, 1214143.1917436551, 2391994.5679469574]]


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
    return     beta*fRF/2

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

def sfreqs2(vrf):
    a, q = q_trap( trap['etaDC'], trap['voltageDC'], 
                  trap['etaRF'], vrf, trap['frequency'], 
                  trap['eps'], 1.6e-19, mIon, trap['z0'])
    return secular2(a[0],q[0], trap['frequency']), secular2(a[1],q[1], trap['frequency']), secular2(a[2],q[2], trap['frequency'])

def sfreqs1(vrf):
    a, q = q_trap( trap['etaDC'], trap['voltageDC'], 
                  trap['etaRF'], vrf, trap['frequency'], 
                  trap['eps'], 1.6e-19, mIon, trap['z0'])
    return secular(a[0],q[0], trap['frequency']), secular(a[1],q[1], trap['frequency']), secular(a[2],q[2], trap['frequency'])

#%%
plt.figure()
plt.subplot(2,3,1)
plt.plot(Vrf, [x[0] for x in fs],'o')
v = np.linspace(295,325,10)
plt.plot(v, [sfreqs2(x)[0] for x in v],'-',label='high order')
plt.plot(v, [sfreqs1(x)[0] for x in v],'--',label='low order')
plt.legend()
plt.title('X')
plt.xlim((295,325))

plt.subplot(2,3,2)
plt.plot(Vrf, [x[1] for x in fs],'o')
v = np.linspace(295,325,10)
plt.plot(v, [sfreqs2(x)[1] for x in v],'-',label='high order')
plt.plot(v, [sfreqs1(x)[1] for x in v],'--',label='low order')
plt.legend()
plt.title('Y')
plt.xlim((295,325))

plt.subplot(2,3,3)
plt.plot(Vrf, [x[2] for x in fs],'o')
v = np.linspace(295,325,10)
plt.plot(v, [sfreqs2(x)[2] for x in v],'-',label='high order')
plt.plot(v, [sfreqs1(x)[2] for x in v],'--',label='low order')
plt.legend()
plt.title('Z')
plt.xlim((295,325))

plt.subplot(2,3,4)
plt.plot(Vrf, np.array([x[0] for x in fs]) - np.array([sfreqs2(x)[0] for x in Vrf]),'o')
plt.legend()
plt.title('X')
plt.xlim((295,325))

plt.subplot(2,3,5)
plt.plot(Vrf, np.array([x[1] for x in fs]) - np.array([sfreqs2(x)[1] for x in Vrf]),'o')
plt.legend()
plt.title('Y')
plt.xlim((295,325))

plt.subplot(2,3,6)
plt.plot(Vrf, np.array([x[2] for x in fs]) - np.array([sfreqs2(x)[2] for x in Vrf]),'o')
plt.legend()
plt.title('Z')
plt.xlim((295,325))

