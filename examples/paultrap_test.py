import pylion as pl
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from scipy import stats

# use filename for simulation name
name = Path(__file__).stem

s = pl.Simulation(name)

ions = {'mass': 88, 'charge': 1}
positions = [[0, 0, 0]]
ions = pl.placeions(ions, positions)
s.append(ions)

trap = {'radius': 7e-3, 'length': 5.5e-3, 'kappa': 0.244, 'frequency': 3.85e6,
        'voltage': 300, 'endcapvoltage': -0.01, 'pseudo': False}
trap = {'z0': 0.86e-3/2,  'frequency': 14.4e6,
        'voltageRF': 300, 'etaRF': 0.97, 'eps':5e-2,
        'voltageDC': +0.0, 'etaDC': 0.97, }

# 279.86,
        
        #ltrap = pl.linearpaultrap(trap, ions, all=False)
#ltrap = modtrap(123, trap)
#ltrap = endcaptrap(123, trap)
ltrap = pl.endcappaultrap(trap)
s.append(ltrap)
#print(ltrap)

uid=s._uids
print(uid)

# temperature, dampingtime
s.append(pl.langevinbath(1e-6, 1e-5))
print('bath append')
s.append(pl.dump('positions.txt', variables=['x', 'y', 'z']))
vavg = pl.timeaverage(1, variables=['vx', 'vy', 'vz']) # numer of time-steps to average over
s.append(pl.dump('secv.txt', vavg))
s.attrs['timestep'] = 1e-9

s.append( pl.efield( 0.0, 0, 0) )
x_ampl=1
xff=122e3
uid=99
lines = [ '\n# Oscillating E-field',
         f'variable oscAMPL{uid}\t\tequal {x_ampl:e}',
         f'variable oscFF{uid}\t\tequal {2*np.pi*xff:e}',
         f'variable oscEX2{uid}\t\tequal "v_oscAMPL{uid} * cos( v_oscFF{uid} *step*dt )"',
         f'fix {uid} all efield v_oscEX2{uid} 0 0']
osc_dict= {'code': lines}
s.append(osc_dict)

s.append(pl.evolve(1e5))
s.execute()

steps , data = pl.readdump('positions.txt')
steps2 , vels = pl.readdump('secv.txt')

data *= 1e9 # to nanometers

print(steps)
print("Timestep ",s.attrs['timestep'] )
t = steps*s.attrs['timestep']
timestep=s.attrs['timestep']
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#for idx in [0]:
idx=0
x, y, z = data[:, idx, 0],  data[:, idx, 1], data[:, idx, 2]
vx, vy, vz = vels[:, idx, 0], vels[:, idx, 1], vels[:, idx, 2]
kB = 1.380649e-23
amu = 1.66053906660e-27
mIon = 88*amu
Tx = 0.5*mIon*np.mean(np.square(vx)) / kB
Ty = 0.5*mIon*np.mean(np.square(vy)) / kB
Tz = 0.5*mIon*np.mean(np.square(vz)) / kB
print('Temperature: ', Tx, Ty, Tz)

ax.plot(x, y, z)
ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
ax.set_zlabel('z (nm)')


#%%
plt.figure()
import scipy.signal
import scipy.optimize
[fx, px] = scipy.signal.welch(x , fs=1.0/(10*timestep), nperseg=100000)
[fy, py] = scipy.signal.welch(y , fs=1.0/(10*timestep), nperseg=100000)
[fz, pz] = scipy.signal.welch(z , fs=1.0/(10*timestep), nperseg=100000)
plt.semilogy(fx/1e6, px)
plt.semilogy(fy/1e6, py)
plt.semilogy(fz/1e6, pz)

def lorentzian( f, f0, a, gam ):
    return a * gam**2 / ( gam**2 + ( f - f0 )**2)

# fit to frequencies below fRF/2
fitidx=min(np.argwhere(fz>trap['frequency']/2))[0]
x_f0, x_a, x_gamma = scipy.optimize.curve_fit( lorentzian, xdata = fx[0:fitidx], ydata = px[0:fitidx], 
                                              p0=(fz[np.argmax(px)],1e-4,10e3),
                                              bounds=((0.5e6, 1e-6, 1e3), (2e6, 1e-1, 2e5))  , method='dogbox')[0]
y_f0, y_a, y_gamma = scipy.optimize.curve_fit( lorentzian, xdata = fy[0:fitidx], ydata = py[0:fitidx], 
                                              p0=(fz[np.argmax(py)],1e-4,10e3),
                                              bounds=((0.5e6, 1e-6, 1e3), (2e6, 1e-1, 2e5))  , method='dogbox')[0]
z_f0, z_a, z_gamma = scipy.optimize.curve_fit( lorentzian, xdata = fz[0:fitidx], ydata = pz[0:fitidx], 
                                              p0=(fz[np.argmax(pz)],1e-4,10e3),
                                              bounds=((2e6, 1e-6, 1e3), (4e6, 1e-1, 2e4)) , method='dogbox')[0]

print('Lorentzian fit to x PSD:', x_f0, x_a, x_gamma)
print('Lorentzian fit to y PSD:', y_f0, y_a, y_gamma)
print('Lorentzian fit to z PSD:', z_f0, z_a, z_gamma)
plt.semilogy(fx/1e6, lorentzian(fx, x_f0, x_a, x_gamma))
plt.semilogy(fy/1e6, lorentzian(fy, y_f0, y_a, y_gamma))
plt.semilogy(fz/1e6, lorentzian(fz, z_f0, z_a, z_gamma))

plt.grid()
plt.xlim((0,35))
plt.xlabel('Frequency / MHz')
plt.ylabel('Ion position PSD / nm^2/Hz')


#%% trajectory plot
plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#   kB T = (1/2) m v**2

plt.subplot(3,1,1)
plt.plot(t, vx, label='V_x, T_x = %.4g K'%Tx)
plt.grid()
plt.legend()
plt.ylabel('v_x')


plt.subplot(3,1,2)
plt.plot(t, vy, label='V_y, T_y = %.4g K'%Ty)
plt.grid()
plt.legend()
plt.ylabel('v_y')


plt.subplot(3,1,3)
plt.plot(t, vz, label='V_z, T_z = %.4g K'%Tz)
plt.ylabel('v_z')
plt.xlabel('Time / s')
plt.grid()
plt.legend()

plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#   kB T = (1/2) m v**2

plt.subplot(3,1,1)
plt.plot(t, x, label='T_x = %.4g K'%Tx)

kx = 6.0759325413619195e-12
#  F = k x = E q
# x = E q / k
oscampl = x_ampl * 1.6e-19 / kx
plt.plot(t, oscampl*1e9*np.cos(2*np.pi*xff*t),'r-',label=' ( EQ/ k_x ) * cos(omega_mod t)')

plt.grid()
plt.legend()
plt.ylabel('x / nm')

plt.subplot(3,1,2)
plt.plot(t, vy, label='T_y = %.4g K'%Ty)
plt.grid()
plt.legend()
plt.ylabel('y / nm')


plt.subplot(3,1,3)
plt.plot(t, vz, label='T_z = %.4g K'%Tz)
plt.ylabel('z / nm')
plt.xlabel('Time / s')
plt.grid()
plt.legend()


#%%
xmin,xmax,ymin,ymax = -15,15,-15,15
heatmap, xedges, yedges = np.histogram2d(x, y, bins=250, range=[[xmin, xmax], [ymin, ymax]])
s=2
from scipy.ndimage.filters import gaussian_filter
heatmap = gaussian_filter(heatmap, sigma=s)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

heatmap2, xedges2, yedges2 = np.histogram2d(x, z, bins=250, range=[[xmin, xmax], [ymin, ymax]])
heatmap2 = gaussian_filter(heatmap2, sigma=s)
extent2 = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]

heatmap3, xedges3, yedges3 = np.histogram2d(y, z, bins=250, range=[[xmin, xmax], [ymin, ymax]])
heatmap3 = gaussian_filter(heatmap3, sigma=s)
extent3 = [xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]]


#plt.clf()
plt.figure()
plt.subplot(1,3,1)
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(1,3,2)
plt.imshow(heatmap2.T, extent=extent2, origin='lower')
plt.xlabel('x')
plt.ylabel('z')
plt.subplot(1,3,3)
plt.imshow(heatmap3.T, extent=extent3, origin='lower')
plt.xlabel('y')
plt.ylabel('z')

plt.show()


plt.show()
