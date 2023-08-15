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
positions = [[0, 0, 600e-9]] # with large static E-field, start simulation at expected equilibrium
ions = pl.placeions(ions, positions)
s.append(ions)


trap = {'z0': 0.86e-3/2,  'frequency': 14.4e6,
        'voltageRF': 300, 'etaRF': 0.97, 'eps':5e-2,
        'voltageDC': 0.0, 'etaDC': 0.97, }

frf = trap['frequency']
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
Tion=1e-3
dampingtime=1e-5
s.append(pl.langevinbath(Tion, dampingtime))
print('bath append')
s.append(pl.dump('positions.txt', variables=['x', 'y', 'z']))
s.append(pl.dump('forces.txt', variables=['fx', 'fy', 'fz']))
vavg = pl.timeaverage(1, variables=['vx', 'vy', 'vz']) # numer of time-steps to average over
s.append(pl.dump('secv.txt', vavg))
timestep=0.25e-9
s.attrs['timestep'] = timestep

# addiotional constant E-field
s.append( pl.efield( 0.0, 0.0, 100.0) )

num_steps=1e6
#s.append(pl.evolve(1e6))
s.append(pl.evolve(num_steps))
s.execute()

###############################################################################
# End simulation
###############################################################################
#%%

steps , data = pl.readdump('positions.txt')
steps2 , vels = pl.readdump('secv.txt')
steps3 , fs = pl.readdump('forces.txt')

data *= 1e9 # to nanometers
# F = qE
fx, fy, fz = fs[:, idx, 0], fs[:, idx, 1], fs[:, idx, 2]
ex,ey,ez = fx/charge, fy/charge, fz/charge # E in V/m

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
charge=1.60217663e-19
Tx = 0.5*mIon*np.mean(np.square(vx)) / kB
Ty = 0.5*mIon*np.mean(np.square(vy)) / kB
Tz = 0.5*mIon*np.mean(np.square(vz)) / kB
print('Temperature: %.3f, %.3f, %.3f mK'%( 1e3*Tx, 1e3*Ty, 1e3*Tz ))

ax.plot(x, y, z)
ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
ax.set_zlabel('z (nm)')

#%%
plt.figure()
plt.subplot(3,1,1)
plt.plot(ex,label='E_x mean %.3f, stdev %.3f, mean(square()) %.3f'%(np.mean(ex), np.std(ex), np.mean(np.power(ex,2))))
plt.grid()
plt.legend()
plt.subplot(3,1,2)
plt.plot(ey,label='E_y mean %.3f, stdev %.3f, mean(square()) %.3f'%(np.mean(ey), np.std(ey), np.mean(np.power(ey,2))))
plt.grid()
plt.legend()
plt.subplot(3,1,3)
plt.plot(ez,label='E_z mean %.3f, stdev %.3f, mean(square()) %.3f'%(np.mean(ez), np.std(ez), np.mean(np.power(ez,2))))
plt.grid()
plt.legend()


#%%
# Attaching 3D axis to the figure
import matplotlib.animation

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim((-200,200))
ax.set_ylim((-200,200))
ax.set_zlim((-200,200))

def update_frame(num, lines, ball, tail):
    last = 10*num+tail
    first = 10*num
    lines.set_data( x[first:last], y[first:last]) 
    lines.set_3d_properties( z[first:last] )
    ball.set_data( x[last], y[last]) 
    ball.set_3d_properties( z[last] )

tail=1000
num_steps=int( (len(t)-tail)/10)
lines = ax.plot([], [], [])[0] 
ball = ax.plot([], [], [],'ro')[0] 
# Creating the Animation object
ani = matplotlib.animation.FuncAnimation(
    fig, update_frame, num_steps, fargs=(lines, ball, tail), interval=10)


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
"""
x_f0, x_a, x_gamma = scipy.optimize.curve_fit( lorentzian, xdata = fx[0:fitidx], ydata = px[0:fitidx], 
                                              p0=(fz[np.argmax(px)],1e-4,10e3),
                                              bounds=((0.5e6, 1e-6, 1e3), (2e6, 7e-1, 2e5))  , method='dogbox')[0]
y_f0, y_a, y_gamma = scipy.optimize.curve_fit( lorentzian, xdata = fy[0:fitidx], ydata = py[0:fitidx], 
                                              p0=(fz[np.argmax(py)],1e-4,10e3),
                                              bounds=((0.5e6, 1e-6, 1e3), (2e6, 7e-1, 2e5))  , method='dogbox')[0]
z_f0, z_a, z_gamma = scipy.optimize.curve_fit( lorentzian, xdata = fz[0:fitidx], ydata = pz[0:fitidx], 
                                              p0=(fz[np.argmax(pz)],1e-4,10e3),
                                              bounds=((2e6, 1e-6, 1e3), (4e6, 7e-1, 2e4)) , method='dogbox')[0]

print('Lorentzian fit to x PSD:', x_f0, x_a, x_gamma)
print('Lorentzian fit to y PSD:', y_f0, y_a, y_gamma)
print('Lorentzian fit to z PSD:', z_f0, z_a, z_gamma)
fplot=np.linspace(100,10e6,5000)
plt.semilogy(fplot/1e6, lorentzian(fplot, x_f0, x_a, x_gamma), label='f_x = %.1f Hz, (fit/predicted -1) = %.3g'%(x_f0,(x_f0-sx)/sx))
plt.semilogy(fplot/1e6, lorentzian(fplot, y_f0, y_a, y_gamma), label='f_y = %.1f Hz, (fit/predicted -1) = %.3g'%(y_f0,(y_f0-sy)/sy))
plt.semilogy(fplot/1e6, lorentzian(fplot, z_f0, z_a, z_gamma), label='f_z = %.1f Hz, (fit/predicted -1) = %.3g'%(z_f0,(z_f0-sz)/sz))
"""

plt.semilogy([sx/1e6, sx/1e6], [1e-7, 2e-1],'--', label='predicted f_x = %.1f Hz'%sx)
plt.semilogy([sy/1e6, sy/1e6], [1e-7, 2e-1],'--', label='predicted f_y = %.1f Hz'%sy)
plt.semilogy([sz/1e6, sz/1e6], [1e-7, 2e-1],'--', label='predicted f_z = %.1f Hz'%sz)


plt.grid()
plt.legend()
plt.xlim((0,35))
plt.xlabel('Frequency / MHz')
plt.ylabel('Ion position PSD / nm^2/Hz')
plt.title('Trap: f_RF = %.3f MHz, V_RF = %.1f V, eta_RF = %.3f, eps=%.3f'%(trap['frequency']/1e6,trap['voltageRF'],trap['etaRF'],trap['eps'] ))
    
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

#%%
plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#   kB T = (1/2) m v**2

plt.subplot(3,1,1)
plt.plot(t, x, label='mean_x = %.3f nm, T_x = %.3f mK'% (np.mean(x),1e3*Tx))


plt.grid()
plt.legend()
plt.ylabel('x / nm')
plt.title('Position')

plt.subplot(3,1,2)
plt.plot(t, y, label='mean_y = %.3f nm, T_y = %.3f mK'% (np.mean(y),1e3*Ty))
plt.grid()
plt.legend()
plt.ylabel('y / nm')


plt.subplot(3,1,3)
plt.plot(t, z, label='mean_z = %.3f nm, T_z = %.3f mK'% (np.mean(z),1e3*Tz))
plt.ylabel('z / nm')
plt.xlabel('Time / s')
plt.grid()
plt.legend()


#%%
plt.figure()
trf = 1.0/frf
tmod = t % trf
tbins = np.linspace(0,trf,25)
tcenters = (tbins[0:-1]+tbins[1:])/2
idxs = np.digitize(tmod, tbins) # Return the indices of the bins to which each value in input array belongs.
vxa=[]
vya=[]
vza=[]

for n in range(24):
    vxa.append(np.mean( vx[idxs==n+1] ))
    vya.append(np.mean( vy[idxs==n+1] ))
    vza.append(np.mean( vz[idxs==n+1] ))

tmod2 = t % (1.0/(2*frf)) # double frequency
tbins2 = np.linspace(0,trf*0.5,25)
tcenters2 = (tbins2[0:-1]+tbins2[1:])/2
idxs2 = np.digitize(tmod2, tbins2)
vxa2=[]
for n in range(24):
    vxa2.append(np.mean( vx[idxs2==n+1] ))
    
    
plt.plot(tcenters,vxa,'o',label='v_x')
plt.plot(tcenters,vya,'o',label='v_y')
plt.plot(tcenters,vza,'o',label='v_z')

#plt.plot(tcenters2,vxa2,'o')
plt.ylabel('Mean Velocity / m/s')
plt.xlabel(' Time modulo 1/f_RF / s')
plt.grid()
plt.legend()

#%% Heat map
xmin,xmax,ymin,ymax = -50,50,-50,50
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[xmin, xmax], [ymin, ymax]])
s=2
from scipy.ndimage.filters import gaussian_filter
heatmap = gaussian_filter(heatmap, sigma=s)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

heatmap2, xedges2, yedges2 = np.histogram2d(x, z, bins=50, range=[[xmin, xmax], [ymin, ymax]])
heatmap2 = gaussian_filter(heatmap2, sigma=s)
extent2 = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]

heatmap3, xedges3, yedges3 = np.histogram2d(y, z, bins=50, range=[[xmin, xmax], [ymin, ymax]])
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
