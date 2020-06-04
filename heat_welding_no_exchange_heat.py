# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:15:00 2020

@author: andre
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import glob

#Set Time Step
dt=0.0002
# Set colour interpolation and colour map.
# You can try set it to 10, or 100 to see the difference
# You can also try: colourMap = plt.cm.coolwarm
colorinterpolation = 50
colourMap = plt.cm.jet


# Set Total Domain
lenX = lenY = 20 #we set it rectangular
# intervals in x-, y- directions, mm
dx = dy = 0.2
dx2 = dx*dx
dy2 = dy*dy
#number of elements 
nx = int(lenX/dx)
ny = int(lenY/dy)

#-------------PART 1 ------------

# Thermal diffusivity in X, mm2.s-1
kx1 = 10.
# Thermal diffusivity in y, mm2.s-1
ky1 = 10.
# Boundary condition
# Initial Temperature of interior grid
T1ini = 10.
nx1=nx
ny1=int(ny/2)
#-------------PART 2 ------------
# Thermal diffusivity in X, mm2.s-1
kx2 = 10.
# Thermal diffusivity in y, mm2.s-1
ky2 = 10.
# Boundary condition
# Initial Temperature of interior grid
T2ini = 10.
nx2=nx
ny2=ny


# Set meshgrid
X, Y = np.meshgrid(np.arange(0, nx), np.arange(0, ny))
X=X[1,:].copy()
Y=Y[:,1].copy()


# Initial conditions
T = np.zeros((nx, ny))
T[0:ny1, 0:nx1]=T1ini # Set array size and set the interior value with Tini
T[ny1:ny2, 0:nx2]=T2ini # Set array size and set the interior value with Tini
T[int(nx/2), int(0.2*nx):int(0.8*nx+1)] = 100
T[int(nx/2-1), int(0.2*nx):int(0.8*nx+1)] = 100
T0=T.copy()

# start calculation

def do_timestep(u0, u,kx,ky):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + kx* dt * ((u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2 )+ ky* dt * ( (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )
    u0 = u.copy()
    return u0, u


# Number of timesteps
nsteps = 10001
t=[]

T1=T[0:ny1, 0:nx1].copy()
T01=T1.copy()
T2=T[ny1:ny2, 0:nx2].copy()
T02=T1.copy()

# Output figures at these timesteps
mfig=[]
for i in range (0,10001,500):
    mfig.append(i)
fignum = 0
fig = plt.figure()
Tall=[]
for m in range(nsteps):
    T01, T1 = do_timestep(T01, T1,kx1,ky1)
    T02, T2 = do_timestep(T02, T2,kx1,ky1)


    T01=T1.copy()
    T02=T2.copy()

    Tall= np.concatenate((T1, T2), axis=0)
    T0all= np.concatenate((T01, T02), axis=0)

    if m in mfig:
        plt.clf()
        fignum += 1
        print(m, fignum)
        plt.pcolormesh(X,Y,T0all,vmin=0, vmax=100,cmap=colourMap)
#        plt.contourf(X, Y, T0,Vmin=0, Vmax=100 colorinterpolation, cmap=colourMap)
        # Set Colorbar
        plt.colorbar()
        plt.savefig('temperature' + str("%03d" %fignum) + '.jpg')
        
#fig.subplots_adjust(right=0.85)
#cbar_ax = fig.add_axes([.9, 0.15, .03, 0.7])
#cbar_ax.set_xlabel('$T$  (C)', labelpad=20)
#fig.colorbar(im, cax=cbar_ax)
plt.show()
    


#creating the gif
# Create the frames
frames = []
imgs = glob.glob("*.jpg")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
    print(i)
 
# Save into a GIF file that loops forever
frames[0].save('temperature.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=200, loop=0)


#results[1][1,1] is the Temperature at time 1 on X1,Y1
