import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sps


# sample dimensions
S_Nx = 300
S_Ny = 300
S_Nz = 300

try:
    sample = np.load('sample.npy')
    assert sample.shape[0]==S_Nz
    assert sample.shape[1]==S_Ny
    assert sample.shape[2]==S_Nx
except Exception as e:


    sample = np.random.rand(S_Nz,S_Ny,S_Nx)
    sample = np.round(sample*0.501)

    ZZ,YY,XX = np.meshgrid(np.arange(S_Nz),np.arange(S_Ny),np.arange(S_Nx))

    ZZ = ZZ - S_Nz//2
    YY = YY - S_Ny//2
    XX = XX - S_Nx//2

    rad = np.sqrt(ZZ**2+YY**2+XX**2)
    kernel = np.zeros(rad.shape)
    kernel[np.where(rad<3)] = 1.0
    sample = sps.fftconvolve(sample,kernel,mode='same')
    np.save('sample.npy',sample)

# for z in range(sample.shape[0]):
#     plt.cla()
#     plt.imshow(sample[z,:,:])
#     plt.pause(.00001)


# image dimensions
I_Nx = 100
I_Ny = 100
I_Nz = 100

I_x0 = (S_Nx-I_Nx)//2
I_y0 = (S_Ny-I_Ny)//2
I_z0 = (S_Nz-I_Nz)//2

dt = 1e-4
t_vec = np.arange(0,10,dt)

# acceleration standard deviations
ddx_std = 5000*dt
ddy_std = 5000*dt
ddz_std = 5000*dt

# initial accelerations
ddx0 = 0.0
ddy0 = 0.0
ddz0 = 0.0

# initial velocities
dx0 = 0.0
dy0 = 0.0
dz0 = 0.0

# initial positions
x0 = 0.0
y0 = 0.0
z0 = 0.0


# eye traces
x_trace = np.zeros(t_vec.shape)
y_trace = np.zeros(t_vec.shape)
z_trace = np.zeros(t_vec.shape)

x,dx,ddx = x0,dx0,ddx0
y,dy,ddy = y0,dy0,ddy0
z,dz,ddz = z0,dz0,ddz0


def move(p,dp,ddp,ddp_std,dt):
    # acceleration is always a new normally distributed value, with a slight bias toward zero (fixation)
    ddp = dt*(np.random.randn()*ddp_std - np.sign(p)*.1*ddp_std)
    dp = dp + ddp
    p = p + dp
    return p,dp,ddp

for idx,t in enumerate(t_vec):
    x,dx,ddx=move(x,dx,ddx,ddx_std,dt)
    y,dy,ddy=move(y,dy,ddy,ddy_std,dt)
    z,dz,ddz=move(z,dz,ddz,ddz_std,dt)
    x_trace[idx] = x
    y_trace[idx] = y
    z_trace[idx] = z

def get_position(t):
    idx = np.argmin(np.abs(t_vec-t))
    return int(round(x_trace[idx])),int(round(y_trace[idx])),int(round(z_trace[idx]))

I = np.zeros((I_Nz,I_Ny,I_Nx))

t = 0.0
scan_time = 1e-4
for ky in range(I_Ny):
    for kx in range(I_Nx):
        xerr,yerr,zerr = get_position(t)
        print(t,xerr,yerr,zerr)
        kz1 = I_z0
        kz2 = I_z0+I_Nz
        line = sample[kz1+zerr:kz2+zerr,+I_y0+ky+yerr,I_x0+kx+xerr]
        I[:,ky,kx] = line
        t = t + scan_time

    plt.cla()
    plt.imshow(I[:,ky,:])
    plt.pause(.00001)

        
    
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_trace, y_trace, z_trace, 'k.',label='parametric curve',markersize=1)
plt.show()



sys.exit()
data = np.random.rand(8,8,8)

a = np.random.rand(10,10,10)
b = np.random.rand(10,10,10)

a[:8,:8,:8] = data
b[2:,2:,2:] = data

af = np.fft.fftn(a)
bf = np.fft.fftn(b).conj()

xc = np.abs(np.fft.ifftn(af*bf))

clim = (xc.min(),xc.max())
for k in range(xc.shape[0]):
    plt.clf()
    plt.imshow(xc[k,:,:],clim=clim)
    plt.colorbar()
    plt.pause(1)

