import numpy as np 
import scipy.fft as fft
#For initializing the grid in simulation wuth functionality with the forms repurposed from former project
#It will be moved from a 3D to 2D grid
class Torus2D:
    def __init__(self, sizex, sizey, resx, resy):
        #Size is size of the grid 
        #res is number of grid points in each dimension 
        self.sizex, self.sizey = sizex, sizey
        self.resx, self.resy,  =round(resx), round(resy)

        #dx, dy is the edge length 
        self.dx = self.sizex/self.resx
        self.dy = self.sizey/self.resy

        #ix, iy, iz is the 1-D index array 
        self.ix = np.arange(self.resx)
        self.iy = np.arange(self.resy)

        #2D index array 
        self.iix, self.iiy  = np.meshgrid(self.ix, self.iy, indexing= 'ij')
        self.px = (self.iix )*self.dx 
        self.py = (self.iiy )*self.dy

        #spectral wavenumbers
        #kx, ky in units of radian per unit length 
        kx_1d = 2*np.pi * fft.fftfreq(self.resx, d = self.dx)
        ky_1d = 2*np.pi * fft.fftfreq(self.resy, d =self.dy)
        self.kx, self.ky = np.meshgrid(kx_1d, ky_1d, indexing = 'ij')

        #k^2 and k^4 which will be used by sepctral Laplacian and Bilaplacian 
        self.k2 = self.kx**2 + self.ky**2
        self.k4 = self.k2**2
    
    def derivativeofFunction(self, f):
        ixp = (self.ix +1) % self.resx
        iyp = (self.iy +1) % self.resy
        vx = f[ixp, :] - f
        vy = f[:, iyp] - f
        return vx,vy
    #1 form to 2 form differential
    def derivative_of_one_form(self,vx, vy, vz):
        ixp = (self.ix +1) % self.resx
        iyp = (self.iy +1) % self.resy

        w = vx - vx[:, iyp] + vy[ixp, :] - vy
        return w

    def div(self, vx, vy, vz):
        ixm = ((self.ix -1 ) % self.resx) 
        iym = ((self.iy -1 ) % self.resy) 
        f = (vx - vx[ixm, :]) / (self.dx**2)
        f += (vy - vy[:, iym]) / (self.dy**2)
        return f

    def sharp(self, vx, vy, vz):
        ixm = ((self.ix -1 ) % self.resx) 
        iym = ((self.iy -1 ) % self.resy) 

        ux = 0.5 * (vx[ixm, :] + vx) / self.dx
        uy = 0.5 * (vy[:, iym] + vy) / self.dy
        return ux, uy

    def spectral_laplacian(self, f):
        '''
        Computes laplacian spectrally 
        In fourier space Lapl(f) = -k^2 f'
        '''
        f_hat = fft.fftn(f)
        return np.real(fft.ifftn(-self.k2 * f_hat))
    
    def spectral_bilaplacian(self,f):
        #Now k^4 * f'
        f_hat = fft.fftn(f)
        return np.real(fft.ifftn(self.k4 * f_hat))

    def poisson_solve(self, f):
        f = fft.fftn(f)
        sx = np.sin(np.pi * self.iix / self.resx) / self.dx
        sy = np.sin(np.pi * self.iiy / self.resy) / self.dy

        denom = sx ** 2 + sy ** 2 
        denom[0, 0, 0] = .1  # Avoid division by zero
        f_hat = -.25/denom
        f_hat[0, 0] = 0  # Ensure zero mean
        f = f*f_hat
        f = fft.ifftn(f)
        return f