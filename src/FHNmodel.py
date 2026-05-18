
import numpy as np
from abc import ABC, abstractmethod 

'Base class to be inherited by regular FHN and mass conserved FHN'
class FHNBase(ABC):
    def __init__(self, a= 0.7 ,b = 0.8, epsilon = 0.08, Du =1.0 , Dv = 0.5):
        self.a = a
        self.b = b 
        self.epsilon = epsilon 
        self.Du = Du 
        self.Dv = Dv

    def f(self, u,v):
        #equation 10 
        return u - u**3/3 -v
    
    def g(self, u, v):
        #equation 11
        return u + self.a - self.b * v 
    
    @abstractmethod
    def build_masks(self, k2, k4, dt):
        'Abstract method for setting up the integrating factor (the linear/diffusion term in FHN model)'
        ...
    
    @abstractmethod
    def spectral_update(self, u_hat, v_hat, f_hat, g_hat, k2, dt):
        'Return u_hat_new, v_hat_new after a step of exponential time difference'
        ...
 
    #This for naming and displaying information when doing plots
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"a= {self.a}, b = {self.b}, epsilon = {self.epsilon,}, "
                f"Du = {self.Du}, Dv = {self.Dv}")
     

class RegularFHN(FHNBase):
    """
    Regular FHN 
       du/dt = f + Du∇²u  
       dv/dt = ε·g + Dv∇²v  (2nd-order)
    """

    def build_masks(self, k2, k4, dt):
        return (np.exp(-self.Du * k2 * dt),
                np.exp(-self.Dv * k2 * dt))

    def spectral_update(self, u_hat, v_hat, f_hat, g_hat, k2, dt):
        mask_u, mask_v = self.masks         
        u_new = mask_u * (u_hat + dt * f_hat)
        v_new = mask_v * (v_hat + dt * self.epsilon * g_hat)
        return u_new, v_new


    
class MassConservedFHN(FHNBase):
    """
    Mass-conserved FHN  (eqs. 12 & 13):
        du/dt = −∇²[ f(u,v) + Du · ∇²u ]
        dv/dt = −∇²[ ε·g(u,v) + Dv · ∇²v ]
    """
    def build_masks(self, k2, k4, dt):
        return(np.exp(-self.Du * k4 * dt),
               np.exp(-self.Dv * k4 * dt))
    
    def spectral_update(self, u_hat, v_hat, f_hat, g_hat, k2, dt):
        mask_u, mask_v = self.masks
        u_new = mask_u * (u_hat + dt * k2 * f_hat)
        v_new = mask_v * (v_hat + dt * self.epsilon * k2 * g_hat)
        return u_new, v_new