import numpy as np
from abc import ABC, abstractmethod 

#Base class to be inherited by regular FHN and mass conserved FHN
class FHNBase(ABC):
    def __init__(self, a= 0.7 ,b = 0.8, epsilon = 0.08, Du =1.0 , Dv = 0.5):
        self.a = a
        self.b = b 
        self.epsilon = epsilon 
        self.Du = Du 
        self.Dv = Dv

    def f(self, u,v):
        #equation 10
        return u - u ** 3 / 3 - v
    
    def g(self, u, v):
        #equation 11
        return u + self.a - self.b * v 
    
 
    #This for naming and displaying information when doing plots
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"a= {self.a}, b = {self.b}, epsilon = {self.epsilon,}, "
                f"Du = {self.Du}, Dv = {self.Dv}")
     

class RegularFHN(FHNBase):
    """
    Regular FHN  (eqs. 8 & 9):
 
        du/dt = f(u,v) + Du · ∇²u
        dv/dt = ε·g(u,v) + Dv · ∇²v
 
    Second-order PDE.  Simulation handles the spectral stepping.
    """
    pass


    
class MassConservedFHN(FHNBase):
    """
    Mass-conserved FHN  (eqs. 12 & 13):
 
        du/dt = −∇²[ f(u,v) + Du · ∇²u ]
        dv/dt = −∇²[ ε·g(u,v) + Dv · ∇²v ]
 
    Fourth-order PDE.  Requires bilaplacian (∇⁴) in Fourier space.
    Conserves total mass: ∫u dA = constant.
    Simulation handles the spectral stepping.
    """
    pass