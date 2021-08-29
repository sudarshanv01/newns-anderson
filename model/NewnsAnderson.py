""" Perform the Newns-Anderson model calculations."""

from dataclasses import dataclass
import numpy as np
from scipy import signal
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

@dataclass
class NewnsAndersonModel:
    """ Perform the Newns-Anderson model calculations."""
    coupling: float
    eps_a: float
    eps_d: float
    eps: float

    def create_semi_elliptical_dos(self):
        """ Create the semi-elliptical DOS."""
        delta_ =  (1 - ( self.eps - self.eps_d )**2 / self.coupling**2)**0.5 
        # convert nan to zeros
        self.Delta = np.nan_to_num(delta_)
        self.Lambda = np.imag(signal.hilbert(self.Delta))

    
    def generate_adsorbate_states(self):
        """ Generate the adsorbate states."""
        na_ = self.Delta / ( (self.eps - self.eps_a - self.Lambda)**2 + self.Delta**2 )
        self.na = na_ / np.pi
    
    def calculate(self):
        """ Calculate the Newns-Anderson model."""
        self.create_semi_elliptical_dos()
        self.generate_adsorbate_states()

        # Get the negative energies
        neg_index = np.where(self.eps <= 0)
        self.fill_energy = self.eps[neg_index]
        self.fill_na = self.na[neg_index]
        self.fill_Delta = self.Delta[neg_index]
        self.fill_Lambda = self.Lambda[neg_index]

        hybridisation_ = np.arctan(self.fill_Delta / (self.fill_energy - self.eps_a - self.fill_Lambda))
        self.energy = 2 / np.pi * np.trapz(hybridisation_, self.fill_energy) #- self.eps_a
    
    