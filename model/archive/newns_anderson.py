
"""Base class for all quantities to plot the equations for the Newns Anderson Model."""

import numpy as np
from scipy import signal

class NewnsAnderson:


    def __init__(self, **kwargs):
        ENERGY_RANGE = np.linspace(-15, 15, 1000)
        EPS_A = 2.5
        EPS_D = 2.5
        COUPLING_ELEMENT = 2.5

        """Initialize the Newns Anderson model."""
        self.eps = ENERGY_RANGE
        self.eps_a = EPS_A
        self.eps_d = EPS_D
        self.V = COUPLING_ELEMENT

        # override any defaults with the kwargs
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        """Return a string representation of the Newns Anderson model."""
        return f"NewnsAnderson({self.__dict__})"

    def _create_metal_dos(self):
        """Create the ideal semi-eliptical DOS for the Newns Anderson model.""" 
        #delta_ = ( 1 - ( ( np.abs(self.eps - self.eps_d) ) / self.V )**2 ) **0.5
        delta_ = ( self.V**2 - ( self.eps - self.eps_d )**2 ) **0.5
        delta_ = np.nan_to_num(delta_)
        # set the negative numbers to zero
        self.Delta = delta_
        # store the Hilbert transform of Delta
        self.Lambda = np.imag(signal.hilbert(delta_))
    
    def _create_adsorbate_dos(self):
        """Create the adsorbate states from the Newns Anderson model."""
        na_ = self.Delta / ( (self.eps - self.eps_a - self.Lambda)**2 + self.Delta**2 ) / np.pi
        self.na = na_

    def run(self):
        """Calculate the energy of the adsorbate states."""
        self._create_metal_dos()
        self._create_adsorbate_dos()

        # get all the negative elements of the energy
        neg_indices = np.where(self.eps < 0)
        filled_energies = self.eps[neg_indices]
        filled_Delta = self.Delta[neg_indices]
        filled_Lambda = self.Lambda[neg_indices]

        energy_integrand = np.arctan( filled_Delta / ( filled_energies - self.eps_a - filled_Lambda ) )
        energy_integrand *= 2
        energy_integrand /= np.pi

        self.energy = np.trapz(energy_integrand, filled_energies)
        self.energy -= self.eps_a


    