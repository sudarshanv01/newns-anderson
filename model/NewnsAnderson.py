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
        neg_index = [ a for a in range(len(self.eps)) if self.eps[a] < 0 ]
        self.fill_energy = self.eps[neg_index]
        self.fill_na = self.na[neg_index]
        self.fill_Delta = self.Delta[neg_index]
        self.fill_Lambda = self.Lambda[neg_index]

        pre_tan_function = self.fill_Delta / (self.fill_energy - self.eps_a - self.fill_Lambda) 
        hybridisation_ = np.arctan(pre_tan_function)
        for i, hyb in enumerate(hybridisation_):
            if hyb > 0:
                hybridisation_[i] = hyb - np.pi 
        assert all(hybridisation_ <= 0)
        assert all(hybridisation_ >= -np.pi)

        self.energy = 2 / np.pi * np.trapz(hybridisation_, self.fill_energy) 

@dataclass
class NewnsAndersonAnalytical:
    """ Perform the Newns-Anderson model analytically for a semi-elliplical delta.""" 
    # Note: The units used throughout are that of 2Beta
    # just like in the paper
    # This class is mainly to used for recreating the figures of the paper
    beta: float
    eps_sigma: float
    eps_range: list
    eps_d : float

    def __post_init__(self):
        """ Create the output quantities."""
        self.eps = self.eps_range
        self.Delta = np.zeros(len(self.eps))
        self.Lambda = np.zeros(len(self.eps))
        self.rho_aa = np.zeros(len(self.eps))

        self.create_quantities()

    def create_quantities(self):
        """ Create the needed quantities for the Newns Anderson model."""

        self.Delta = 2 * self.beta**2 * ( 1 - (self.eps - self.eps_d)**2 )**0.5
        self.Delta = np.nan_to_num(self.Delta)
        # Unit coversion to 2beta
        self.Delta /= (2 * self.beta)

        eps_wrt_d = self.eps - self.eps_d
        for i, eps in enumerate(self.eps):
            # Define the quantities based on the absolute values of energy
            if np.abs(eps_wrt_d[i]) <= 1:
                self.Lambda[i] = 2 * self.beta**2 * eps_wrt_d[i]  
            elif eps_wrt_d[i] > 1:
                self.Lambda[i] = 2 * self.beta**2 * ( eps_wrt_d[i] - (eps_wrt_d[i]**2 - 1)**0.5 )
            elif eps_wrt_d[i] < -1:
                self.Lambda[i] = 2 * self.beta**2 * ( eps_wrt_d[i] + (eps_wrt_d[i]**2 - 1)**0.5 )
            else:
                raise ValueError("The epsilon value is not valid.")
        # unit coversion
        self.Lambda /= (2 * self.beta)

        # denom = self.eps**2 * (1 - 4 * self.beta**2) - 2 * self.eps * self.eps_sigma * ( 1 - 2 * self.beta**2 ) + 4 * self.beta**4 + self.eps_sigma**2
        # self.rho_aa = 2 / np.pi * self.beta**2 * ( 1 - self.eps**2 )**0.5 / denom
        # self.rho_aa = np.nan_to_num(self.rho_aa)

        self.rho_aa = 1 / np.pi * self.Delta / ( ( (self.eps - self.eps_sigma)/self.beta/2 - self.Lambda )**2 + self.Delta**2 )

        # Determining the energy for this configuration
        # intergrand = -2 * self.beta**2 * ( 1 - eps_wrt_d**2 )**0.5 
        # intergrand /= ( ( 2 * self.beta**2 - 1) * self.eps + self.eps_sigma )
        # intergrand = np.nan_to_num(intergrand)
        occupied_states = [i for i in range(len(self.eps)) if  self.eps[i]/2/self.beta < 0]

        integrand = self.Delta / ((self.eps - self.eps_sigma)/2/self.beta - self.Lambda)
        integrand = integrand[occupied_states]

        # Adjust the limits
        #tan_integrand =  np.arctan(integrand) # These angles are from -pi/2 to pi/2
        #tan_integrand = tan_integrand - np.pi/2 % (np.pi/2) # These angles are from 0 to pi
        pre_tan_function = self.Delta / ( (self.eps - self.eps_sigma)/2/self.beta - self.Lambda)
        tan_integrand = np.arctan(pre_tan_function)
        for i in range(len(tan_integrand)):
            if tan_integrand[i] > 0:
                tan_integrand[i] -= np.pi
        tan_integrand = tan_integrand[occupied_states]
        assert all(tan_integrand <= 0)
        assert all(tan_integrand >= -np.pi)
        
        self.energy = 2 * np.trapz(tan_integrand, self.eps[occupied_states] / 2 / self.beta) / np.pi 
        self.hyb_energy = self.energy - self.eps_sigma / 2 / self.beta