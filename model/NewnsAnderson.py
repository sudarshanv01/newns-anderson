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
    """ Perform the Newns-Anderson model analytically for a semi-elliplical delta.
        Inputs 
        ------
        eps_sigma: float
            normalised energy of the adsorbate in units of eV
        beta_p: float
            Coupling element of the adsorbate with the metal atom in units of 2beta
        beta: float
            Coupling element of the metal with metal in units of eV
        eps_d: float
            center of Delta in units of eV
        eps_range: list
            Range of energies to plot in units of eV

        NB: The units used throughout the analysis are: 2Beta
    """ 
    beta_p: float
    eps_sigma: float
    eps: list
    eps_d : float
    beta: float

    def __post_init__(self):
        """ Create the output quantities."""
        self.Delta = np.zeros(len(self.eps))
        self.Lambda = np.zeros(len(self.eps))
        self.rho_aa = np.zeros(len(self.eps))
        self.create_quantities()


    def create_quantities(self):
        """ Create the needed quantities for the Newns Anderson model."""

        # Convert the energies to units of 2beta
        self.eps = self.eps / 2 / self.beta
        self.eps_d = self.eps_d / 2 / self.beta
        self.eps_sigma = self.eps_sigma / 2 / self.beta

        # Energies with respect to moving d-band energy
        eps_wrt_d = self.eps - self.eps_d

        # Some specifics about the calculation
        print(f'Epsilon_sigma {self.eps_sigma}')

        # Calculate the lower band edge
        self.lower_band_edge = self.eps_d - self.beta_p**2 / self.beta / self.beta_p 
        print(f'The lower band edge is {self.lower_band_edge} for eps_d {self.eps_d}')

        # Construct Delta in units of 2beta
        self.Delta = 2 * self.beta_p**2 * ( 1 - (self.eps - self.eps_d)**2 )**0.5
        self.Delta = np.nan_to_num(self.Delta)
        self.Delta /= (2 * self.beta)

        # Construct Lambda in units of 2beta
        for i, eps in enumerate(self.eps):
            if np.abs(eps_wrt_d[i]) <= 1:
                self.Lambda[i] = 2 * self.beta_p**2 * eps_wrt_d[i]  
            elif eps_wrt_d[i] > 1:
                self.Lambda[i] = 2 * self.beta_p**2 * ( eps_wrt_d[i] - (eps_wrt_d[i]**2 - 1)**0.5 )
            elif eps_wrt_d[i] < -1:
                self.Lambda[i] = 2 * self.beta_p**2 * ( eps_wrt_d[i] + (eps_wrt_d[i]**2 - 1)**0.5 )
            else:
                raise ValueError("The epsilon value is not valid.")
        self.Lambda /= (2 * self.beta)

        # Adsorbate density of states ( in the units of 2beta )
        self.rho_aa = 1 / np.pi * self.Delta / ( ( self.eps - self.eps_sigma - self.Lambda )**2 + self.Delta**2 )

        # ---------------- Check all the possible root combinations ----------------

        # Check if there is a virtual root
        if 2 * self.beta_p**2 / self.beta + self.eps_sigma**2 < 1:
            self.has_complex_root = True
        else:
            self.has_complex_root = False 
        
        if self.has_complex_root:
            root_positive = ( 1 - self.beta_p**2 / self.beta ) * self.eps_sigma \
                            + 1j * ( self.beta_p**2 / self.beta ) * ( 1 - 2 * self.beta_p**2 / self.beta - self.eps_sigma**2 )**0.5 
            root_positive /= ( 1 - 2 * self.beta_p**2 / self.beta )
            root_negative = ( 1 - self.beta_p**2 / self.beta ) * self.eps_sigma \
                            - 1j * ( self.beta_p**2 / self.beta ) * ( 1 - 2 * self.beta_p**2 / self.beta - self.eps_sigma**2 )**0.5
            root_negative /= ( 1 - 2 * self.beta_p**2 / self.beta )
        else:
            if self.beta_p == 0.5:
                root_positive = ( 1 + 4 * self.eps_sigma**2 ) / 4 / self.eps_sigma**2
                root_negative = root_positive
            else:
                root_positive = ( 1 - self.beta_p**2 / self.beta ) * self.eps_sigma \
                                + 1 * ( self.beta_p**2 / self.beta ) * ( - 1 + 2 * self.beta_p**2 / self.beta + self.eps_sigma**2 )**0.5 
                root_positive /= ( 1 - 2 * self.beta_p**2 / self.beta )
                root_negative = ( 1 - self.beta_p**2 / self.beta ) * self.eps_sigma \
                                - 1 * ( self.beta_p**2 / self.beta ) * ( - 1 + 2 * self.beta_p**2 / self.beta + self.eps_sigma**2 )**0.5
                root_negative /= ( 1 - 2 * self.beta_p**2 / self.beta )

        self.root_positive = root_positive
        self.root_negative = root_negative
        
        # The energy for the state which is finally occupied
        self.eps_l_sigma = self.root_positive

        # Determine if there is an occupied localised state
        if self.eps_sigma <= self.eps_d and self.eps_l_sigma <= self.lower_band_edge:
            self.has_localised_occupied_state = True
            print('This is a localised occupied state, will correct energies...')
        else:
            self.has_localised_occupied_state = False

        # ---------- Calculate the energy ----------

        occupied_states = [i for i in range(len(self.eps)) if self.lower_band_edge < self.eps[i] < 0]

        # pre_tan_function = self.Delta / ( self.eps - self.eps_sigma - self.Lambda)
        # eps_wrt_d_region = eps_wrt_d[occupied_states]
        # pre_tan_function_numer = -1 * self.beta_p**2 / self.beta * ( 1 - eps_wrt_d_region**2 )**0.5 
        # pre_tan_function_denom = ((self.beta_p**2 / self.beta - 1 ) * eps_wrt_d_region + self.eps_sigma)
        pre_tan_function_numer = self.Delta[occupied_states] 
        pre_tan_function_denom = (self.eps - self.eps_sigma - self.Lambda)
        pre_tan_function_denom = pre_tan_function_denom[occupied_states]
        # integrand = integrand[occupied_states]
        tan_integrand = np.arctan(pre_tan_function_numer / pre_tan_function_denom)
        # tan_integrand = np.arctan2(pre_tan_function_numer, pre_tan_function_denom)
        check_tan = np.tan(tan_integrand)
        assert all(tan_integrand <= np.pi/2)
        assert all(tan_integrand >= -np.pi/2)

        # Modify the range of arctan depending on 
        # if there is a localised state or not
        # if there is it must be within 0 to pi
        # otherwise it must be between -pi to 0
        # In addition, if there is a localised state
        # it must also have the extra energy associated with 
        # that state epsilon_l_sigma
        if not self.has_localised_occupied_state:
            for i in range(len(tan_integrand)):
                if tan_integrand[i] > 0:
                    tan_integrand[i] -= np.pi
            assert all(tan_integrand <= 0)
            assert all(tan_integrand >= -np.pi)
            assert np.allclose(np.tan(tan_integrand), check_tan)
            self.energy =  np.trapz(tan_integrand, self.eps[occupied_states] ) / np.pi 

        elif self.has_localised_occupied_state:
            for i in range(len(tan_integrand)):
                if tan_integrand[i] < 0:
                    tan_integrand[i] += np.pi
            assert all(tan_integrand >= 0)
            assert all(tan_integrand <= np.pi)
            assert np.allclose(np.tan(tan_integrand), check_tan)
            self.energy =  np.trapz(tan_integrand, self.eps[occupied_states] ) / np.pi  + self.eps_l_sigma 

        # Calculate the Delta E for U = 0
        self.DeltaE = 2 * self.energy  -  self.eps_sigma 


        # Analytical rho
        # rho_aa = 2 / np.pi * self.beta_p**2 * ( 1 - eps_wrt_d**2 )**0.5 
        # rho_aa /= (eps_wrt_d**2 * (1 - 4*self.beta_p**2) - 2*eps_wrt_d*self.eps_sigma*(1 - 2*self.beta_p**2) + 4*self.beta_p**4 + self.eps_sigma**2)
        # self.rho_aa = rho_aa