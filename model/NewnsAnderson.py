""" Perform the Newns-Anderson model calculations."""

from dataclasses import dataclass
import numpy as np
from scipy import signal
from scipy import integrate
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
    """ 
    beta_p: float
    eps_sigma: float
    eps: list
    eps_d : float
    beta: float
    fermi_energy:float = 0.0

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
        self.eps -= self.fermi_energy

        self.eps_wrt_d = self.eps - self.eps_d

        # Calculate the lower band edge
        self.lower_band_edge = - 1 + self.eps_d 
        index_lower_band_edge = np.argmin(np.abs(self.eps - self.lower_band_edge))

        # Construct Delta in units of 2beta
        self.Delta = 2 * self.beta_p**2 * ( 1 - self.eps_wrt_d**2 )**0.5
        self.Delta = np.nan_to_num(self.Delta)
        self.Delta_at_band_edge = self.Delta[index_lower_band_edge]

        # Construct Lambda in units of 2beta
        lower_hilbert_args = []
        upper_hilbert_args = []
        for i, eps in enumerate(self.eps_wrt_d):
            if np.abs(eps) <= 1: 
                self.Lambda[i] = 2 * self.beta_p**2 * eps 
            elif eps > 1:
                self.Lambda[i] = 2 * self.beta_p**2 * ( eps - (eps**2 - 1)**0.5 )
                upper_hilbert_args.append(i)
            elif eps < -1:
                self.Lambda[i] = 2 * self.beta_p**2 * ( eps + (eps**2 - 1)**0.5 )
                lower_hilbert_args.append(i)
            else:
                raise ValueError("The epsilon value is not valid.")

        self.Lambda_at_band_edge = self.Lambda[index_lower_band_edge]

        # Adsorbate density of states ( in the units of 2 beta)
        self.rho_aa = 1 / np.pi * self.Delta / ( ( self.eps - self.eps_sigma - self.Lambda )**2 + self.Delta**2 )

        # ---------------- Check all the possible root combinations ----------------

        # Check if there is a virtual root
        if 4 * self.beta**2 + self.eps_sigma**2 < 1:
            self.has_complex_root = True
        else:
            self.has_complex_root = False 
        
        if not self.has_complex_root:
            # Find where the epsilon - epsilon_sigma line equals the Lambda line
            lower_lambda_expression = 2 * self.beta_p**2 * ( self.eps_wrt_d[lower_hilbert_args] 
                                    + (self.eps_wrt_d[lower_hilbert_args]**2 - 1)**0.5 ) 
            upper_lambda_expression = 2 * self.beta_p**2 * ( self.eps_wrt_d[upper_hilbert_args] 
                                    - (self.eps_wrt_d[upper_hilbert_args]**2 - 1)**0.5 ) 
            
            assert np.isnan(lower_lambda_expression).any() == False
            assert np.isnan(upper_lambda_expression).any() == False
            assert all(j > 0 for j in upper_lambda_expression)
            assert all(j < 0 for j in lower_lambda_expression)

            linear_energy = self.eps - self.eps_sigma
            # There is no restriction on the sign of eps - eps_a in this region
            linear_energy_lower = linear_energy[lower_hilbert_args]
            linear_energy_upper = linear_energy[upper_hilbert_args]

            # Find the roots of the two lines
            index_positive_root = np.argmin(np.abs(linear_energy_lower - lower_lambda_expression))
            index_negative_root = np.argmin(np.abs(linear_energy_upper - upper_lambda_expression))

            root_positive = self.eps[lower_hilbert_args][index_positive_root]
            root_negative = self.eps[upper_hilbert_args][index_negative_root]            

        elif self.has_complex_root:
            root_positive = ( 1 - 2*self.beta_p**2 ) * self.eps_sigma \
                            + 2j * self.beta_p**2 * ( 1 - 4 * self.beta_p**2 - self.eps_sigma**2 )**0.5 
            root_positive /= ( 1 - 4 * self.beta_p**2)
            root_negative = ( 1 - 2*self.beta_p**2 ) * self.eps_sigma \
                            - 2j * self.beta_p**2 * ( 1 - 4 * self.beta_p**2 - self.eps_sigma**2 )**0.5 
            root_negative /= ( 1 - 4 * self.beta_p**2)

            # We do not care about the imaginary root for now
            root_positive = np.real(root_positive)
            root_negative = np.real(root_negative)
            print('Has a complex root...')

            assert root_negative == root_positive

        self.root_positive = root_positive
        self.root_negative = root_negative
        
        # The energy for the state which is finally occupied or not 
        # It is the lowest lying state for the adsorbate
        self.eps_l_sigma = self.root_positive

        # Determine if there is an occupied localised state
        if self.eps_l_sigma < self.lower_band_edge and self.lower_band_edge - self.eps_sigma > self.Lambda_at_band_edge: 
            assert np.min(np.abs(linear_energy_lower - lower_lambda_expression)) < 1e-1
            self.has_localised_occupied_state = True
        else:
            self.has_localised_occupied_state = False

        # ---------- Calculate the energy ----------

        occupied_states = [i for i in range(len(self.eps)) if self.lower_band_edge < self.eps[i] < 0]

        # eps_wrt_d_region = eps_wrt_d[occupied_states]
        # pre_tan_function_numer = -1 * self.beta_p**2 / self.beta * ( 1 - eps_wrt_d_region**2 )**0.5 
        # pre_tan_function_denom = ((self.beta_p**2 / self.beta - 1 ) * eps_wrt_d_region + self.eps_sigma)

        pre_tan_function_numer = self.Delta[occupied_states] 
        pre_tan_function_denom = (self.eps - self.eps_sigma - self.Lambda)
        pre_tan_function_denom = pre_tan_function_denom[occupied_states]
        tan_integrand = np.arctan(pre_tan_function_numer / pre_tan_function_denom)

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
            self.energy =  np.trapz(tan_integrand, self.eps[occupied_states] ) / np.pi + self.eps_l_sigma

        # Calculate the Delta E for U = 0
        self.DeltaE = 2 * self.energy  -  self.eps_sigma + self.fermi_energy


        # Analytical rho
        # rho_aa = 2 / np.pi * self.beta_p**2 * ( 1 - eps_wrt_d**2 )**0.5 
        # rho_aa /= (eps_wrt_d**2 * (1 - 4*self.beta_p**2) - 2*eps_wrt_d*self.eps_sigma*(1 - 2*self.beta_p**2) + 4*self.beta_p**4 + self.eps_sigma**2)
        # self.rho_aa = rho_aa

        # if self.has_complex_root:

            # use the analytical expressions

        #     self.eps_sigma_d = self.eps_sigma + self.eps_d
        #     if self.beta_p != 0.5:
        #         root_positive = ( 1 - 2*self.beta_p**2 ) *  self.eps_sigma_d
        #         root_positive += 2*self.beta_p**2 * (4*self.beta_p**2 + self.eps_sigma_d**2 - 1)**0.5
        #         root_positive /= ( 1 - 4 * self.beta_p**2 )
        #         root_negative = ( 1 - 2*self.beta_p**2 ) * self.eps_sigma_d
        #         root_negative -= 2*self.beta_p**2 * (4*self.beta_p**2 + self.eps_sigma_d**2 - 1)**0.5
        #         root_negative /= ( 1 - 4 * self.beta_p**2 )
        #     else:
        #         root_positive = 1 + 4*self.eps_sigma_d**2
        #         root_positive /= ( 4 * self.eps_sigma_d)
        #         root_negative = root_positive

            # General expressions for the roots
            # root_positive = ( self.eps_d + self.eps_sigma ) / 2 
            # root_positive += ( (self.eps_sigma - self.eps_d )**2 + 4*self.beta**2 )**0.5 / 2
            # root_negative = ( self.eps_d + self.eps_sigma ) / 2
            # root_negative -= ( (self.eps_sigma - self.eps_d )**2 + 4*self.beta**2 )**0.5 / 2
