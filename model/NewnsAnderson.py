""" Perform the Newns-Anderson model calculations."""

from dataclasses import dataclass
import numpy as np
from scipy import signal
from scipy import integrate
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

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
        fermi_energy: float
            Fermi energy in the units of eV 
        
        NB: The energies in this class, just like in the Newns paper are referenced to 
            the metal band center.
    """ 
    beta_p: float
    eps_sigma: float
    eps: list
    eps_d : float
    beta: float
    fermi_energy: float = 0.0

    def __post_init__(self):
        """ Create the output quantities."""
        self.Delta = np.zeros(len(self.eps))
        self.Lambda = np.zeros(len(self.eps))
        self.rho_aa = np.zeros(len(self.eps))
        self.create_quantities()


    def create_quantities(self):
        """ Create the needed quantities for the Newns Anderson model."""
        # coversion factor to divide by to convert to 2beta units
        self.convert = 2 * self.beta

        # Convert the energies to units of 2beta
        # The zero of these energies is defined by the d-band center
        self.eps = self.eps / self.convert 
        self.eps_d = self.eps_d / self.convert
        self.eps_sigma = self.eps_sigma / self.convert 
        self.fermi_energy = self.fermi_energy / self.convert

        # Energies referenced to the d-band center
        self.eps_wrt_d = self.eps - self.eps_d
        self.eps_sigma_wrt_d = self.eps_sigma - self.eps_d

        # Construct Delta in units of 2beta
        self.width_of_band =  1 
        self.Delta = 2 * self.beta_p**2 * ( 1 - self.eps_wrt_d**2 )**0.5
        self.Delta = np.nan_to_num(self.Delta)

        # Calculate the positions of the upper and lower band edge
        self.lower_band_edge = - self.width_of_band + self.eps_d
        self.upper_band_edge = + self.width_of_band + self.eps_d
        index_lower_band_edge = np.argmin(np.abs(self.eps - self.lower_band_edge))
        index_upper_band_edge = np.argmin(np.abs(self.eps - self.upper_band_edge))
        self.Delta_at_lower_band_edge = self.Delta[index_lower_band_edge]
        self.Delta_at_upper_band_edge = self.Delta[index_upper_band_edge]

        # Construct Lambda in units of 2beta
        lower_hilbert_args = []
        upper_hilbert_args = []
        for i, eps in enumerate(self.eps_wrt_d):
            if np.abs(eps) <= self.width_of_band: 
                self.Lambda[i] = 2 * self.beta_p**2 * eps 
            elif eps > self.width_of_band:
                self.Lambda[i] = 2 * self.beta_p**2 * ( eps - (eps**2 - 1)**0.5 )
                upper_hilbert_args.append(i)
            elif eps < -self.width_of_band:
                self.Lambda[i] = 2 * self.beta_p**2 * ( eps + (eps**2 - 1)**0.5 )
                lower_hilbert_args.append(i)
            else:
                raise ValueError("The epsilon value is not valid.")

        self.Lambda_at_lower_band_edge = self.Lambda[index_lower_band_edge]
        self.Lambda_at_upper_band_edge = self.Lambda[index_upper_band_edge]

        # ---------------- Adsorbate density of states ( in the units of 2 beta)
        rho_aa_ = self.eps_wrt_d**2 * ( 1 - 4 * self.beta_p**2 )
        rho_aa_ += - 2 * self.eps_wrt_d * self.eps_sigma_wrt_d * ( 1 - 2 * self.beta_p**2 )
        rho_aa_ += 4 *self.beta_p**4 + self.eps_sigma_wrt_d**2
        self.rho_aa = 2 * self.beta_p**2 * ( 1 - self.eps_wrt_d**2 )**0.5
        self.rho_aa /= rho_aa_ 
        self.rho_aa /= np.pi
        self.rho_aa = np.nan_to_num(self.rho_aa)

        # ---------------- Check all the possible root combinations ----------------
        # Check if there is a virtual root
        if 4 * self.beta_p**2 + self.eps_sigma_wrt_d**2 < 1:
            self.has_complex_root = True
            print('Complex root found!')
        else:
            self.has_complex_root = False 
        
        if not self.has_complex_root:
            if self.beta_p != 0.5:
                root_positive = ( 1 - 2*self.beta_p**2 ) *  self.eps_sigma_wrt_d
                root_positive += 2*self.beta_p**2 * (4*self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**0.5
                root_positive /= ( 1 - 4 * self.beta_p**2 )
                root_negative = ( 1 - 2*self.beta_p**2 ) * self.eps_sigma_wrt_d
                root_negative -= 2*self.beta_p**2 * (4*self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**0.5
                root_negative /= ( 1 - 4 * self.beta_p**2 )
            else:
                root_positive = 1 + 4*self.eps_sigma_wrt_d**2
                root_positive /= ( 4 * self.eps_sigma_wrt_d)
                root_negative = root_positive

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

            # assert root_negative == root_positive

        # Store the root referenced to the d-band center
        self.root_positive = root_positive + self.eps_d
        self.root_negative = root_negative + self.eps_d
        
        # Determine if there is an occupied localised state
        if self.root_positive < self.lower_band_edge and self.lower_band_edge - self.eps_sigma > self.Lambda_at_lower_band_edge:
            # Check if the root is below the Fermi level
            if self.root_positive < self.fermi_energy:
                # the energy for this point is to be included
                self.has_localised_occupied_state_positive = True
            else:
                self.has_localised_occupied_state_positive = False
            # in both cases it is appropriate to store it as eps_l_sigma because it is localised state
            self.eps_l_sigma_pos = self.root_positive
        else:
            self.eps_l_sigma_pos = None
            self.has_localised_occupied_state_positive = False
        
        # Check if there is a localised occupied state for the negative root
        if self.root_negative > self.upper_band_edge and self.upper_band_edge - self.eps_sigma < self.Lambda_at_upper_band_edge:
            # Check if the root is below the Fermi level
            if self.root_negative < self.fermi_energy:
                # the energy for this point is to be included
                self.has_localised_occupied_state_negative = True
            else:
                self.has_localised_occupied_state_negative = False
            # in both cases it is appropriate to store it as eps_l_sigma because it is localised state
            self.eps_l_sigma_neg = self.root_negative
        else:
            self.eps_l_sigma_neg = None
            self.has_localised_occupied_state_negative = False

        # Expectancy value of the occupied localised state
        if self.has_localised_occupied_state_positive:
            # Compute the expectancy value
            if self.beta_p != 0.5:
                self.na_sigma_pos = (1 - 2 * self.beta_p**2)
                self.na_sigma_pos -= 2 * self.beta_p**2 * self.eps_sigma_wrt_d * (4 * self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**0.5 
                self.na_sigma_pos /= (1 - 4 * self.beta_p**2)
            else:
                self.na_sigma_pos = 4 * self.eps_sigma_wrt_d**2 - 1
                self.na_sigma_pos /= (4 * self.eps_sigma_wrt_d**2)
        else:
            self.na_sigma_pos = None
        
        if self.has_localised_occupied_state_negative:
            # Compute the expectancy value
            if self.beta_p != 0.5:
                self.na_sigma_neg = (1 - 2 * self.beta_p**2)
                self.na_sigma_neg += 2 * self.beta_p**2 * self.eps_sigma_wrt_d * (4 * self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**0.5 
                self.na_sigma_neg /= (1 - 4 * self.beta_p**2)
            else:
                self.na_sigma_neg = 4 * self.eps_sigma_wrt_d**2 - 1
                self.na_sigma_neg /= (4 * self.eps_sigma_wrt_d**2)

        # ---------- Calculate the energy ----------
        # Determine the upper bounds for the contour integration
        if self.upper_band_edge > 0:
            upper_bound = 0 
        else:
            upper_bound = self.upper_band_edge
        occupied_states = [i for i in range(len(self.eps)) if self.lower_band_edge < self.eps[i] < upper_bound]

        # Determine the integrand 
        energy_occ = self.eps_wrt_d[occupied_states]
        numerator = - 2 * self.beta_p**2 * (1 - energy_occ**2)**0.5
        denominator = energy_occ * (2*self.beta_p**2 - 1) + self.eps_sigma_wrt_d

        # This number will always be between [-pi, 0]
        arctan_integrand = np.arctan2(numerator, denominator)
        assert all(arctan_integrand < 0)
        assert all(arctan_integrand > -np.pi)

        if self.has_localised_occupied_state_positive and self.has_localised_occupied_state_negative:
            # Both positive and negative root are localised and occupied
            arctan_integrand += np.pi
            self.arctan_component =  np.trapz( arctan_integrand, energy_occ )
            self.arctan_component /= np.pi
            self.energy = self.arctan_component
            self.energy += self.eps_l_sigma_pos - self.eps_l_sigma_neg
        elif self.has_localised_occupied_state_positive:
            # Has only positive root and it is a localised occupied state 
            arctan_integrand += np.pi
            self.arctan_component =  np.trapz( arctan_integrand, energy_occ )
            self.arctan_component /= np.pi
            self.energy = self.arctan_component
            self.energy += self.eps_l_sigma_pos
        elif self.has_localised_occupied_state_negative:
            # Has only negative root and it is a localised occupied state
            self.arctan_component =  np.trapz( arctan_integrand, energy_occ )
            self.arctan_component /= np.pi
            self.energy = self.arctan_component
            self.energy -= self.eps_l_sigma_neg
            self.energy += self.upper_band_edge
        else:
            # Has no localised occupied states
            self.arctan_component =  np.trapz( arctan_integrand, energy_occ )
            self.arctan_component /= np.pi
            self.energy = self.arctan_component
        
        self.DeltaE_1sigma = self.energy 
        self.DeltaE = 2 * self.DeltaE_1sigma + self.fermi_energy -  1 * self.eps_sigma