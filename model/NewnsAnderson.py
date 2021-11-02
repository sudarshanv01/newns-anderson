""" Perform the Newns-Anderson model calculations."""

from dataclasses import dataclass
import numpy as np
from scipy import signal
from scipy import integrate
import warnings
from pprint import pprint
warnings.filterwarnings("ignore", category=RuntimeWarning) 

@dataclass
class NewnsAndersonNumerical:
    """Perform numerical calculations of the Newns-Anderson model to get 
    the chemisorption energy.
    Vak: float
        Coupling matrix element
    eps_a: float
        Energy of the adsorbate
    eps_d: float
        Energy of the d-band
    eps: list
        Energy range to consider
    width: float
        Width of the d-band
    k: float
        Parameter to control the amount of added extra states
    """
    Vak: float
    eps_a: float
    width: float
    eps_d: float
    eps: float
    k: float
    NOISE_TOLERANCE = 1e-1

    def __post_init__(self):
        """Perform numerical calculations of the Newns-Anderson model to get 
        the chemisorption energy."""
        # Store some useful variables
        self.eps_wrt_d = self.eps - self.eps_d
    
    def _create_Delta(self):
        """Create Delta from Vak and eps_d."""
        self.Delta =  ( 1  - ( self.eps_wrt_d )**2 / ( self.width / 2 )**2  )**0.5
        self.Delta = np.nan_to_num(self.Delta, nan=0)
        # Add a constant Delta0 to the Delta
        area_under_Delta = integrate.simps(self.Delta, self.eps)
        self.Delta /= area_under_Delta 
        # This pi**2 must come from the change in beta to Vak
        self.Delta *= np.pi**2
        # The area that it should be for a certain Vak
        area_should_be = np.pi * self.Vak
        self.Delta *= area_should_be

        self.sp_contributions = self.k
        self.Delta += self.sp_contributions

    
    def _create_Lambda(self):
        """Create Lambda by performing the Hilbert transform of Delta."""
        self.Lambda = np.imag(signal.hilbert(self.Delta))

    def calculate_dos(self):
        """Calculate the DOS from the Delta and Lambda."""
        dos_ = self.Delta / ( (self.eps - self.eps_a - self.Lambda)**2 + self.Delta**2 )
        dos_ /= np.pi
        self.dos = dos_
        # Determine the first part of the occupancy by integrating up the dos between 
        # eps_d +- width/2
        between_d_energies = np.where((self.eps >= self.eps_d - self.width/2) & (self.eps <= self.eps_d + self.width/2))[0]
        integrated_rho = integrate.simps(dos_[between_d_energies], self.eps[between_d_energies])
        # Now look for the localised states
        # Find the numerical derivative of Lambda
        self.dLambda = np.diff(self.Lambda) / np.diff(self.eps)
        # Get the index of Delta closest to the d-band center
        self.delta_index = np.argsort(np.abs(self.eps - self.eps_d))[0]

        # Find cases where eps - eps_a - Lambda is close to 0
        self.dLambda_index = np.where(np.abs(self.eps - self.eps_a - self.Lambda) < self.NOISE_TOLERANCE)[0]
        assert len(self.dLambda_index) > 0, "There must be at least one root"
        # Sort all these cases to find which ones are roots 
        self.dLambda_index = np.sort(self.dLambda_index)
        # There are only three possible roots possible
        # If the d-band center index is between the first and the third root index
        # Then there are three roots
        self.lower_index_root = None
        self.upper_index_root = None
        if np.min(self.dLambda_index) < self.delta_index < np.max(self.dLambda_index):
            # There are three possible roots is the two indices enclose that of delta
            print(f"There are three roots for {self.eps_d} eV")
            na_ = 0
            if self.eps[np.min(self.dLambda_index)] <= 0:
                # Check if the index isnt in the d-band
                if self.Delta[np.min(self.dLambda_index)] == self.sp_contributions:
                    print('Lower one occupied')
                    na_ += 1 / ( 1 - self.dLambda[np.min(self.dLambda_index)] )
                    self.lower_index_root = np.min(self.dLambda_index)
            if self.eps[np.max(self.dLambda_index)] <= 0:
                # Check if the index isnt in the d-band
                if self.Delta[np.max(self.dLambda_index)] == self.sp_contributions:
                    print('Upper one occupied')
                    na_ += 1 / ( 1 - self.dLambda[np.max(self.dLambda_index)] )
                    self.upper_index_root = np.max(self.dLambda_index)
            na_ += integrated_rho
        else:
            # There is only one root here, find the one which has the 
            # lowest energy when the energies are negative
            filled_index = np.where(self.eps <= 0)[0]
            self.dLambda_index = np.argmin(np.abs(self.eps[filled_index] - self.eps_a - self.Lambda[filled_index]))
            # Check if the value of Delta is zero at this index
            if self.Delta[self.dLambda_index] == self.sp_contributions and self.eps[self.dLambda_index] <= 0:
                print(f'Only one root at {self.eps_d}')
                na_ = 1 / ( 1 - self.dLambda[self.dLambda_index] )
                na_ += integrated_rho
                self.lower_index_root = self.dLambda_index
                self.upper_index_root = self.dLambda_index
            else:
                # Root inside the d-band
                print(f'Root inside d-band at {self.eps_d}')
                na_ = integrated_rho 

        # Remove any numerical noise by setting na_ to be at most 1
        self.na = np.min([1, na_])

    def calculate_energy(self):
        self._create_Delta()
        self._create_Lambda()
        self.calculate_dos()

        # Create the arctan integrand
        numerator = self.Delta
        denominator = self.eps  - self.eps_a - self.Lambda
        assert all( numerator >= 0), "Numerator must be positive"

        # find where self.eps is lower than 0
        filled_eps_index = [i for i in range(len(self.eps)) if self.eps[i] <= 0]
        arctan_integrand = np.arctan2(numerator[filled_eps_index], denominator[filled_eps_index])
        arctan_integrand -= np.pi

        assert all(arctan_integrand <= 0), "Arctan integrand must be negative"
        assert all(arctan_integrand >= -np.pi), "Arctan integrand must be greater than -pi"

        # Integrate to get the energies
        delta_E_ = 2 / np.pi * integrate.simps( arctan_integrand , self.eps[filled_eps_index] )

        # Subtract the energy of the adsorbate
        delta_E_ -= 2 * self.eps_a

        if delta_E_ > 0:
            delta_E_ = 0

        # Store the energy 
        self.DeltaE = delta_E_
    
@dataclass
class NewnsAndersonAnalytical:
    """ Perform the Newns-Anderson model analytically for a semi-elliplical delta.
        Inputs 
        ------
        eps_a: float
            renormalised energy of the adsorbate in units of eV wrt Fermi level
        beta_p: float
            Coupling element of the adsorbate with the metal atom in units of 2beta 
        beta: float
            Coupling element of the metal with metal in units of eV
        eps_d: float
            center of Delta in units of eV wrt Fermi level
        eps: list
            Range of energies to plot in units of eV wrt d-band center 
        fermi_energy: float
            Fermi energy in the units of eV
        U: float
            Coulomb interaction parameter in units of eV
    """ 
    beta_p: float
    eps_a: float
    eps: list
    eps_d : float
    beta: float
    fermi_energy: float
    U: float
    grid_size = 20

    def __post_init__(self):
        """Setup the quantities for a self-consistent calculation."""
        # Setup spin polarised calculation with different up and down spins
        # To determine the lowest energy spin configuration determine the variation
        # n_-sigma with a fixed n_sigma and vice-versa. The point where the two
        # curves meet is the self-consistency point.
        self.nsigma_range = np.linspace(0, 1.0, self.grid_size)
        self.nmsigma_range = np.linspace(0, 1.0, self.grid_size)
        # Unit conversion details
        # coversion factor to divide by to convert to 2beta units
        self.convert = 2 * self.beta
        self.U = self.U / self.convert
        self.eps_a = self.eps_a / self.convert
        self.eps = self.eps / self.convert 
        self.eps_d = self.eps_d / self.convert
        self.fermi_energy = self.fermi_energy / self.convert
        # The quantities that will be of interest here 
        self.Delta = np.zeros(len(self.eps))
        self.Lambda = np.zeros(len(self.eps))
        self.rho_aa = np.zeros(len(self.eps))
        # Print out details of the quantities
        input_data = {
            "beta_p": self.beta_p,
            "eps_a": self.eps_a,
            "eps_d": self.eps_d,
            "beta": self.beta,
            "fermi_energy": self.fermi_energy,
            "U": self.U,
        }
        pprint(input_data)

    
    def self_consistent_calculation(self):
        """ Calculate the self-consistency point for the given parameters."""
        if self.U == 0:
            # There is no columb interaction, so the self-consistency point is
            print('No need for self-consistent calculation, U=0')
            self.n_minus_sigma = 1.
            self.n_plus_sigma = 1.
        else:
            # Find the lowest maximum value for varying n_down
            lowest_energy = None
            index_nup_overall = None
            index_ndown_overall = None
            # Store all the energies of the grid
            self.energies_grid = np.zeros((len(self.nsigma_range), len(self.nmsigma_range)))
            for j, n_down in enumerate(self.nmsigma_range):
                # Fix n_minus sigma and determine energies for different 
                # values of n_plus sigma
                energies_down = np.zeros(len(self.nsigma_range))
                for i, n_up in enumerate(self.nsigma_range):

                    # Define the energies for up and down spins
                    self.eps_sigma_up = self.eps_a + self.U * n_down 
                    self.eps_sigma_down = self.eps_a + self.U * n_up

                    # Calculate the 1electron energies
                    # First calculate the spin up energy
                    self.eps_sigma = self.eps_sigma_up
                    self.calculate_energies()
                    energies_down_ = self.DeltaE_1sigma
                    # Now calculate the spin down energy
                    self.eps_sigma = self.eps_sigma_down
                    self.calculate_energies()
                    energies_down_ += self.DeltaE_1sigma

                    # Calculate the coulomb contribution
                    coulomb_energy = self.U * n_down * n_up
                    energies_down_ -= coulomb_energy

                    # Subtract the adsorbate energy
                    energies_down_ -= self.eps_a 

                    # Store the energies to check if it is the lowest
                    energies_down[i] = energies_down_

                # determine the maximum value of the energy
                maximum_energy = np.max(energies_down)
                index_nup = np.argmax(energies_down)
                # Store the energies for the grid
                self.energies_grid[:, j] = energies_down
                
                # Choose whether to store this energy or not
                if lowest_energy is not None:
                    lowest_energy = maximum_energy if maximum_energy <= lowest_energy else lowest_energy
                    index_nup_overall = index_nup if maximum_energy <= lowest_energy else index_nup_overall
                    index_ndown_overall = j if maximum_energy <= lowest_energy else index_ndown_overall
                else:
                    # First run, store these quantities
                    lowest_energy = maximum_energy
                    index_nup_overall = index_nup
                    index_ndown_overall = j
            
            # Determine the values that give the lowest energy
            self.n_minus_sigma = self.nmsigma_range[index_ndown_overall]
            self.n_plus_sigma = self.nsigma_range[index_nup_overall]


        # Store all the quantities for the self-consistency point
        self.eps_sigma_up = self.eps_a + self.U * self.n_minus_sigma 
        self.eps_sigma_down = self.eps_a + self.U * self.n_plus_sigma

        # Sum up the energies from both of the spins
        self.eps_sigma = self.eps_sigma_up
        self.calculate_energies()
        self.rho_aa_up = self.rho_aa
        DeltaE_ = self.DeltaE_1sigma
        self.eps_sigma = self.eps_sigma_down
        self.calculate_energies()
        self.rho_aa_down = self.rho_aa
        DeltaE_ += self.DeltaE_1sigma

        # The variable rhoaa will be the sum of the two rhos
        self.rho_aa = self.rho_aa_up + self.rho_aa_down 

        # Coulomb contribution
        DeltaE_ -= self.U * self.n_minus_sigma * self.n_plus_sigma
        # Adsorbate energy difference
        DeltaE_ -= self.eps_a
        # Fermi energy
        DeltaE_ += self.fermi_energy

        # The converged energy
        self.DeltaE = DeltaE_ 

        # Print out the final results
        print('--------------------------')
        print(f"Spin up expectation value   : {self.n_minus_sigma} e")
        print(f"Spin down expectation value : {self.n_plus_sigma} e")
        print(f"Self-consistency energy     : {self.DeltaE} (2beta)")


    def calculate_energies(self):
        """Calculate the 1e energies from the Newns-Anderson model."""

        # Energies referenced to the d-band center
        # Needed for some manipulations later
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
                root_positive = 1 + 4 * self.eps_sigma_wrt_d**2
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

        # Store the root referenced to the energy reference scale that was chosen 
        self.root_positive = root_positive + self.eps_d #- self.fermi_energy
        self.root_negative = root_negative + self.eps_d #- self.fermi_energy
        
        # Determine if there is an occupied localised state
        if not self.has_complex_root:
            if self.root_positive < self.lower_band_edge and self.eps_sigma_wrt_d < 2 * self.beta_p**2 - 1:
                # Check if the root is below the Fermi level
                if self.root_positive < self.fermi_energy:
                    # the energy for this point is to be included
                    # print('Positive root is below the Fermi level.')
                    self.has_localised_occupied_state_positive = True
                else:
                    self.has_localised_occupied_state_positive = False
                # in both cases it is appropriate to store it as eps_l_sigma because it is localised state
                self.eps_l_sigma_pos = self.root_positive
            else:
                self.eps_l_sigma_pos = None
                self.has_localised_occupied_state_positive = False
            
            # Check if there is a localised occupied state for the negative root
            if self.root_negative > self.upper_band_edge and self.eps_sigma_wrt_d > 1 - 2 * self.beta_p**2:
                # Check if the root is below the Fermi level
                if self.root_negative < self.fermi_energy:
                    # the energy for this point is to be included
                    # print('Negative root is below the Fermi level.')
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
                    self.na_sigma_pos += 2 * self.beta_p**2 * self.eps_sigma_wrt_d * (4 * self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**-0.5 
                    self.na_sigma_pos /= (1 - 4 * self.beta_p**2)
                else:
                    self.na_sigma_pos = 4 * self.eps_sigma_wrt_d**2 - 1
                    self.na_sigma_pos /= (4 * self.eps_sigma_wrt_d**2)
            else:
                self.na_sigma_pos = 0.0
            
            if self.has_localised_occupied_state_negative:
                # Compute the expectancy value
                if self.beta_p != 0.5:
                    self.na_sigma_neg = (1 - 2 * self.beta_p**2)
                    self.na_sigma_neg -= 2 * self.beta_p**2 * self.eps_sigma_wrt_d * (4 * self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**-0.5 
                    self.na_sigma_neg /= (1 - 4 * self.beta_p**2)
                else:
                    self.na_sigma_neg = 4 * self.eps_sigma_wrt_d**2 - 1
                    self.na_sigma_neg /= (4 * self.eps_sigma_wrt_d**2)
            else:
                self.na_sigma_neg = 0.0
        else:
            # This is a complex root
            assert self.has_complex_root
            self.has_localised_occupied_state_positive = False
            self.has_localised_occupied_state_negative = False
            self.na_sigma_neg = 0
            self.na_sigma_pos = 0

        # ---------- Calculate the energy ----------
        # Determine the upper bounds for the contour integration
        if self.upper_band_edge > self.fermi_energy:
            upper_bound = self.fermi_energy
        else:
            upper_bound = self.upper_band_edge
        occupied_states = [i for i in range(len(self.eps)) if self.lower_band_edge < self.eps[i] < upper_bound]

        # Determine the integrand 
        energy_occ = self.eps_wrt_d[occupied_states]
        numerator = - 2 * self.beta_p**2 * (1 - energy_occ**2)**0.5
        numerator = np.nan_to_num(numerator)
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
            self.energy += self.eps_l_sigma_pos 
            self.energy -= self.eps_l_sigma_neg 
        elif self.has_localised_occupied_state_positive:
            # Has only positive root and it is a localised occupied state 
            arctan_integrand += np.pi
            self.arctan_component =  np.trapz( arctan_integrand, energy_occ )
            self.arctan_component /= np.pi
            self.energy = self.arctan_component
            self.energy += self.eps_l_sigma_pos
            self.energy -= self.fermi_energy
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

        # The one electron energy is just the difference of eigenvalues 
        self.DeltaE_1sigma = self.energy 
        # assert self.na_sigma_pos + self.na_sigma_neg <= 1.0
        # assert self.na_sigma_pos >= 0
        # assert self.na_sigma_neg >= 0
        # self.DeltaE_1sigma -= ( self.na_sigma_pos + self.na_sigma_neg ) * self.eps_sigma
        # self.DeltaE_1sigma -= self.eps_sigma
