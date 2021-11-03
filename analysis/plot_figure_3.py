"""Get parameters for the Newns-Anderson model and plot Figure 3 of the manuscript."""
import json
import yaml
import numpy as np
from dataclasses import dataclass
from NewnsAnderson import NewnsAndersonNumerical
from collections import defaultdict
from scipy.optimize import minimize, least_squares, leastsq, curve_fit
from pprint import pprint
import matplotlib.pyplot as plt
from plot_params import get_plot_params
from adjustText import adjust_text
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

@dataclass
class JensNewnsAnderson:
    """Class for fitting parameters for the Newns-Anderson and the
    d-band model with calculations performed with DFT. The class 
    expects all metal quantities for a particular eps_a (adsorbate) value."""
    Vsd: list
    filling: list
    width: list
    eps_a: float

    def __post_init__(self):
        """Extra variables that are needed for the model."""
        self.eps = np.linspace(-25, 20, 100000)

        # convert all lists to numpy arrays
        self.Vsd = np.array(self.Vsd)
        self.filling = np.array(self.filling)
        self.width = np.array(self.width)

    def fit_parameters(self, eps_ds, alpha, beta, constant):
        """Fit the parameters alpha, beta, delta0."""
        # All the parameters here will have positive values
        # Vak assumed to be proportional to Vsd
        Vak = np.sqrt(beta) * self.Vsd

        # Store the hybridisation energy for all metals to compare later
        hybridisation_energy = np.zeros(len(eps_ds))

        # We will need the occupancy of the single particle state
        na = np.zeros(len(eps_ds))
        # Loop over all the metals
        for i, eps_d in enumerate(eps_ds):
            hybridisation = NewnsAndersonNumerical(
                Vak = Vak[i],
                eps_a = self.eps_a,
                eps_d = eps_d,
                width = self.width[i],
                eps = self.eps,
                k = constant,
            )
            # The first component of the hybridisation energy
            # is the hybdridisation coming from the sp and d bands
            hybridisation.calculate_energy()
            hybridisation_energy[i] = hybridisation.DeltaE
            # Store the occupancies of each state
            na[i] = hybridisation.na

        # Ensure that the hybridisation energy is negative always
        assert all(hybridisation_energy <= 0), "Hybridisation energy is negative"

        # orthonogonalisation energy
        ortho_energy = 2 * ( na +  self.filling ) * alpha * np.sqrt(beta) * self.Vsd**2
        ortho_energy = np.array(ortho_energy)

        # Ensure that the orthonogonalisation energy is positive always
        assert all(ortho_energy >= 0), "Orthogonalisation energy is positive"

        # Add the orthogonalisation energy to the hybridisation energy
        hybridisation_energy += ortho_energy

        # Store the hybridisation energy for all metals
        self.hybridisation_energy = hybridisation_energy
        # Store the occupancies as well
        self.na = na
        
        return hybridisation_energy


if __name__ == '__main__':
    """Determine the fitting parameters for a particular adsorbate."""

    REMOVE_LIST = []
    KEEP_LIST = []

    # Choose a sequence of adsorbates
    ADSORBATES = ['C', 'O']
    EPS_A_VALUES = [ -1, -5 ] # eV
    print(f"Fitting parameters for adsorbate {ADSORBATES} with eps_a {EPS_A_VALUES}")

    # The functional and type of calculation we will use
    # scf only calculations in order to avoid any noise and look only for 
    # the electronic structure contribution
    FUNCTIONAL = 'PBE_scf'

    # get the width and d-band centre parameters
    # The moments of the density of states comes from a DFT calculation 
    # and the adsorption energy is from scf calculations of the adsorbate
    # at a fixed distance from the surface.
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{FUNCTIONAL}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{FUNCTIONAL}.json'))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))


    # Plot the Fitted and the real adsorption energies
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for i in range(len(ax)):
        ax[i].set_xlabel('DFT energy (eV)')
        ax[i].set_ylabel('Hybridisation energy (eV)')
        ax[i].set_title(f'{ADSORBATES[i]}* with $\epsilon_a=$ {EPS_A_VALUES[i]} eV')


    # simulatenously iterate over ADSORBATES and EPS_A_VALUES
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Fitting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
        # Store the parameters in order of metals in this list
        parameters = defaultdict(list)
        # Store the final DFT energies
        dft_energies = []
        metals = []

        for metal in data_from_energy_calculation[adsorbate]:
            if KEEP_LIST:
                if metal not in KEEP_LIST:
                    continue
            if REMOVE_LIST:
                if metal in REMOVE_LIST:
                    continue

            # get the parameters from DFT calculations
            width = data_from_dos_calculation[metal]['width']
            parameters['width'].append(width)
            d_band_centre = data_from_dos_calculation[metal]['d_band_centre']
            parameters['d_band_centre'].append(d_band_centre)

            # get the parameters from the energy calculations
            adsorption_energy = data_from_energy_calculation[adsorbate][metal]
            dft_energies.append(adsorption_energy)

            # get the idealised parameters 
            Vsd = np.sqrt(data_from_LMTO['Vsdsq'][metal])
            parameters['Vsd'].append(Vsd)

            # Get the metal filling
            filling = data_from_LMTO['filling'][metal]
            parameters['filling'].append(filling)

            # Store the order of the metals
            metals.append(metal)

        # Fit the parameters
        fitting_function = JensNewnsAnderson(
            Vsd = parameters['Vsd'],
            filling = parameters['filling'],
            width = parameters['width'],
            eps_a = eps_a,
        )

        # Make the constrains for curve_fit such that all the 
        # terms are positive
        constraints = [ [0, 0, 0], [np.inf, np.inf, np.inf] ]
        initial_guess = [0.05, 0.15, 2]
        popt, pcov = curve_fit(f=fitting_function.fit_parameters,
                            xdata=parameters['d_band_centre'],
                            ydata=dft_energies,
                            p0=initial_guess,
                            bounds=constraints)  

        error_fit = np.sqrt(np.diag(pcov))
        print(f'Fit: alpha: {popt[0]}, beta: {popt[1]}, constant:{popt[2]} ')
        print(f'Error: alpha:{error_fit[0]}, beta: {error_fit[1]}, constant:{error_fit[2]} ')

        # Get the final hybridisation energy
        optimised_hyb = fitting_function.fit_parameters(parameters['d_band_centre'], *popt)
        occupancies_final = np.array(fitting_function.na)[np.argsort(parameters['d_band_centre'])]
        print(f'Occupancies: {occupancies_final}', file=open(f'output/{adsorbate}_occupancies.txt', 'w'))
        print(f"d-band center: {np.sort(parameters['d_band_centre'])}", file=open(f'output/{adsorbate}_occupancies.txt', 'a'))

        # get the error in the hybridisation energy
        positive_optimised_hyb_error = fitting_function.fit_parameters(parameters['d_band_centre'], *(popt + error_fit/2))
        negative_error = optimised_hyb - positive_optimised_hyb_error
        negative_optimised_hyb_error = optimised_hyb + negative_error
        # plot the parity line
        x = np.linspace(np.min(dft_energies)-0.25, np.max(dft_energies)+0.25, 2)
        ax[i].plot(x, x, '--', color='black')
        # Fix the axes to the same scale 
        ax[i].set_xlim(np.min(x), np.max(x))
        ax[i].set_ylim(np.min(x), np.max(x))

        texts = []
        for j, metal in enumerate(metals):
            # Choose the colour based on the row of the TM
            if metal in FIRST_ROW:
                colour = 'red'
            elif metal in SECOND_ROW:
                colour = 'orange'
            elif metal in THIRD_ROW:
                colour = 'green'
            ax[i].plot(dft_energies[j], optimised_hyb[j], 'o', color=colour)
            # Plot the error bars
            ax[i].plot([dft_energies[j], dft_energies[j]], [negative_optimised_hyb_error[j], \
                        positive_optimised_hyb_error[j]], '-', color=colour, alpha=0.25)

            texts.append(ax[i].text(dft_energies[j], optimised_hyb[j], metal, color=colour))

        adjust_text(texts, ax=ax[i]) 

        # Write out the fitted parameters as a json file
        json.dump({
            'alpha': popt[0],
            'beta': popt[1],
            'constant': popt[2],
        }, open(f'output/{adsorbate}_parameters_{FUNCTIONAL}.json', 'w'))

    fig.savefig(f'output/figure_3.png', dpi=300)