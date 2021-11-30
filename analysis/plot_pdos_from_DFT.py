"""Plot the projected density of states from the DFT calculations."""
import sys
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from ase.dft import get_distribution_moment
import numpy as np
from scipy.integrate import simps
from scipy import odr
from scipy import signal
from scipy.optimize import curve_fit
from pprint import pprint
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical
from plot_params import get_plot_params
get_plot_params()

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 
METALS = [FIRST_ROW, SECOND_ROW, THIRD_ROW]

def semi_ellipse(energies, eps_d, width, amp):
    energy_ref = ( energies - eps_d ) / width
    delta = np.zeros(len(energies))
    for i, eps_ in enumerate(energy_ref):
        if np.abs(eps_) < 1:
            delta[i] = amp * (1 - eps_**2)**0.5
    return delta

def normalise_na_quantities(quantity, energies):
    """Make sure that the quantity provided integrates to 1."""
    return quantity / simps(quantity, x=energies)

class FittingVak:
    """Fit Vak based on the features of the (sp) projected
    density of states of the adsorbates."""
    def __init__(self, eps_a, eps_d, width):
        self.eps_a = eps_a
        self.eps_d = eps_d
        self.width = width
        self.Delta0_mag = 0.0
    
    def fitting_function(self, parameters, eps):
        """Fit the pdos from the Newns-Anderson model
        to that from the (sp) projected density of states
        coming from a DFT calculation."""
        Vak = parameters
        # Vak must be positive
        Vak = abs(Vak)
        # Delta0_mag = abs(Delta0_mag)
        hybridisation = NewnsAndersonNumerical(
            Vak = Vak, 
            eps_a = self.eps_a, 
            eps_d = self.eps_d,
            width = self.width,
            eps_sp_max=20,
            eps_sp_min=-20,
            eps = eps, 
            Delta0_mag = self.Delta0_mag,
            verbose=False,
        )
        hybridisation.calculate_energy()
        hybridisation.calculate_occupancy()

        if len(hybridisation.poles) > 0:
            # Store the poles
            self.poles = hybridisation.poles
        else:
            self.poles = []

        return hybridisation.get_dos_on_grid()

if __name__ == "__main__":
    """Plot the pdos for the metal and of the adsorbates 
    from a DFT calculation and compare the result with 
    the fitted Newns-Anderson Delta."""
    # Choice of functional
    FUNCTIONAL = 'PBE_scf' 
    
    # Adsorbates to plot 
    adsorbates = {'O':-5, 'C':-1}

    # Parameters to generate initial guesses
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    # Get projected density of states from a DFT calculation
    with open(f'output/pdos_{FUNCTIONAL}.json', 'r') as handle:
        pdos_data = json.load(handle)

    # Plot three figures, one for the metal density of
    # states compared with Delta, second for the C (sp)
    # states and third for the O (sp) states.
    fig, ax = plt.subplots(len(METALS), len(METALS[0]), 
                           figsize=(16,12), sharey=True,
                           constrained_layout=True)
    figc, axc = plt.subplots(len(METALS), len(METALS[0]), 
                           figsize=(16,12), sharey=True,
                           constrained_layout=True)
    figo, axo = plt.subplots(len(METALS), len(METALS[0]), 
                           figsize=(16,12), sharey=True,
                           constrained_layout=True)

    # Store the moments to plot later
    moments = defaultdict(dict)
    # Store the fitted Vak values to use later
    filling_data = defaultdict(dict)

    # Keep track of which elements are computed
    used_ij = []

    for metal in pdos_data['slab']:
        # Get all the pdos
        energies, pdos, pdos_sp = pdos_data['slab'][metal]
        energies_c, pdos_c = pdos_data['C'][metal]
        energies_o, pdos_o = pdos_data['O'][metal]

        # Normalise the quantities to get a better
        # numerical fit
        pdos_c = normalise_na_quantities(pdos_c, energies_c)
        pdos_o = normalise_na_quantities(pdos_o, energies_o)

        # make pdos and energies into numpy arrays
        energies = np.array(energies)

        # in case the maximum value of energies is
        # less than +5 eV, pad the pdos with 0's
        if energies[-1] < 5:
            assert energies[0] < energies[-1]
            energy_spacing = energies[1] - energies[0]
            extra_zeros = np.arange(energies[-1], 5, energy_spacing)
            if len(extra_zeros) > 0:
                energies = np.append(energies, extra_zeros)
                energies_o = np.append(energies_o, extra_zeros)
                energies_c = np.append(energies_c, extra_zeros)
                pdos = np.append(pdos, np.zeros(len(extra_zeros)))
                pdos_c = np.append(pdos_c, np.zeros(len(extra_zeros)))
                pdos_o = np.append(pdos_o, np.zeros(len(extra_zeros)))

        # Decide on the index based on the metal
        if metal in METALS[0]:
            i = 0
            j = METALS[0].index(metal)
        elif metal in METALS[1]:
            i = 1
            j = METALS[1].index(metal)
        elif metal in METALS[2]:
            i = 2
            j = METALS[2].index(metal)
        else:
            raise ValueError('Metal not in chosen list of metals.')

        # If we have gotten so far, there is data to plot
        # the metal.
        used_ij.append((i,j))

        # Plot the dft projected density of states
        ax[i,j].plot(pdos, energies, color='tab:red')
        axo[i,j].plot(pdos_o, energies_o, color='tab:blue')
        # axo[i,j].plot(pdos, energies, color='tab:red')
        axc[i,j].plot(pdos_c, energies_c, color='tab:blue')
        # axc[i,j].plot(pdos, energies, color='tab:red')
        ax[i,j].fill_between(pdos, energies, color='tab:red', alpha=0.2)
        axo[i,j].fill_between(pdos_o, energies_o, color='tab:blue', alpha=0.2)
        axc[i,j].fill_between(pdos_c, energies_c, color='tab:blue', alpha=0.2)
        ax[i,j].set_title(metal)
        axo[i,j].set_title(metal)
        axc[i,j].set_title(metal)
        ax[i,j].set_xticks([])
        axo[i,j].set_xticks([])
        axc[i,j].set_xticks([])
        ax[i,j].set_ylim(-10, 5)
        axo[i,j].set_ylim(-10, 5)
        axc[i,j].set_ylim(-10, 5)

        if metal == 'Pt':
            # TODO: Something with the BZ-integration?
            # Get the index for the second largest peak
            index_plot = np.argpartition(pdos, -2)[-2]
            ax[i,j].set_xlim([None, pdos[index_plot]])

        # get the d-band center and the width
        # Fit a semi-ellipse to the data by passing in the 
        # moments as an initial guess to the curve fitting procedure
        center, second_moment = get_distribution_moment(energies, pdos, (1, 2)) 
        popt, pcov = curve_fit(semi_ellipse, energies, pdos, p0=[center, 4*np.sqrt(second_moment), 1])
        # The center is just the first element of the popt array
        eps_d = popt[0]
        # Store the width as the 2*width from the fitting procedure
        width =  2 * popt[1] 
        # Store the moments
        moments[metal]['d_band_centre'] = eps_d
        moments[metal]['width'] = width

        # Get the filling of the pdos of C, O atoms
        energies_o = np.array(energies_o)
        energies_c = np.array(energies_c)
        pdos_c = np.array(pdos_c)
        pdos_o = np.array(pdos_o)
        # index_o_filled = np.argwhere(energies_o < 0).flatten()
        # index_c_filled = np.argwhere(energies_c < 0).flatten()
        # Find the index where energies are between eps_d + width/2
        # and eps_d - width/2
        index_o_filled = np.argwhere((energies_o < eps_d + width/2) & (energies_o > eps_d - width/2)).flatten()
        index_c_filled = np.argwhere((energies_c < eps_d + width/2) & (energies_c > eps_d - width/2)).flatten()

        # Get the filling of the pdos of the metal
        f_c = np.trapz(pdos_c[index_c_filled], energies_c[index_c_filled]) / np.trapz(pdos_c, energies_c)
        f_o = np.trapz(pdos_o[index_o_filled], energies_o[index_o_filled]) / np.trapz(pdos_o, energies_o)

        filling_data['C'][metal] = f_c
        filling_data['O'][metal] = f_o

        axc[i,j].annotate(f'f={f_c:1.2f}', xy=(0.2, 0.2), xycoords='axes fraction')
        axo[i,j].annotate(f'f={f_o:1.2f}', xy=(0.2, 0.2), xycoords='axes fraction')

        # Fit the Vak for the different adsorbates
        # for adsorbate, eps_a in adsorbates.items():
        #     # Get adsorbate speicifc fitted quantities
        #     if adsorbate == 'C':
        #         eps = energies_c
        #         pdos_dft = pdos_c
        #         ax_p = axc[i,j]
        #     elif adsorbate == 'O':
        #         eps = energies_o
        #         pdos_dft = pdos_o
        #         ax_p = axo[i,j]
        #     # Get the fitted Vak function
        #     fit_Vak = FittingVak(eps_a, eps_d, width)
        #     # Get the initial guesses
        #     Vsd = 0.1 * np.sqrt(data_from_LMTO['Vsdsq'][metal])
        #     initial_guess = [Vsd] + [2]
        #     # Fit the data
        #     data = odr.Data(eps, pdos_dft)
        #     fitting_model = odr.Model(fit_Vak.fitting_function)
        #     fitting_odr = odr.ODR(data, fitting_model, initial_guess)
        #     fitting_odr.set_job(fit_type=2)
        #     output = fitting_odr.run()
        #     print(f'Metal: {metal} with fitted Vak: {output.beta[0]} and fitted Delta0: {output.beta[1]}')

        #     # Plot the fitted result
        #     ax_p.plot(fit_Vak.fitting_function(output.beta, eps), eps, color='tab:green')

        #     # Plot the poles if they exist
        #     if len(fit_Vak.poles) > 0:
        #         ax_p.plot(np.zeros(len(fit_Vak.poles)), fit_Vak.poles, '*', color='tab:green')
            
        #     # Store the fitted values
        #     fitted_Vak[adsorbate][metal] = np.abs(output.beta[0])
        #     # fitted_Delta0[adsorbate][metal] = output.beta[1]

        ax[i,j].axhline(y=center, color='k', linestyle='-')
        ax[i,j].axhline(y=center + width / 2, color='k', linestyle='--')
        ax[i,j].axhline(y=center - width / 2, color='k', linestyle='--')
        axc[i,j].axhline(y=center + width / 2, color='k', linestyle='--')
        axc[i,j].axhline(y=center - width / 2, color='k', linestyle='--')
        axo[i,j].axhline(y=center + width / 2, color='k', linestyle='--')
        axo[i,j].axhline(y=center - width / 2, color='k', linestyle='--')

    # Delete unused axes 
    for i in range(len(METALS)):
        for j in range(len(METALS[i])):
            if (i,j) not in used_ij:
                ax[i,j].axis('off')
                axc[i,j].axis('off')
                axo[i,j].axis('off')

    fig.savefig(f'output/pdos_{FUNCTIONAL}.png')
    figc.savefig(f'output/pdos_c_{FUNCTIONAL}.png')
    figo.savefig(f'output/pdos_o_{FUNCTIONAL}.png')

    pprint(moments)

    with open(f'output/pdos_moments_{FUNCTIONAL}.json', 'w') as handle:
        json.dump(moments, handle, indent=4)
    with open(f'output/filling_data_{FUNCTIONAL}.json', 'w') as handle:
        json.dump(filling_data, handle, indent=4)