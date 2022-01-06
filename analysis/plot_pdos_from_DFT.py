"""Plot the projected density of states from the DFT calculations."""
import sys
import json
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
from ase.dft import get_distribution_moment
import numpy as np
from scipy.integrate import simps
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

def semi_ellipse(energies, eps_d, width, amp):
    energy_ref = ( energies - eps_d ) / width
    delta = np.zeros(len(energies))
    for i, eps_ in enumerate(energy_ref):
        if np.abs(eps_) < 1:
            delta[i] = amp * (1 - eps_**2)**0.5
    return delta

def normalise_na_quantities(quantity):
    """Utility function to align the density of states for Newns-Anderson plots."""
    return quantity / np.max(np.abs(quantity))

if __name__ == "__main__":
    """Plot the pdos for the metal and of the adsorbates 
    from a DFT calculation and compare the result with 
    the fitted Newns-Anderson Delta."""
    # Choice of functional
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))['group'][0]
    # Remove the following metals
    REMOVE_LIST = ['X', 'Al', 'Mg'] # yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']

    with open(f'output/pdos_{COMP_SETUP}.json', 'r') as handle:
        pdos_data = json.load(handle)

    METALS = [FIRST_ROW, SECOND_ROW, THIRD_ROW]

    fig, ax = plt.subplots(len(METALS), len(METALS[0]), 
                           figsize=(16,12), sharey=True,
                           constrained_layout=True)

    # Store the moments to plot later
    moments = defaultdict(dict)

    # Keep track of which elements are computed
    used_ij = []

    for metal in pdos_data['slab']:
        # Do not plot metals in the remove list
        if metal in REMOVE_LIST:
            continue
        # Get all the pdos
        energies, pdos, _ = pdos_data['slab'][metal]
        energies_c, pdos_c = pdos_data['C'][metal]
        energies_o, pdos_o = pdos_data['O'][metal]

        # Normalise the quantities
        pdos_c = normalise_na_quantities(pdos_c)
        pdos_o = normalise_na_quantities(pdos_o)
        pdos = normalise_na_quantities(pdos)

        # in case the maximum value of energies is
        # less than +5 eV, pad the pdos with 0's
        if energies[-1] < 5:
            assert energies[0] < energies[-1]
            energy_spacing = energies[1] - energies[0]
            extra_zeros = np.arange(energies[-1], 5, energy_spacing)
            if len(extra_zeros) > 0:
                energies = np.append(energies, extra_zeros)
                pdos = np.append(pdos, np.zeros(len(extra_zeros)))
        assert len(energies) == len(pdos)

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
            print(metal)
            raise ValueError('Metal not in chosen list of metals.')

        # If we have gotten so far, there is data to plot
        # the metal.
        used_ij.append((i,j))

        # make pdos and energies into numpy arrays
        energies = np.array(energies)

        ax[i,j].plot(pdos, energies, color='tab:red')
        ax[i,j].fill_between(pdos, energies, color='tab:red', alpha=0.2)
        ax[i,j].set_title(metal)
        ax[i,j].set_xticks([])
        ax[i,j].set_ylim(-10, 5)

        # get the d-band center and the width
        # Fit a semi-ellipse to the data by passing in the 
        # moments as an initial guess to the curve fitting procedure
        center, second_moment = get_distribution_moment(energies, pdos, (1, 2)) 
        popt, pcov = curve_fit(semi_ellipse, energies, pdos, p0=[center, 4*np.sqrt(second_moment), 1])
        # The center is just the first element of the popt array
        center = popt[0]
        # Store the width as the 2*width from the fitting procedure
        width =  2 * popt[1] 
        # Also store the upper edge of the d-band centre
        # by determining the maximum of the Hilbert transform
        Lambda = np.imag(signal.hilbert(pdos))
        index_max = np.argmax(Lambda)
        # Store the moments
        moments[metal]['d_band_centre'] = center
        moments[metal]['width'] = width
        moments[metal]['d_band_upper_edge'] = energies[index_max]

        # Plot the density of states from the Newns-Anderson model 
        hybridisation = NewnsAndersonNumerical(
            Vak = 1, 
            eps_a = -1, 
            eps_d = center,
            width = width,
            eps = np.linspace(-20, 20, 1000),
            Delta0_mag = 0.0,
        )
        hybridisation.calculate_energy()
        hybridisation.calculate_occupancy()

        # Get the Newns-Anderson Delta
        Delta_na = hybridisation.get_Delta_on_grid()
        energy_na = hybridisation.eps

        # Normalise to plot on the same graph
        Delta_na = normalise_na_quantities(Delta_na)

        # Get the filling of Delta
        filling_dband = hybridisation.get_dband_filling()

        # Annotate the filling_dband
        ax[i,j].annotate(f'f={filling_dband:.2f}',
                            xy=(0.1, 0.2),
                            xycoords='axes fraction',
                            fontsize=14,
        )

        # Plot the Newns-Anderson Delta
        ax[i,j].plot(Delta_na, energy_na, color='tab:blue', ls='--')

        # ax[i,j].axhline(y=center, color='k', linestyle='-')
        # ax[i,j].axhline(y=center + width / 2, color='k', linestyle='--')
        # ax[i,j].axhline(y=center - width / 2, color='k', linestyle='--')
        ax[i,j].axhline(y=energies[index_max], color='k', linestyle='--')
    
    for i in range(len(METALS)):
        for j in range(len(METALS[i])):
            if (i,j) not in used_ij:
                ax[i,j].axis('off')

    fig.savefig(f'output/pdos_{COMP_SETUP}.png')

    pprint(moments)

    with open(f'output/pdos_moments_{COMP_SETUP}.json', 'w') as handle:
        json.dump(moments, handle, indent=4)