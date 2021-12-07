"""Plot figure 2 of the manuscipt."""
import string
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.integrate import simps
from scipy.linalg.misc import norm
from plot_params import get_plot_params
import matplotlib.ticker as ticker
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical
import yaml
get_plot_params()

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 

def moment_generator(energies, all_dos, moment):
    """Get the moment of a distribution from the density of states"""
    def integrand_numerator(e, rho):
        return e * rho
    def integrand_denominator(e,rho):
        return rho
    def integrand_moment_numerator(e, epsilon_d, moment, rho):
        return ( (e - epsilon_d ) ** moment ) * rho
    moments = []
    eps_d = []
    energies = energies
    for alldos in all_dos:
        dos = np.array(alldos)
        epsilon_d_num = simps(integrand_numerator(energies,dos), energies)
        epsilon_d_den = simps(integrand_denominator(energies,dos), energies)
        epsilon_d = epsilon_d_num / epsilon_d_den
        moment_numerator = simps( integrand_moment_numerator(energies, epsilon_d, moment, \
                dos), energies)
        moment_denom = epsilon_d_den
        moment_spin = moment_numerator / moment_denom
        moments.append(moment_spin)
        eps_d.append(epsilon_d)
    return moments, eps_d

def normalise_na_quantities(quantity, x_add):
    """Utility function to align the density of states for Newns-Anderson plots."""
    return quantity + x_add

def get_plot_layout():
    #-------- Plot parameters --------#
    fig = plt.figure(figsize=(14,12), constrained_layout=True)
    gs = fig.add_gridspec(nrows=11, ncols=3,)

    # Newns-Anderson dos plot
    ax1 = fig.add_subplot(gs[0:4,:])
    # ax1.set_xlabel(r'$\Delta, n_a$ (eV)')
    ax1.set_title('Newns-Anderson Density of States')
    ax1.set_ylabel(r'$\epsilon - \epsilon_f$ (eV)')
    ax1.set_xticks([])
    ax1.set_ylim([-10,5])

    # pdos plots
    axp = []
    for j in range(3):
        axp.append(fig.add_subplot(gs[4:,j]))
    axp = np.array(axp)
    axp = axp.reshape(1, 3)
    axp[0,0].set_ylabel('Project Density of States')
    axp[0,0].set_xlabel('$\epsilon - \epsilon_{F}$ (eV)')
    axp[0,1].set_xlabel('$\epsilon - \epsilon_{F}$ (eV)')
    axp[0,2].set_xlabel('$\epsilon - \epsilon_{F}$ (eV)')
    for i in range(3):
        axp[0,i].set_xlim([-30,5])
        axp[0,i].set_yticks([])
    ax1.plot([], [], '-', color='tab:red', label='O*')
    ax1.plot([], [], '-', color='k', label='C*')
    ax1.legend(loc='best')
    return fig, ax1, axp

if __name__ == "__main__":
    """Generate figures with all the parameters for the
    Newns-Anderson model."""
    REMOVE_LIST = [] # yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    FUNCTIONAL = 'PBE_scf_smeared'

    # Get a cycle of with colormap
    colors =  plt.cm.viridis(np.linspace(0, 1, 10))

    # Get the plot parameters
    fig, ax1, axp = get_plot_layout()

    # Plot the Newns-Anderson DOS for a few d-band centres
    newns_epsds = [ -4, -3, -2, -1 ]
    newns_epsas = [-5, -1]
    x_add = 0.0
    for i, newns_epsd in enumerate(newns_epsds):
        for j, newns_epsa in enumerate(newns_epsas):

            hybridisation = NewnsAndersonNumerical(
                Vak = 1, 
                eps_a = newns_epsa, 
                eps_d = newns_epsd,
                width = 3,
                eps = np.linspace(-40, 40, 1000),
                Delta0_mag = 2,
                eps_sp_max = 15,
                eps_sp_min = -15,
            )
            hybridisation.calculate_energy()
            hybridisation.calculate_occupancy()
            
            # Decide on the x-position based on the d-band centre
            if j == 0:
                color='tab:red'
            elif j == 1:
                color='k'
            
            # Get the metal projected density of states
            Delta = normalise_na_quantities( hybridisation.get_Delta_on_grid(), x_add )
            x_add += 2. * np.max(Delta) 
            Lambda = normalise_na_quantities( hybridisation.get_Lambda_on_grid(), x_add )
            # Get the line representing the eps - eps_a state
            eps_a_line = normalise_na_quantities( hybridisation.get_energy_diff_on_grid(), x_add )
            # Get the adsorbate density of states
            na = normalise_na_quantities( hybridisation.get_dos_on_grid(), x_add) 
            # Plot the dos and the quantities that make the dos            
            ax1.plot(Delta, hybridisation.eps, color='tab:blue', lw=3)
            ax1.plot(na, hybridisation.eps, color=color)
            occupied_energies = np.where(hybridisation.eps <= 0)[0] 
            ax1.fill_betweenx(hybridisation.eps[occupied_energies],  x_add, na[occupied_energies], color=color, alpha=0.25)
            # ax1.plot(Lambda, hybridisation.eps, color='tab:orange', lw=3)
            # ax1.plot(eps_a_line, hybridisation.eps, color='tab:green', lw=3)

            if j == 0:
                ax1.annotate(r'$\epsilon_{d} = %.1f$ eV' % newns_epsd, xy=(x_add+0.1, newns_epsd + 1), xytext=(x_add+0.4, 3),
                            xycoords='data', textcoords='data',
                            color='tab:blue',
                            arrowprops=dict(arrowstyle="->",
                                            connectionstyle="arc3,rad=0.2",
                                            color='tab:blue'),
                            fontsize=15)

    #-------- Read in the DFT data --------#
    # Load the Vsd and filling data
    with open("inputs/data_from_LMTO.json", 'r') as handle:
        data_from_LMTO = json.load(handle)
    Vsd_data = data_from_LMTO["Vsdsq"]
    filling_data = data_from_LMTO["filling"]
    # Load the pdos data
    pdos_data = json.load(open(f'output/pdos_{FUNCTIONAL}.json'))

    x_add = 0
    #-------- Plot Figure 1 --------#
    for row_index, row_metals in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        for i, element in enumerate(row_metals):
            # Get the data for the element
            if element == 'X':
                continue
            if element in REMOVE_LIST:
                continue
            try:
                energies, pdos_metal_d, pdos_metal_sp = pdos_data['slab'][element]
                energies_C, pdos_C = pdos_data['C'][element]
                energies_O, pdos_O = pdos_data['O'][element]
            except KeyError:
                continue
            pdos_C = np.array(pdos_C)
            pdos_O = np.array(pdos_O)

            x_add += np.max(pdos_metal_d)
            pdos_C *= np.max(pdos_metal_d) / np.max(pdos_C)
            pdos_O *= np.max(pdos_metal_d) / np.max(pdos_O)
            pdos_metal_d = normalise_na_quantities(pdos_metal_d, x_add)
            pdos_metal_sp = normalise_na_quantities(pdos_metal_sp, x_add)
            pdos_C = normalise_na_quantities(pdos_C, x_add)
            pdos_O = normalise_na_quantities(pdos_O, x_add)
            # Set the maximum of the C, O pdos to the maximum of the metal pdos

            # Plot the pdos onto the metal states
            axp[0,row_index].fill_between(energies, x_add, pdos_metal_d, color=colors[i], alpha=0.5) 
            # axp[0,row_index].plot(energies, pdos_metal_sp, color=colors[i], ls='--')
            axp[0,row_index].annotate(element, xy=(-8.5, pdos_metal_d[-1]+0.5), color=colors[i])
            # Plot the C (sp) pdos
            axp[0,row_index].plot(energies_C, pdos_C, color='k', alpha=0.75)
            # Plot the O (sp) pdos
            axp[0,row_index].plot(energies_O, pdos_O, color='tab:red', alpha=0.75)

            # Plot all the quantities that will be useful in the model.
            if row_index == 0:
                color_row ='tab:blue'
            elif row_index == 1:
                color_row ='tab:orange'
            elif row_index == 2:
                color_row ='tab:green'

    # Add figure numbers
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate([ax1] + list(axp.flatten())):
        if i in [1, 2, 3]:
            a.annotate(alphabet[i]+')', xy=(0.01, 0.95), xycoords='axes fraction')
        else:
            a.annotate(alphabet[i]+')', xy=(0.01, 0.85), xycoords='axes fraction')


    # Save the figure
    fig.savefig(f'output/figure_2_{FUNCTIONAL}.png', dpi=300)