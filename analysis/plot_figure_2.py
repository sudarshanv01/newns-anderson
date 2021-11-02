"""Plot figure 2 of the manuscipt."""
import string
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.integrate import simps
from plot_params import get_plot_params
import matplotlib.ticker as ticker
from NewnsAnderson import NewnsAndersonNumerical
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

def normalise_pdos(pdos):
    """Normalise the PDOS to 1 for plotting purposes."""
    return np.array(pdos) / np.max(pdos)

if __name__ == "__main__":
    """Generate all plots for Figure 1 of the manuscript."""
    #-------- Plot parameters --------#
    fig = plt.figure(figsize=(14,12), constrained_layout=True)
    gs = fig.add_gridspec(nrows=12, ncols=4,)

    # Newns-Anderson dos plot
    ax1 = fig.add_subplot(gs[0:3,:])
    ax1.set_xlabel(r'$\Delta, n_a$ (eV)')
    ax1.set_ylabel(r'$\epsilon - \epsilon_f$ (eV)')
    ax1.set_xticks([])

    # pdos plots
    axp = []
    for j in range(3):
        axp.append(fig.add_subplot(gs[3:,j]))
    axp = np.array(axp)
    axp = axp.reshape(1, 3)
    axp[0,0].set_ylabel('Project Density of States')
    axp[0,1].set_xlabel('Energy (eV)')
    for i in range(3):
        axp[0,i].set_xlim([-10,5])
        axp[0,i].set_yticks([])
        if i == 0:
            # Show adsorbate information
            axp[0,i].plot([], [], '-', color='k', label='O*')
            axp[0,i].plot([], [], '-', color='tab:red', label='C*')
            axp[0,i].legend(loc='best')
    ax2 = fig.add_subplot(gs[3:6,-1])
    ax2.set_ylabel(r'$\epsilon_{d}$ (eV)')
    ax2.xaxis.set_ticks(np.arange(0, 1.1, 0.2))

    # width of the d-band plot
    ax3 = fig.add_subplot(gs[6:9,-1])
    ax3.set_ylabel(r'$w_d$ (eV)')
    ax3.xaxis.set_ticks(np.arange(0, 1.1, 0.2))

    # Vsd plot
    ax4 = fig.add_subplot(gs[9:,-1])
    ax4.set_ylabel(r'$V_{sd}^2$ (eV)')
    ax4.set_xlabel('Filling of $d$-band')
    ax4.xaxis.set_ticks(np.arange(0, 1.1, 0.2))
    # Plot the legend for the all the parameter variation subplots
    ax4.plot([], [], 'o', color='tab:blue', label=r'$3d$')
    ax4.plot([], [], 'o', color='tab:orange', label=r'$4d$')
    ax4.plot([], [], 'o', color='tab:green', label=r'$5d$')
    ax4.legend(loc='best')

    # Get a cycle of with colormap
    colors =  plt.cm.viridis(np.linspace(0, 1, 10))

    # Plot the Newns-Anderson DOS for a few d-band centres
    newns_epsds = [ -4, -2, -1, 0 ]
    newns_epsas = [-5, -1]
    for i, newns_epsd in enumerate(newns_epsds):
        for j, newns_epsa in enumerate(newns_epsas):

            hybridisation = NewnsAndersonNumerical(
                Vak = 1, 
                eps_a = newns_epsa, 
                eps_d = newns_epsd,
                width = 3,
                eps = np.linspace(-10, 6, 100000),
                k = 0,
            )
            hybridisation.calculate_energy()
            
            # Decide on the x-position based on the d-band centre
            x_pos = 3 * i 
            if j == 0:
                color='tab:red'
            elif j == 1:
                color='k'
            
            # Get the metal projected density of states
            Delta = hybridisation.Delta
            Delta /= np.pi**2
            Delta += x_pos
            # Get the hilbert transform
            Lambda = hybridisation.Lambda
            Lambda /= np.pi**2
            Lambda += x_pos
            # Get the line representing the eps - eps_a state
            eps_a_line = hybridisation.eps - hybridisation.eps_a
            eps_a_line /= np.pi**2
            eps_a_line += x_pos
            # Get the adsorbate density of states
            na = hybridisation.dos + x_pos 
            na_localised_states = np.zeros_like(na)
            # Check for the existance of any roots and add them
            if hybridisation.lower_index_root is not None:
                # Plot a Delta function at this index of eps
                na_localised_states[hybridisation.lower_index_root] = 1
            if hybridisation.upper_index_root is not None:
                # Plot a Delta function at this index of eps
                na_localised_states[hybridisation.upper_index_root] = 1

            na += na_localised_states
            
            ax1.plot(Delta, hybridisation.eps, color='tab:blue', lw=3)
            ax1.plot(na, hybridisation.eps, color=color)
            ax1.plot(Lambda, hybridisation.eps, color='tab:orange', lw=3, alpha=0.25)
            ax1.plot(eps_a_line, hybridisation.eps, color=color, lw=3, alpha=0.25)

            if j == 0:
                ax1.annotate(r'$\epsilon_{d} = %.1f$ eV' % newns_epsd, xy=(x_pos+0.1, newns_epsd + 1.5), xytext=(x_pos+0.1, 5),
                            xycoords='data', textcoords='data',
                            color='tab:blue',
                            arrowprops=dict(arrowstyle="->",
                                            connectionstyle="arc3,rad=0.2",
                                            color='tab:blue'),
                            fontsize=12)

    #-------- Read in the DFT data --------#
    # Load the Vsd and filling data
    with open("inputs/data_from_LMTO.json", 'r') as handle:
        data_from_LMTO = json.load(handle)
    Vsd_data = data_from_LMTO["Vsdsq"]
    filling_data = data_from_LMTO["filling"]
    # Load the pdos data
    pdos_data = json.load(open('output/pdos_PBE_scf.json'))

    #-------- Plot Figure 1 --------#
    for row_index, row_metals in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        for i, element in enumerate(row_metals):
            # Get the data for the element
            if element == 'X':
                continue
            try:
                energies, pdos_metal_unnorm = pdos_data['slab'][element]
                energies_C, pdos_C_unnorm = pdos_data['C'][element]
                energies_O, pdos_O_unnorm = pdos_data['O'][element]
            except KeyError:
                continue
            
            # Normalise the pdos just that it has a max of 1
            pdos_metal = normalise_pdos(pdos_metal_unnorm)
            pdos_C = normalise_pdos(pdos_C_unnorm)
            pdos_O = normalise_pdos(pdos_O_unnorm)

            # Add a constant to make the dos look shifted
            # dependending on the element
            pdos_metal += i
            pdos_C += i
            pdos_O += i

            # Plot the pdos onto the metal states
            axp[0,row_index].fill_between(energies, i, pdos_metal, color=colors[i], alpha=0.5) 
            axp[0,row_index].annotate(element, xy=(-8.5, pdos_metal[-1]+0.5), color=colors[i])
            # Plot the C (sp) pdos
            axp[0,row_index].plot(energies_C, pdos_C, color='k', alpha=0.75)
            # Plot the O (sp) pdos
            axp[0,row_index].plot(energies_O, pdos_O, color='tab:red', alpha=0.75)

            # Get the d-band centre and d-band width
            second_moment, eps_d = moment_generator(energies, [pdos_metal_unnorm], 2)
            width = 4 * np.sqrt(second_moment)

            # Plot all the quantities that will be useful in the model.
            if row_index == 0:
                color_row ='tab:blue'
            elif row_index == 1:
                color_row ='tab:orange'
            elif row_index == 2:
                color_row ='tab:green'
            ax2.plot(filling_data[element], eps_d, 'o', color=color_row)
            ax3.plot(filling_data[element], width, 'o', color=color_row)
            ax4.plot(filling_data[element], Vsd_data[element], 'o', color=color_row)

    # Add figure numbers
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate([ax1] + list(axp.flatten()) + [ax2, ax3, ax4]):
        if i in [1, 2, 3]:
            a.annotate(alphabet[i]+')', xy=(0.01, 0.95), xycoords='axes fraction')
        else:
            a.annotate(alphabet[i]+')', xy=(0.01, 0.85), xycoords='axes fraction')


    # Save the figure
    fig.savefig('output/figure_2.png', dpi=300)