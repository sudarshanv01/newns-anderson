"""Plot scaling based on parameters from LCAO calculations."""
import json
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()

# Remove elements that you don't want to plot
REMOVE_ELEMENTS = ['Al', ]
# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu' ]
SECOND_ROW  = [ 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag' ]
THIRD_ROW   = [ 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au' ] 

ANNOTATE_SELECT = ['Ag', 'Au', 'Pt']

def plot_dos(ax):
    """Plot the density of states for a given set of parameters in the newns class.
    The units will be in eV, and must replicate the density of states that we see while
    fitting the semi-ellipse."""
    convert = 2 * newns.beta

    # All quantities plotted in units of 2beta
    ax.plot( newns.eps * convert, newns.Delta * convert, label = r'$\Delta$', lw=3)
    ax.plot( newns.eps * convert, newns.Lambda * convert, label = r'$\Lambda$', lw=3)
    ax.plot( newns.eps * convert, ( newns.eps - newns.eps_a ) * convert, label = r'$\epsilon$' )

    ax.axvline( newns.root_positive * convert, ls='-.', color='tab:olive')
    ax.axvline( newns.root_negative * convert, ls='-.', color='tab:olive')

    # Plot the density of states of the adsorbate
    ax.fill_between( newns.eps * convert, newns.rho_aa * convert, color='tab:red', label='$\rho_{aa}$')

    # line for the eps_d
    ax.axvline( newns.eps_d * convert, ls='--', color='tab:green')

    # line for eps_a
    ax.axvline( newns.eps_a * convert, ls='--', color='tab:blue')

    # annotate beta_p on the top right
    ax.annotate( r"$\beta^{'} = %1.2f$"%newns.beta_p, xy=(0.1, 0.8), fontsize=9, xycoords='axes fraction')

    # Plot parameters
    ax.set_ylim([-np.max(newns.Delta * convert), np.max(newns.Delta * convert)])
    if index[1] == 0:
        ax.set_ylabel( r'$\Delta, \Lambda$ (eV)' )
    if index[0] == 2:
        ax.set_xlabel( r'$\epsilon - \epsilon_{f}$ (eV)' )

if __name__ == '__main__':
    """Plot the energy variation with d-band centre with 
    parameters obtained from LCAO calculations The idea is to 
    vary the eps_a values for some representative examples and see
    if change in the chemisorption energy as a function of the 
    d-band center."""

    # Load data
    with open('fit_results.json') as f:
        fit_results = json.load(f)
    
    # Common parameters
    fermi_energy = 0.0 # reference fermi level set to 0.
    U = 0.0 # No Coulomb interaction for this run 
    EPSILON_RANGE = np.linspace(-10, 5, 1000)

    # Choose different values of epsilon_a to plot against
    epsilon_a_values = [ -2, -1, 0, 1, 2 ]

    # Choose the adsorbate that will be used to get Delta
    ads = 'C'

    # Plot for each adsorbate
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), 
                        squeeze=False, constrained_layout=True)
    
    # Plot the scaling relations for subset of the adsorbates
    scaling_plots = [ [0, 1], [1, 2], [3, 4] ] 
    figs, axs = plt.subplots(1, len(scaling_plots), figsize=(4*len(scaling_plots), 4.),
                                    constrained_layout=True)

    # Create a set of plots for all elements in the FIRST, SECOND and THIRD rows
    figd, axd = plt.subplots(3, len(FIRST_ROW), figsize=(30, 10), constrained_layout=True)

    # Create a plot for the components of the energy
    figc, axc = plt.subplots(1, 3, figsize=(14, 4.5),
                        squeeze=False, constrained_layout=True)

    colors = ['tab:red', 'tab:blue', 'tab:brown', 'tab:orange', 'tab:purple'] 

    # Chosen parameters for beta, beta_prime
    beta = 0.5
    beta_p = 1 
    eps_d_ideal = np.linspace(-5, 2.5, 100)

    # Store the energies for different eps_a
    energies_epsa = []
    metals_epsa = []

    # Separate plot for different adsorbates
    for i, eps_a in enumerate(epsilon_a_values):

        # Plot the ideal variation
        energy_ideal = []
        for eps_d in eps_d_ideal:
            newns_ideal = NewnsAndersonAnalytical( eps_a = eps_a,
                                                   eps_d = eps_d,
                                                   beta = beta,
                                                   beta_p = beta_p,
                                                   fermi_energy = fermi_energy,
                                                   U = U,
                                                   eps = EPSILON_RANGE * 2 * beta)
            newns_ideal.self_consistent_calculation()
            energy_ideal.append( newns_ideal.DeltaE )

        # Decide which axis to plot on
        if eps_a >= 0:
            plot_no = 1
        else:
            plot_no = 0 
        ax[0,plot_no].plot( eps_d_ideal, energy_ideal, '--', color=colors[i], lw=2, label=r'$\epsilon_{a} = %1.1f$ eV'%eps_a)

        # If eps_a is positive, plot the V2/(e-e_a) variation as well
        if eps_a >= 0:
            # Get only the d-band values below the fermi level minus something
            # so that the quantity doesn't diverge at 0
            eps_d_ideal_below_fermi = eps_d_ideal[ np.where( eps_d_ideal < -0.5 ) ]
            single_state_energy = -eps_a + (beta_p*2*beta)**2 / ( eps_d_ideal_below_fermi - eps_a )
            ax[0,plot_no].plot( eps_d_ideal_below_fermi, single_state_energy, '-', color=colors[i], 
                        lw=2, label=r'$-\epsilon_a + V^2 / ( \epsilon_a - \epsilon_d ) $')

        # Remove results that might not have d-bands
        for remove in REMOVE_ELEMENTS:
            fit_results[ads].pop(remove, {})

        data_energy = []
        for j, metal in enumerate(fit_results[ads]):

            # The only thing we need from LCAO is the eps_d
            eps_d = fit_results[ads][metal]['eps_d']

            newns = NewnsAndersonAnalytical( eps_a = eps_a,
                                             eps_d = eps_d,
                                             beta = beta,
                                             beta_p = beta_p,
                                             U = U,
                                             eps = EPSILON_RANGE * 2 * beta,
                                             fermi_energy = fermi_energy )
            newns.self_consistent_calculation()

            # Store the energy to plot with variation in the d-band center
            DeltaE = newns.DeltaE 
            descriptor = eps_d 
            data_energy.append([eps_d, DeltaE, metal])

            ax[0,plot_no].plot( eps_d, DeltaE, 'o', color=colors[i])
            if i == 0:
                metals_epsa.append(metal)
            if metal in FIRST_ROW:
                selected_row = [0, FIRST_ROW.index(metal)]
            elif metal in SECOND_ROW:
                selected_row = [1, SECOND_ROW.index(metal)]
            elif metal in THIRD_ROW:
                selected_row = [2, THIRD_ROW.index(metal)]
            
            # Plot the density of states
            index = [ selected_row[0], selected_row[1] ]
            plot_dos(axd[index[0], index[1]])

            # Plot the components of the energy
            if newns.has_localised_occupied_state_positive:
                axc[0,0].plot( eps_d, newns.eps_l_sigma_pos, 'o', color=colors[i] )
                if metal in ANNOTATE_SELECT:
                    axc[0,0].annotate( metal, (eps_d, newns.eps_l_sigma_pos), color=colors[i], fontsize=12)
            if newns.has_localised_occupied_state_negative:
                axc[0,1].plot( eps_d, newns.eps_l_sigma_neg, 'o', color=colors[i])
                if metal in ANNOTATE_SELECT:
                    axc[0,1].annotate( metal, (eps_d, newns.eps_l_sigma_neg), color=colors[i], fontsize=12)

            # Plot also the tan-1 component
            axc[0,2].plot( eps_d, newns.arctan_component, 'o', color=colors[i]) 
    
        # Plot the energy variation with d-band centre
        d_band_centres, energies, metals = np.array(data_energy).T
        d_band_centres = np.array(d_band_centres, dtype=float)
        energies = np.array(energies, dtype=float)

        # Sort both d_band centre and energies on the basis of d-band centres
        sort_index = np.argsort(d_band_centres)
        d_band_centres = d_band_centres[sort_index]
        energies = energies[sort_index]
        energies_epsa.append(energies)
        if i == 0:
            metals_epsa = np.array(metals_epsa)
            metals_epsa = metals_epsa[sort_index]

    # Plot the energies of the two occupied states against each other
    # A sort of scaling relation between states, either occupied or unoccupied
    for i, scaling in enumerate(scaling_plots):
        # get the index of what we want to scale
        ads1, ads2 = scaling
        # Get the energies of the two states
        energies1 = energies_epsa[ads1]
        energies2 = energies_epsa[ads2]
        # Plot the energies against each other
        axs[i].plot( energies1, energies2, 'o', ms=8, color='tab:red')
        # label the axes by their energies_epsa
        axs[i].set_xlabel(r'$\Delta E(\epsilon_a=%s)$ eV'%epsilon_a_values[ads1])
        axs[i].set_ylabel(r'$\Delta E(\epsilon_a=%s)$ eV'%epsilon_a_values[ads2])

    # for i in range(len(metals_epsa)):
    #     if metals_epsa[i] in ANNOTATE_SELECT:
    #         ax[0,2].annotate( metals_epsa[i], (energies_epsa[0][i], energies_epsa[1][i]), color='k', fontsize=15)
    # ax[0,2].annotate('Early\ntransition\nmetals', xy=(0.1,0.6), xycoords='axes fraction', fontsize=15)
    # ax[0,2].set_xlabel(r'$\Delta E(\epsilon_{a}=%1.1f)$ (eV)'%epsilon_a_values[0], fontsize=16)
    # ax[0,2].set_ylabel(r'$\Delta E(\epsilon_{a}=%1.1f)$ (eV)'%epsilon_a_values[1], fontsize=16)

    ax[0, 0].set_ylabel('Energy (eV)' )
    ax[0, 0].set_xlabel('d-band centre (eV)' )
    ax[0, 1].set_ylabel('Energy (eV)' )
    ax[0, 1].set_xlabel('d-band centre (eV)' )
    # ax[0,0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #             mode="expand", borderaxespad=0, ncol=1)
    ax[0,0].legend(loc='best')
    # ax[0,1].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #             mode="expand", borderaxespad=0, ncol=1)
    ax[0,1].legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    fig.savefig('output/energy_variation_lcao.png')

    figd.suptitle('Density of states', fontsize=16)
    figd.savefig('output/dos_lcao.png')

    for i in range(len(axc)):
        axc[i,0].set_ylabel(r'$\epsilon_l^{+}$ ($2\beta$)')
        axc[i,1].set_ylabel(r'$\epsilon_l^{-}$ ($2\beta$)')
        axc[i,2].set_ylabel(r'$\pi^{-1}\int \mathregular{arctan} ( \Delta / \epsilon - \epsilon_{\sigma} - \Lambda ) $ ($2\beta$)')

    figc.savefig('output/energy_components_lcao.png')

    figs.savefig('output/energy_scaling_lcao.png')
