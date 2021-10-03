"""Plot the variation of the energy against the d-band center."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()

def plot_dos(ax):
    """Plot the density of states for a given set of parameters in the newns class."""

    # All quantities plotted in units of 2beta
    ax.plot( newns.eps , newns.Delta, label = r'$\Delta$', lw=3)
    ax.plot( newns.eps , newns.Lambda, label = r'$\Lambda$', lw=3)
    ax.plot( newns.eps , newns.eps - newns.eps_sigma, label = r'$\epsilon$' )

    ax.axvline( newns.root_positive, ls='-.', color='tab:olive')
    ax.axvline( newns.root_negative, ls='-.', color='tab:olive')

    # Plot the density of states of the adsorbate
    ax.fill_between( newns.eps , newns.rho_aa, color='tab:red', label='$\rho_{aa}$')

    # Plot parameters
    ax.set_ylim([-np.max(newns.Delta), np.max(newns.Delta)])
    if d == 0:
        ax.set_ylabel( r'$\Delta, \Lambda$ ($2\beta$)' )
    ax.set_xlabel( r'$\epsilon (2\beta)$' )


if __name__ == '__main__':
    """Plot the variation of the energy against the d-band center.
    The energy is in units of 2beta and the d-band center is in units of 2beta.
    """
    EPSILON_RANGE = np.linspace(-15, 15, 4000) # range of energies plot in dos
    BETA_PRIME = [ 2, 4 ] # Interaction of metal and adsorbate in 2beta units 
    EPSILON_SIGMA = [ -2, -4 ] # renormalised energy of adsorbate
    EPSILON_D = np.linspace(-8, 2) # Band center in eV 
    BETA = 2 # in units of eV
    NUM_DENSITY_OF_STATES = 5 # Number of density of states to plot
    colors = cm.RdBu(np.linspace(0, 1, len(EPSILON_SIGMA) * len(BETA_PRIME)))
    FERMI_ENERGY = 0.0 # Fermi energy in 2beta units
    U = 0.0 # no coulomb interaction

    # Plot the energies in this figure
    fige, axe = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    # Specifics of the plot
    axe.axvline(-2 * BETA, ls='--', color='k', alpha=0.5)
    # axe.annotate(r'$\it{d}$-band' +'\noutside \nFermi level', xy=(0.6, 0.7), xycoords='axes fraction',)
    # Plot the components of the energy in this figure
    figs, axs = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # eps_sigma here would be for different adsorbates
    index = 0
    for s, eps_sigma in enumerate(EPSILON_SIGMA):
        for b, beta_p in enumerate(BETA_PRIME):

            all_energies = []
            all_eps_sigma_pos = []
            all_eps_sigma_neg = []
            all_tan_comp = []

            # Plot a subsection of the density of states at different epsilon_d values
            figd, axd = plt.subplots(1, NUM_DENSITY_OF_STATES, figsize=(18, 4), constrained_layout=True)
            # Pick 5 evenly spread out indices for the density of states
            plot_indices = np.linspace(0, len(EPSILON_D) - 1, NUM_DENSITY_OF_STATES)
            plot_indices = np.int_(plot_indices)
            plotted_index = 0

            for d, eps_d in enumerate(EPSILON_D):
                newns = NewnsAndersonAnalytical(beta = BETA, 
                                                beta_p = beta_p/2/BETA, 
                                                eps_d = eps_d,
                                                eps_a = eps_sigma,
                                                eps = EPSILON_RANGE,
                                                fermi_energy = FERMI_ENERGY,
                                                U = U)
                newns.self_consistent_calculation()
                energy_in_eV = newns.eps_d * 2 * BETA
                deltaE_in_eV = newns.DeltaE * 2 * BETA
                # The quantity that we want to plot
                all_energies.append     ( [ energy_in_eV, deltaE_in_eV           ] )
                if newns.has_localised_occupied_state_positive:
                    all_eps_sigma_pos.append( [ energy_in_eV, newns.eps_l_sigma_pos  ] )
                if newns.has_localised_occupied_state_negative:
                    all_eps_sigma_neg.append( [ energy_in_eV, newns.eps_l_sigma_neg  ] )
                all_tan_comp.append     ( [ energy_in_eV, newns.arctan_component ] )

                if newns.has_localised_occupied_state_positive:
                    axe.plot( energy_in_eV, deltaE_in_eV, '*', color='k')
                    # axe.annotate('L+', xy=( energy_in_eV, deltaE_in_eV+0.3), fontsize=8)
                    # axe.plot( energy_in_eV, deltaE_in_eV, 'o', color=colors[index])
                if newns.has_localised_occupied_state_negative:
                    # axe.annotate('L-', xy=( energy_in_eV, deltaE_in_eV+0.3), fontsize=8)
                    axe.plot( energy_in_eV, deltaE_in_eV, 'v', color='k')
                
                if d in plot_indices:
                    plot_dos(axd[plotted_index])
                    plotted_index += 1

            # Plot the energies 
            eps_a_in_eV = newns.eps_sigma * 2 * BETA
            all_energies = np.array(all_energies).T
            axe.plot( all_energies[0], all_energies[1], '-o', alpha=0.5, color=colors[index],
                     label = r"$ V_{\rm ak} = %1.2f$ eV, $\epsilon_a = %1.2f$ eV"%(beta_p, eps_a_in_eV))

            all_eps_sigma_pos = np.array(all_eps_sigma_pos).T
            if all_eps_sigma_pos.size != 0:
                axs[0].plot( all_eps_sigma_pos[0], all_eps_sigma_pos[1], '-v', alpha=0.5, color=colors[index], 
                                label = r"$ \beta' = %1.2f, \epsilon_\sigma(+) = %1.2f$"%(beta_p, newns.eps_sigma))

            all_eps_sigma_neg = np.array(all_eps_sigma_neg).T
            if all_eps_sigma_neg.size != 0:
                axs[0].plot( all_eps_sigma_neg[0], all_eps_sigma_neg[1], '-o', alpha=0.5, color=colors[index],
                                label = r"$ \beta' = %1.2f, \epsilon_\sigma(-) = %1.2f$"%(beta_p, newns.eps_sigma))

            all_tan_comp = np.array(all_tan_comp).T
            axs[1].plot( all_tan_comp[0], all_tan_comp[1], '-o', alpha=0.5, color=colors[index], 
                        label = r"$ \beta' = %1.2f, \epsilon_\sigma = %1.2f$"%(beta_p, newns.eps_sigma))

            # Save the figure for the density of states
            figd.savefig('output/dos/dos_%1.2f_%1.2f.png'%(beta_p, eps_sigma))
            index += 1
            

    axe.set_xlabel(r'$\epsilon_d$ (eV) ')
    axe.set_ylabel(r'$\Delta E$ (eV)')
    # axe.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    axe.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    fige.savefig('output/NewnsAnderson_vary_eps_d.png')

    axs[0].set_xlabel(r'$\epsilon_d$ (eV) ')
    axs[0].set_ylabel(r'$\epsilon_{l,\sigma}$ ($2\beta$)')
    axs[1].set_xlabel(r'$\epsilon_d$ ($2\beta$) ')
    axs[1].set_ylabel(r'$\pi^{-1}\int \mathregular{arctan} ( \Delta / \epsilon - \epsilon_{\sigma} - \Lambda ) $ ($2\beta$)')
    axs[0].legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    figs.savefig('output/NewnsAnderson_vary_eps_d_components.png')



