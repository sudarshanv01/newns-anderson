"""Plot the variation of the energy against the d-band center."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()

def plot_dos(ax):

    # Plot the quantities first
    ax.axvline( newns.eps_d, color='k', ls='--' )
    ax.axvline( newns.eps_sigma, color='b', ls='--')
    ax.axvline( newns.lower_band_edge, ls='-.', color='tab:grey')
    ax.plot( newns.lower_band_edge, newns.Lambda_at_band_edge, '*', color='tab:green')
    ax.plot( newns.lower_band_edge, newns.Delta_at_band_edge, '*', color='tab:green')
    ax.axvline( newns.root_positive, ls='-.', color='tab:olive')
    ax.axvline( newns.root_negative, ls='-.', color='tab:olive')

    # All quantities plotted in units of 2beta
    ax.plot( newns.eps , newns.Delta, label = r'$\Delta$', lw=3)
    ax.plot( newns.eps , newns.Lambda, label = r'$\Lambda$', lw=3)
    ax.plot( newns.eps , newns.eps - newns.eps_sigma, label = r'$\epsilon$' )

    # Plot the density of states of the adsorbate
    ax.fill_between( newns.eps , newns.rho_aa, color='tab:red', label='$\rho_{aa}$')

    # Annotate quantities
    ax.annotate( r"$\beta' = %.1f$" % (beta_p / 2 / BETA),
                        xy = (0.01, 0.9),
                        xycoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='top' )

    if not newns.has_complex_root and newns.has_localised_occupied_state: 
        ax.axvline( newns.root_positive, ls='--', color='tab:green')
        ax.plot( newns.eps_l_sigma, 0, '*', markersize=12, color='tab:red')
    
    ylim = beta_p /  BETA  
    ax.set_ylim([-ylim, ylim])

    # if newns.has_localised_occupied_state:
    #     ax.annotate( r"Localised state", 
    #                     xy=(0.9,0.01),
    #                     xycoords='axes fraction',
    #                     horizontalalignment='right',
    #                     verticalalignment='bottom' )

    # if newns.has_complex_root:
    #     ax[b,d].annotate( r"Complex root", 
    #                     xy=(0.1,0.11),
    #                     xycoords='axes fraction',
    #                     horizontalalignment='left',
    #                     verticalalignment='bottom' )

    if d == 0:
        ax[b,d].set_ylabel( r'$\Delta, \Lambda$ ($2\beta$)' )
    ax[b,d].set_xlabel( r'$\epsilon (2\beta)$' )


if __name__ == '__main__':
    """Plot the variation of the energy against the d-band center.
    The energy is in units of 2beta and the d-band center is in units of 2beta.
    """
    EPSILON_RANGE = np.linspace(-25, 15, 1000) # range of energies plot in dos
    BETA_PRIME = [0.5, 2] # Interaction of metal and adsorbate # in eV 
    EPSILON_SIGMA = [-2.5, -5.] # renormalised energy of adsorbate
    EPSILON_D = np.linspace(-15, 6) # Band center in eV 
    BETA = 2 # in units of eV
    NUM_DENSITY_OF_STATES = 5 # Number of density of states to plot
    colors = cm.viridis(np.linspace(0, 1, len(EPSILON_SIGMA) * len(BETA_PRIME)))

    # Plot the energies in this figure
    fige, axe = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    # Plot the components of the energy in this figure
    figs, axs = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    # Plot a subsection of the density of states at different epsilon_d values
    figd, axd = plt.subplots(1, NUM_DENSITY_OF_STATES, figsize=(12, 4), constrained_layout=True)

    # eps_sigma here would be for different adsorbates
    index = 0
    for s, eps_sigma in enumerate(EPSILON_SIGMA):
        for b, beta_p in enumerate(BETA_PRIME):

            all_energies = []
            all_eps_sigma_pos = []
            all_eps_sigma_neg = []
            all_tan_comp = []

            for d, eps_d in enumerate(EPSILON_D):
                newns = NewnsAndersonAnalytical(beta = BETA, 
                                                beta_p = beta_p, 
                                                eps_d = eps_d,
                                                eps_sigma = eps_sigma,
                                                eps = EPSILON_RANGE )

                # The quantity that we want to plot
                all_energies.append     ( [ newns.eps_d, newns.DeltaE           ] )
                all_eps_sigma_pos.append( [ newns.eps_d, newns.eps_l_sigma_pos  ] )
                all_eps_sigma_neg.append( [ newns.eps_d, newns.eps_l_sigma_neg  ] )
                all_tan_comp.append     ( [ newns.eps_d, newns.arctan_component ] )

                if newns.has_localised_occupied_state_positive:
                    axe.plot(newns.eps_d, newns.DeltaE, '*', color='k')
                    axs[0].plot( newns.eps_d, newns.eps_l_sigma_pos, '*', color='k')
                if newns.has_localised_occupied_state_negative:
                    axe.plot(newns.eps_d, newns.DeltaE, 'v', color='k')
                    axs[0].plot( newns.eps_d, newns.eps_l_sigma_neg, 'v', color='k')

            # Plot the energies 
            all_energies = np.array(all_energies).T
            axe.plot( all_energies[0], all_energies[1], '-o', alpha=0.5, color=colors[index],
                     label = r"$ \beta' = %1.2f, \epsilon_\sigma = %1.2f$"%(beta_p, newns.eps_sigma))

            all_eps_sigma_pos = np.array(all_eps_sigma_pos).T
            axs[0].plot( all_eps_sigma_pos[0], all_eps_sigma_pos[1], '-v', alpha=0.5, color=colors[index], 
                            label = r"$ \beta' = %1.2f, \epsilon_\sigma(+) = %1.2f$"%(beta_p, newns.eps_sigma))

            all_eps_sigma_neg = np.array(all_eps_sigma_neg).T
            axs[0].plot( all_eps_sigma_neg[0], all_eps_sigma_neg[1], '-o', alpha=0.5, color=colors[index],
                            label = r"$ \beta' = %1.2f, \epsilon_\sigma(-) = %1.2f$"%(beta_p, newns.eps_sigma))

            all_tan_comp = np.array(all_tan_comp).T
            axs[1].plot( all_tan_comp[0], all_tan_comp[1], '-o', alpha=0.5, color=colors[index], 
                        label = r"$ \beta' = %1.2f, \epsilon_\sigma = %1.2f$"%(beta_p, newns.eps_sigma))
            index += 1
            
        # Plot the renormalised energy of the adsorbate
        axe.axvline(newns.eps_sigma, color='k', linestyle='--', alpha=0.25)
        axe.axhline(newns.eps_sigma, color='k', linestyle='--', alpha=0.25)



    axe.set_xlabel(r'$\epsilon_d$ ($2\beta$) ')
    axe.set_ylabel(r'$\Delta E$ ($2\beta$)')
    axe.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    fige.savefig('output/NewnsAnderson_vary_eps_d.png')

    axs[0].set_xlabel(r'$\epsilon_d$ ($2\beta$) ')
    axs[0].set_ylabel(r'$\epsilon_{l,\sigma}$ ($2\beta$)')
    axs[1].set_xlabel(r'$\epsilon_d$ ($2\beta$) ')
    axs[1].set_ylabel(r'$\pi^{-1}\int \mathregular{arctan} ( \Delta / \epsilon - \epsilon_{\sigma} - \Lambda ) $ ($2\beta$)')
    axs[0].legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    figs.savefig('output/NewnsAnderson_vary_eps_d_plot_eps_ls.png')



