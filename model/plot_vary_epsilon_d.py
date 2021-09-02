
from NewnsAnderson import NewnsAndersonAnalytical

import numpy as np
import matplotlib.pyplot as plt
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':

    EPSILON_RANGE = np.linspace(-40, 15, 10000) # range of energies plot in dos
    BETA_PRIME = [4, 6] # Interaction of metal and adsorbate # in eV 
    EPSILON_SIGMA = [-2.5, -5.] # renormalised energy of adsorbate
    EPSILON_D = np.linspace(-10, 2, 20) # Band center in eV 
    BETA = 2 # in units of eV
    PLOT_DOS = False

    fige, axe = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    figs, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    for s, eps_sigma in enumerate(EPSILON_SIGMA):

        if PLOT_DOS:
            fig, ax = plt.subplots(len(BETA_PRIME), len(EPSILON_D), figsize=(5*len(EPSILON_D), 4*len(BETA_PRIME)), constrained_layout=True)

        for b, beta_p in enumerate(BETA_PRIME):

            all_energies = []
            all_eps_sigma = [] # See the variation of the localised state if it exists
            for d, eps_d in enumerate(EPSILON_D):
                newns = NewnsAndersonAnalytical(beta = BETA, 
                                                beta_p = beta_p / 2 / BETA,
                                                eps_d = eps_d,
                                                eps_sigma = eps_sigma,
                                                eps = EPSILON_RANGE )
                if not newns.has_complex_root and newns.has_localised_occupied_state: 
                    all_eps_sigma.append([ eps_d, newns.eps_sigma ] )

                if PLOT_DOS:
                    # Plot the quantities first
                    # ax[b,d].axvline( newns.eps_d, color='k', ls='--' )
                    ax[b,d].axvline( newns.eps_sigma, color='b', ls='--')
                    # ax[b,d].axvline( newns.lower_band_edge, ls='-.', color='tab:grey')
                    # ax[b,d].plot( newns.lower_band_edge, newns.Lambda_at_band_edge, '*', color='tab:green')
                    # ax[b,d].plot( newns.lower_band_edge, newns.Delta_at_band_edge, '*', color='tab:green')

                    # All quantities plotted in units of 2beta
                    ax[b,d].plot( newns.eps , newns.Delta, label = r'$\Delta$', lw=3)
                    ax[b,d].plot( newns.eps , newns.Lambda, label = r'$\Lambda$', lw=3)
                    ax[b,d].plot( newns.eps , newns.eps - newns.eps_sigma, label = r'$\epsilon$' )

                    # Plot the density of states of the adsorbate
                    ax[b,d].fill_between( newns.eps , newns.rho_aa, color='tab:red', label='$\rho_{aa}$')

                    # Annotate quantities
                    ax[b,d].annotate( r"$\beta' = %.1f$" % (beta_p / 2 / BETA),
                                        xy = (0.01, 0.9),
                                        xycoords='axes fraction',
                                        horizontalalignment='left',
                                        verticalalignment='top' )

                    if not newns.has_complex_root and newns.has_localised_occupied_state: 
                        ax[b,d].axvline( newns.root_positive, ls='--', color='tab:green')
                        ax[b,d].plot( newns.eps_l_sigma, 0, '*', markersize=12, color='tab:red')
                    
                    ylim = beta_p /  BETA  
                    ax[b,d].set_ylim([-ylim, ylim])

                    if newns.has_localised_occupied_state:
                        ax[b,d].annotate( r"Localised state", 
                                        xy=(0.9,0.01),
                                        xycoords='axes fraction',
                                        horizontalalignment='right',
                                        verticalalignment='bottom' )
                    if newns.has_complex_root:
                        ax[b,d].annotate( r"Complex root", 
                                        xy=(0.1,0.11),
                                        xycoords='axes fraction',
                                        horizontalalignment='left',
                                        verticalalignment='bottom' )

                    if d == 0:
                        ax[b,d].set_ylabel( r'$\Delta, \Lambda$ ($2\beta$)' )
                    ax[b,d].set_xlabel( r'$\epsilon (2\beta)$' )
                # if newns.has_localised_occupied_state:
                all_energies.append([ newns.eps_d, newns.DeltaE ])
                # if newns.has_localised_occupied_state:
                #     axe.plot(eps_d, newns.DeltaE, 'v', color='k')
                # if newns.has_complex_root:
                #     axe.plot(eps_d, newns.DeltaE, '^', color='k')
                #     print('Has complex roots')
            
            all_energies = np.array(all_energies).T
            axe.plot(all_energies[0], all_energies[1], '-o', alpha=0.5, label = r"$ \beta' = %1.2f, \epsilon_\sigma = %1.2f$"%(beta_p, eps_sigma))
            try:
                all_eps_sigma = np.array(all_eps_sigma).T
                axs.plot( all_eps_sigma[0], all_eps_sigma[1], '-o', alpha=0.5, label = r"$ \beta' = %1.2f, \epsilon_\sigma = %1.2f$"%(beta_p, eps_sigma/2/BETA))
            except IndexError:
                pass

        if PLOT_DOS:
            fig.savefig('output/NewnsAnderson_vary_eps_d_DOS_eps_a_%1.2f.png'%eps_sigma)

    axe.set_xlabel(r'$\epsilon_d$ ($2\beta$) ')
    axe.set_ylabel(r'$\Delta E$ ($2\beta$)')
    axe.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    fige.savefig('output/NewnsAnderson_vary_eps_d.png')

    axs.set_xlabel(r'$\epsilon_d$ (eV) ')
    axs.set_ylabel(r'$\epsilon_{l,\sigma}$ ($2\beta$)')
    axs.legend(loc='best')
    figs.savefig('output/NewnsAnderson_vary_eps_d_plot_eps_ls.png')



