
"""Recreating Figure 4 of the Newns-Anderson paper
All units are in 2beta'
"""

import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()


if __name__ == '__main__':
    # Define the parameters
    BETA = 0.5 # eV
    ADSORBATE_ENERGIES = np.linspace(-1, 1, 5) # 2beta' CONVERT UNITS
    METAL_ENERGIES = 0 # eV
    BETA_PRIME = [0.3, 1, 2, 3] 
    EPSILON_RANGE = np.linspace(-15, 15 , 1000)
    PLOT_DOS = True

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)


    for b, beta_p in enumerate(BETA_PRIME): 
        if PLOT_DOS:
            figd, axd = plt.subplots(1, len(ADSORBATE_ENERGIES),
                    figsize=(5*len(ADSORBATE_ENERGIES), 4), constrained_layout=True)

        all_energies = []
        for d, eps_sigma in enumerate(ADSORBATE_ENERGIES):
            eps_sigma_eV = eps_sigma * 2 * beta_p
            newns = NewnsAndersonAnalytical(beta = beta_p, 
                                            beta_p = beta_p,
                                            eps_d = METAL_ENERGIES,
                                            eps_sigma = eps_sigma_eV,
                                            eps = EPSILON_RANGE )
            
            if newns.has_localised_occupied_state:
                ax.plot(newns.eps_sigma, newns.DeltaE, 'v', color='k')

            all_energies.append([newns.eps_sigma, newns.DeltaE])

            if PLOT_DOS:

                # All quantities plotted in units of 2beta
                axd[d].plot( EPSILON_RANGE , newns.Delta, label = r'$\Delta$' )
                axd[d].plot( EPSILON_RANGE , newns.Lambda, label = r'$\Lambda$' )
                axd[d].plot( EPSILON_RANGE , newns.eps - newns.eps_sigma, label = r'$\epsilon$' )
                axd[d].fill_between( EPSILON_RANGE , newns.rho_aa, color='tab:red', label='$\rho_{aa}$')
                axd[d].annotate( r"$\beta' = %.1f$" % beta_p,
                                    xy = (0.01, 0.9),
                                    xycoords='axes fraction',
                                    horizontalalignment='left',
                                    verticalalignment='top' )

                ylim = [ - beta_p - 0.2, beta_p + 0.2 ]
                axd[d].set_ylim(ylim)

                if not newns.has_complex_root:
                    axd[d].axvline(newns.root_positive * 2 * beta_p, ls='--', color='tab:green')
                    axd[d].axvline(newns.root_negative * 2 * beta_p, ls='--', color='tab:green')
                else:
                    pass

                if d == 0:
                    axd[d].set_ylabel( r'$\Delta, \Lambda$ ($2\beta$)' )
                axd[d].set_xlabel( r'$\epsilon (eV)$' )

                if newns.has_localised_occupied_state:
                    axd[d].annotate( r"Localised state", 
                                     xy=(0.9,0.01),
                                     xycoords='axes fraction',
                                     horizontalalignment='right',
                                     verticalalignment='bottom' )
                if newns.has_complex_root:
                    axd[d].annotate( r"Complex root", 
                                     xy=(0.9,0.11),
                                     xycoords='axes fraction',
                                     horizontalalignment='right',
                                     verticalalignment='bottom' )

                axd[d].axvline(newns.lower_band_edge, color='tab:grey', ls='--')

        if PLOT_DOS:
            figd.savefig(f'output/NewnsAnderson_vary_eps_a_DOS_betap_{beta_p}.png')

        all_a, all_hyb = np.array(all_energies).T
        all_a_sorted = all_a[np.argsort(all_a)]
        all_hyb_sorted = all_hyb[np.argsort(all_a)]
        
        ax.plot(all_a_sorted, all_hyb_sorted, 'o-', label=r"$\epsilon_d$ = %1.2f eV, $\beta'$ = %1.2f"%(METAL_ENERGIES, beta_p))

    ax.set_xlabel(r'$\epsilon_a$ ($2 \beta$) ')
    ax.set_ylabel(r'Hybridisation Energy ($2 \beta$)')
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    fig.savefig('output/NewnsAnderson_vary_eps_a.png')

