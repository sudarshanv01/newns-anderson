
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
    BETA = [  2, 3, 4, 6, 8, 10] # interaction with metal atoms, eV
    METAL_ENERGIES = 0 # center of d-band, eV
    BETA_PRIME = 2 # Interaction with the adsorbate, eV 
    ADSORBATE_ENERGIES = np.linspace(-1, 1, 25) # 2beta' CONVERT UNITS
    EPSILON_RANGE = np.linspace(-10, 10 , 10000) # 2beta CONVERT UNITS
    PLOT_DOS = False # Plot the dos is the number is small
    FERMI_LEVEL = [0.0, 0.9]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)

    for e_fermi in FERMI_LEVEL:

        for b, beta in enumerate(BETA): 
            if PLOT_DOS:
                figd, axd = plt.subplots(1, len(ADSORBATE_ENERGIES),
                        figsize=(6*len(ADSORBATE_ENERGIES), 5), constrained_layout=True)

            all_energies = []
            for d, eps_sigma in enumerate(ADSORBATE_ENERGIES):

                eps_sigma_eV = eps_sigma * 2 * BETA_PRIME 
                eps_range = EPSILON_RANGE * 2 * beta

                newns = NewnsAndersonAnalytical(beta = beta, 
                                                beta_p = BETA_PRIME/beta,
                                                eps_d = METAL_ENERGIES,
                                                eps_sigma = eps_sigma_eV,
                                                eps = eps_range,
                                                fermi_energy=e_fermi/2/beta)

                # if newns.has_localised_occupied_state:
                #     ax.plot(newns.eps_sigma*beta/BETA_PRIME, newns.DeltaE*beta/BETA_PRIME, 'v', color='k')

                all_energies.append([newns.eps_sigma*beta/BETA_PRIME - e_fermi/2/BETA_PRIME, newns.DeltaE*beta/BETA_PRIME])

                if PLOT_DOS:

                    # All quantities plotted in units of 2beta
                    axd[d].plot( newns.eps , newns.Delta, label = r'$\Delta$' )
                    axd[d].plot( newns.eps , newns.Lambda, label = r'$\Lambda$' )
                    axd[d].plot( newns.eps , newns.eps - newns.eps_sigma, label = r'$\epsilon$' )
                    axd[d].axvline( newns.eps_sigma, color='tab:orange', ls='-.' )
                    axd[d].fill_between( newns.eps , newns.rho_aa, color='tab:red', label='$\rho_{aa}$')
                    axd[d].annotate( r"$\beta = %.1f$" % beta,
                                        xy = (0.01, 0.9),
                                        xycoords='axes fraction',
                                        horizontalalignment='left',
                                        verticalalignment='top' )

                    ylim = [ - 2*BETA_PRIME - 0.2, 2*BETA_PRIME + 0.2 ]
                    axd[d].set_ylim(ylim)
                    axd[d].plot( newns.lower_band_edge, newns.Lambda_at_band_edge, '*', color='tab:green')
                    axd[d].plot( newns.lower_band_edge, newns.Delta_at_band_edge, '*', color='tab:green')
                    
                    # If it is a real root, plot it
                    if not newns.has_complex_root: 
                        if newns.has_localised_occupied_state: 
                            axd[d].axvline( newns.root_positive, ls='--', color='tab:green')
                            axd[d].plot( newns.eps_l_sigma, 0, '*', color='tab:red')

                    if d == 0:
                        axd[d].set_ylabel( r'$\Delta, \Lambda$ ($2\beta$)' )
                    axd[d].set_xlabel( r'$\epsilon (2\beta)$' )

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
                figd.savefig(f'output/NewnsAnderson_vary_eps_a_DOS_beta_{beta}.png')

            all_a, all_hyb = np.array(all_energies).T
            all_a_sorted = all_a[np.argsort(all_a)]
            all_hyb_sorted = all_hyb[np.argsort(all_a)]

            if e_fermi == 0: 
                ax.plot(all_a_sorted, all_hyb_sorted, 'o-', alpha=0.5, label=r"$\epsilon_f$ = %1.2f, $\beta$ = %1.2f"%(e_fermi, beta))
            else:
                ax.plot(all_a_sorted, all_hyb_sorted, '--', alpha=0.25, label=r"$\epsilon_f$ = %1.2f, $\beta$ = %1.2f"%(e_fermi, beta))
            ax.axvline(0, color='k', ls='--')

    ax.set_xlabel(r"$\epsilon_a $ ($2 \beta'$) ")
    ax.set_ylabel(r" $\Delta E$ ($2 \beta'$)")
    ax.set_xlim([-1, 1])
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    fig.savefig('output/NewnsAnderson_vary_eps_a_vary_beta.png')

