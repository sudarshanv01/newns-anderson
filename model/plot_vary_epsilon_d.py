
from NewnsAnderson import NewnsAndersonAnalytical

import numpy as np
import matplotlib.pyplot as plt
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':

    EPSILON_RANGE = np.linspace(-20, 20, 2000) # in eV
    BETA_PRIME = [1, 1.5] # in 2beta
    EPSILON_SIGMA = [0., -2.5, -5.] # in eV
    EPSILON_D = np.linspace(-7, 4, 5) # in eV
    BETA = 0.5 # in units of eV
    PLOT_DOS = True

    fige, axe = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    for s, eps_sigma in enumerate(EPSILON_SIGMA):

        if PLOT_DOS:
            fig, ax = plt.subplots(len(BETA_PRIME), len(EPSILON_D), figsize=(5*len(EPSILON_D), 4*len(BETA_PRIME)), constrained_layout=True)

        for b, beta_p in enumerate(BETA_PRIME):

            all_energies = []
            for d, eps_d in enumerate(EPSILON_D):

                newns = NewnsAndersonAnalytical(beta = beta_p, 
                                                beta_p = beta_p,
                                                eps_d = eps_d,
                                                eps_sigma = eps_sigma,
                                                eps = EPSILON_RANGE )

                if PLOT_DOS:
                    # All quantities plotted in units of 2beta
                    ax[b,d].plot( newns.eps , newns.Delta, label = r'$\Delta$' )
                    ax[b,d].plot( newns.eps , newns.Lambda, label = r'$\Lambda$' )
                    ax[b,d].plot( newns.eps , newns.eps - newns.eps_sigma, label = r'$\epsilon$' )
                    ax[b,d].fill_between( newns.eps , newns.rho_aa, color='tab:red', label='$\rho_{aa}$')

                    ax[b,d].annotate( r"$\beta' = %.1f$" % beta_p,
                                        xy = (0.01, 0.9),
                                        xycoords='axes fraction',
                                        horizontalalignment='left',
                                        verticalalignment='top' )
                    print(newns.lower_band_edge*2*beta_p)
                    ax[b,d].axvline(newns.root_positive, ls='--', color='tab:grey')

                    ylim = [ - np.max(BETA_PRIME) - 0.2, np.max(BETA_PRIME) + 0.2 ]
                    ax[b,d].set_ylim(ylim)


                    if d == 0:
                        ax[b,d].set_ylabel( r'$\Delta, \Lambda$ ($2\beta$)' )
                    ax[b,d].set_xlabel( r'$\epsilon (eV)$' )

                all_energies.append(newns.DeltaE)
                if newns.has_localised_occupied_state:
                    axe.plot(eps_d, newns.DeltaE, 'v', color='k')
            
            axe.plot(EPSILON_D, all_energies, '-o', label = r"$ \beta' = %1.2f, \epsilon_\sigma = %1.2f$"%(beta_p, eps_sigma))

        if PLOT_DOS:
            fig.savefig('output/NewnsAnderson_analytical_eps_sigma_%1.2f.png'%eps_sigma)

    axe.set_xlabel(r'$\epsilon_\d$ (eV) ')
    axe.set_ylabel(r'$\Delta E_{\sigma}$ ($2\beta$)')
    axe.legend(loc='best')

    fige.savefig('output/NewnsAnderson_analytical_energy.png')


