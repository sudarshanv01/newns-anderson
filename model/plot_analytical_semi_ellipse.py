
from NewnsAnderson import NewnsAndersonAnalytical

import numpy as np
import matplotlib.pyplot as plt
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':

    EPSILON_RANGE = np.linspace(-15, 10, 1000)
    BETA = [4, 6]
    EPSILON_SIGMA = [0, -2.5, -5.]
    EPSILON_D = np.linspace(-7, 4, 50)
    PLOT_DOS = False

    fige, axe = plt.subplots(1, 1, figsize=(8, 6))


    for s, eps_sigma in enumerate(EPSILON_SIGMA):

        if PLOT_DOS:
            fig, ax = plt.subplots(len(BETA), len(EPSILON_D), figsize=(5*len(EPSILON_D), 4*len(BETA)), constrained_layout=True)

        for b, beta in enumerate(BETA):
            all_energies = []
            for d, eps_d in enumerate(EPSILON_D):

                newns = NewnsAndersonAnalytical(beta = beta, 
                                                eps_d = eps_d,
                                                eps_sigma = eps_sigma,
                                                eps_range = EPSILON_RANGE )
                if PLOT_DOS:
                    energy_variation = ( newns.eps - newns.eps_sigma ) / 2 / beta
                    ax[b,d].plot( EPSILON_RANGE, newns.Delta, label = r'$\Delta$' )
                    ax[b,d].plot( EPSILON_RANGE, newns.Lambda, label = r'$\Lambda$' )
                    ax[b,d].plot( EPSILON_RANGE, energy_variation, label = r'$\epsilon$' )
                    ax[b,d].plot( EPSILON_RANGE, newns.rho_aa, label='$\rho_{aa}$')
                    ax[b,d].annotate( r'$\beta = %.1f$' % beta,
                                        xy = (0.01, 0.9),
                                        xycoords='axes fraction',
                                        horizontalalignment='left',
                                        verticalalignment='top' )

                    ylim = [ - np.max(newns.Delta ), np.max(newns.Delta ) ]
                    ax[b,d].set_ylim(ylim)

                    print(f'Energy in units of 2 beta is {newns.energy}')

                    ax[b,d].set_xlabel( r'$\epsilon (2\beta)$' )
                    ax[b,d].set_ylabel( r'$\Delta, \Lambda$ ($2\beta$)' )

                all_energies.append(newns.hyb_energy)
            
            axe.plot(EPSILON_D, all_energies, '-o', label = r'$ \beta = %1.2f, \epsilon_\sigma = %1.2f$'%(beta, eps_sigma))

        if PLOT_DOS:
            fig.savefig('output/NewnsAnderson_analytical_eps_sigma_%1.2f.png'%eps_sigma)

    axe.set_xlabel(r'$\epsilon_\d$')
    axe.set_ylabel(r'$\Delta E_{\sigma}$')
    axe.legend(loc='best')

    fige.savefig('output/NewnsAnderson_analytical_energy.png')


