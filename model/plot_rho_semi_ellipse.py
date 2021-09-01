
from NewnsAnderson import NewnsAndersonAnalytical
import numpy as np
import matplotlib.pyplot as plt
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':

    EPSILON_RANGE = np.linspace(-2, 2, 150) # in eV 
    BETA_PRIME = [1/3, 1/2, 1/np.sqrt(2), 1.975] # in 2 beta 
    BETA = 0.5 # eV 
    EPSILON_SIGMA = np.array([1/2, 0, 0, -0.258]) * 2 * BETA # units of eV
    EPSILON_D = np.array([0, 0, 0, 0]) # in eV
    e_fermi_Ni = 1.76 # eV

    fige, axe = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    for i, beta_p in enumerate(BETA_PRIME):
        print('beta prime = {} (units: 2beta)'.format(beta_p))

        # All energy quantities in eV
        epsilon_sigma = EPSILON_SIGMA[i]
        epsilon_d = EPSILON_D[i]
        epsilon_range = EPSILON_RANGE

        analytical = NewnsAndersonAnalytical(beta_p=beta_p,
                                             beta=BETA, 
                                             eps_sigma=epsilon_sigma, 
                                             eps_d=epsilon_d, 
                                             eps=epsilon_range)
        
        # Convert rho into eV
        rho_aa_eV = analytical.rho_aa / 2 / BETA

        # Plot in terms of 2beta
        axe.plot(analytical.eps, rho_aa_eV, '-', label=r"$\beta' = %1.2f, \epsilon_{\sigma}=%1.2f$"%(beta_p, analytical.eps_sigma))

        axe.set_ylabel(r'$\rho_{aa}^{\sigma}$ (eV$^{-1}$)')
        axe.set_xlabel(r'$\epsilon / 2\beta $')
        axe.legend(loc='best')
        axe.set_ylim([0, 1])

    fige.savefig('output/NewnsAnderson_different_dos.png')

        


