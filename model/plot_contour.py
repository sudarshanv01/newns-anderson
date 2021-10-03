"""Make a contour plot for the different values of eps_a and eps_d."""
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':

    eps_a = np.linspace(-6, 3, 50)
    eps_d = np.linspace(-4, 2, 50)
    BETA = 2 # Metal d-band width
    BETA_P = 2
    EPSILON_RANGE = np.linspace(-20, 20, 1000) # range of energies plot in dos
    FERMI_ENERGY = 0.0 # Fermi energy
    U = 0 # No Columb interaction

    # eps_a, eps_d = np.meshgrid(eps_a, eps_d)
    energies = np.zeros((len(eps_a), len(eps_d)))

    for i in range(len(eps_a)):
        for j in range(len(eps_d)):
            newns = NewnsAndersonAnalytical(beta = BETA, 
                                            beta_p = BETA_P / 2 / BETA, 
                                            eps_d = eps_d[j],
                                            eps_a = eps_a[i],
                                            eps = EPSILON_RANGE,
                                            fermi_energy = FERMI_ENERGY,
                                            U = U)
            newns.self_consistent_calculation()
            energy_in_eV = newns.DeltaE * 2 * BETA 
                
            energies[i,j] = energy_in_eV

    fig, ax = plt.subplots(1, 1, figsize=(8,7), constrained_layout=True) 

    # Plot contourlines
    # ax.contour(eps_d, eps_a, energies, levels=5, cmap=plt.cm.gray)
    ax.set_title(r"$\beta=%1.1f \mathregular{eV}, \beta^\prime=%1.1f \mathregular{eV}$"%(BETA, BETA_P))
    # ax.axvline(x=0, color='k', linestyle='-')
    ax.axhline(y=2.5, color='k', ls='--')
    ax.annotate('CO*', xy=(0, 2.0), color='k')
    ax.axhline(y=-5, color='k', ls='--')
    ax.annotate('C*', xy=(0, -5.5), color='k')
    ax.axhline(y=-1, color='k', ls='--')
    ax.annotate('OH*', xy=(0, -1.5), color='k')
    CS = ax.contourf(eps_d, eps_a, energies, 100, cmap=plt.cm.RdBu)
    ax.set_ylabel(r'$\epsilon_a - \epsilon_f$ (eV)')
    ax.set_xlabel(r'$\epsilon_d - \epsilon_f$ (eV)')
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('Chemisorption Energy (eV)')
    fig.savefig('output/NewnsAnderson_contour_epsa_epsd.png')





