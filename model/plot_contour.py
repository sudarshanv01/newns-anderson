"""Plot the results from the Newns-Anderson model for different combination of parameters."""


import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonModel
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':

    eps_a = np.linspace(-10, 5, 100)
    eps_d = np.linspace(-6, 2, 100)
    COUPLING_ELEMENT = 2
    eps = np.linspace(-20, 20, 500)

    eps_a, eps_d = np.meshgrid(eps_a, eps_d)
    energies = np.zeros(eps_a.shape)

    for i in range(len(eps_a)):
        for j in range(len(eps_d)):
            parameters = {'eps_a':eps_a[i,j], 'eps_d':eps_d[i,j], 'coupling':COUPLING_ELEMENT, 'eps':eps}
            n = NewnsAndersonModel(**parameters)
            n.calculate()
            energies[i,j] = n.energy

    fig, ax = plt.subplots(1, 1, figsize=(8,7), constrained_layout=True) 

    CS = ax.contourf(eps_a, eps_d, energies, 100, cmap='viridis')
    ax.set_ylabel(r'$\epsilon_d$ (eV)')
    ax.set_xlabel(r'$\epsilon_a$ (eV)')
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('Hybridisation Energy (eV)')
    fig.savefig('output/newns_anderson_epsa_epsd.png')





