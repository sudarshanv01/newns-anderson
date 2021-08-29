"""Plot the Newns-Anderson model for some specific instances."""

import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonModel
from plot_params import get_plot_params
get_plot_params()


if __name__ == '__main__':
    # Define the parameters
    ADSORBATE_ENERGIES = np.linspace(-1, 1, 10) 
    COUPLING = [4, 6]
    ENERGY_RANGE = np.linspace(-20, 10 , 1000)
    METAL_ENERGIES = [-5, -3, -1, 0] 
    PLOT_IDEAL_DOS = True

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    
    for i, eps_d in enumerate(METAL_ENERGIES):
        for j, V in enumerate(COUPLING):
            energies = np.zeros(len(ADSORBATE_ENERGIES))
            figd, axd = plt.subplots(1, len(ADSORBATE_ENERGIES), figsize=(4*len(ADSORBATE_ENERGIES), 6), constrained_layout=True)
            all_energies = []
            for k, eps_a in enumerate(ADSORBATE_ENERGIES):
                n = NewnsAndersonModel(
                    eps_a=eps_a, 
                    coupling=V,
                    eps_d=eps_d, 
                    eps=ENERGY_RANGE)
                n.calculate()
                if PLOT_IDEAL_DOS:
                    axd[k].plot(n.Delta, n.eps, color='k', lw=3)
                    axd[k].plot(n.Lambda, n.eps, color='tab:blue')
                    axd[k].plot(n.eps-n.eps_a, n.eps, color='tab:red')
                    axd[k].axhline(n.eps_d, color='k', ls='--')
                    axd[k].axhline(n.eps_a, color='tab:green', ls='--')
                    axd[k].plot(n.na, n.eps, color='tab:green', lw=3)
                    xlim = [-2*np.max(n.Delta), 2*np.max(n.Delta)]
                    axd[k].set_xlim(xlim)
                all_energies.append([n.eps_a, n.energy])
            if PLOT_IDEAL_DOS:
                figd.savefig('output/NewnsAnderson_eps_d_%1.2f_%1.2f.png' % (eps_d, V))
            all_d, all_hyb = np.array(all_energies).T
            all_d_sorted = all_d[np.argsort(all_d)]
            all_hyb_sorted = all_hyb[np.argsort(all_d)]
            ax.plot(all_d_sorted, all_hyb_sorted, 'o-', label=f'$\epsilon_d$ = {eps_d} eV, $V$ = {V} eV')
    ax.set_xlabel('$\epsilon_a$ (eV)')
    ax.set_ylabel('Hybridisation Energy (eV)')
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    fig.savefig('output/hybridisation_energy_vary_eps_a.png')