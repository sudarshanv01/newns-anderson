"""Plot scaling of the Newns Anderson Model based on adsorbates of two eps_a."""

import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonNumerical, NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()
if __name__ == '__main__':
    """Plot the scaling line based on the Newns-Anderson model."""

    # Scale these values
    EPS_A_VALUES = [ -1, -5 ]

    # All the parameters for scaling
    WIDTHS = 4
    K_VALUE =  0.0
    EPS_RANGE = np.linspace(-20, 20, 200000)
    eps_ds = np.linspace(-6, 0, 40)
    Vak_VALUES = [1, 2, 3]
    Delta_0 = 2

    # Figure for scaling
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    # Generate a sequence of linestyles
    linestyles = ['-', '--', '-.', ':']

    # Generate a sequence of markers
    markers = ['o', 'v', 's', 'p', '*', 'h', 'H', 'D', 'd']

    for i, Vak in enumerate(Vak_VALUES):
        energies_epsa = []
        for eps_a in EPS_A_VALUES:
            energies_epsd = []
            for j, eps_d in enumerate(eps_ds):
                # Run the Newns calculation
                newns = NewnsAndersonNumerical(
                    width = WIDTHS,
                    Vak = Vak, 
                    eps_a = eps_a,
                    eps_d = eps_d,
                    eps = EPS_RANGE,
                    k = Delta_0,
                )
                newns.calculate_energy()
                energies_epsd.append( newns.DeltaE )

            energies_epsa.append( energies_epsd )
        
        cax = ax.scatter(energies_epsa[0], energies_epsa[1], c=eps_ds, marker=markers[i], label=r'$V_{ak}$' + f' $= {Vak}$ eV', cmap='coolwarm') 

    ax.set_ylabel('$\Delta E (\epsilon_a=%1.1f$) / eV'%EPS_A_VALUES[1],)
    ax.set_xlabel('$\Delta E (\epsilon_a=%1.1f$) / eV'%EPS_A_VALUES[0],)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.set_ylabel('$\epsilon_d$ (eV)')
    ax.legend(loc='best')

    fig.savefig('output/NewnsAndersonNumerical_scaling.png')