"""Plot the variation of energy with the adsorbate energy based on the numerical class of the 
Newns-Anderson model."""
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonNumerical, NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()
if __name__ == '__main__':
    """Plot the variation of the chemisorption energy against the d-band center."""

    WIDTHS = [ 2, 4, 6 ]
    K_VALUE = 0.0
    eps_as = np.linspace(-4, 1, 100)
    EPS_D = [-4, -2, 0]
    EPS_RANGE = np.linspace(-20, 20, 200000)
    Vak = 1

    fig, ax = plt.subplots(1, len(EPS_D), figsize=(5*len(EPS_D), 4.5), constrained_layout=True)

    for eps_index, eps_d in enumerate(EPS_D):
        for i, width in enumerate(WIDTHS):
            energies = []
            for j, eps_a in enumerate(eps_as):
                newns = NewnsAndersonNumerical(
                    width = width,
                    Vak = Vak, 
                    eps_a = eps_a,
                    eps_d = eps_d,
                    eps = EPS_RANGE,
                    k = K_VALUE,
                )
                newns.calculate_energy()
                
                energies.append( newns.DeltaE )
            ax[eps_index].plot(eps_as, energies, label=f'width: {width} eV', lw=2)

    for i, a in enumerate(ax):
        a.legend(loc='best')
        a.set_xlabel('$\epsilon_a$ (eV)')
        if i == 0:
            a.set_ylabel('$\Delta E_{\mathregular{d-hyb}}$ (eV)')
        a.set_title(f'$\epsilon_d = {EPS_D[i]}$ eV')
    fig.savefig('output/NewnsAndersonNumerical_vary_epsa.png')
