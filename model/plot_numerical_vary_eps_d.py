"""Plot the variation of energy with the d-band center based on the numerical class of the 
Newns-Anderson model."""
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonNumerical, NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()
if __name__ == '__main__':
    """Plot the variation of the chemisorption energy against the d-band center."""

    # Create a contour plot of the energy matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    WIDTHS = [ 2, 4, 6 ]
    eps_ds = np.linspace(-6, 5.5, 100)
    EPS_A = -5
    EPS_RANGE = np.linspace(-20, 20, 40000)

    for i, width in enumerate(WIDTHS):
        energies = []
        for j, eps_d in enumerate(eps_ds):
            newns = NewnsAndersonNumerical(
                width = width,
                Vak = 1, 
                eps_a = EPS_A,
                eps_d = eps_d,
                eps = EPS_RANGE,
            )
            newns.calculate_energy()
            
            energies.append( newns.DeltaE )
        
        ax.plot(eps_ds, energies, label=f'width: {width} eV', lw=3)

    ax.legend(loc='best')
    ax.set_xlabel('$\epsilon_d$ (eV)')
    ax.set_ylabel('$\Delta E$ (eV)')
    fig.savefig('output/NewnsAndersonNumerical_vary_epsd.png')
