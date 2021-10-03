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

    WIDTHS = [ 1, 1.5, 2.5, 3.5 ]
    eps_ds = np.linspace(-10, 5.5, 100)
    EPS_A = -5
    EPS_RANGE = np.linspace(-20, 20, 4000)

    for i, width in enumerate(WIDTHS):
        energies = []
        for j, eps_d in enumerate(eps_ds):
            newns = NewnsAndersonAnalytical(
                            beta=width,
                            eps_a=EPS_A,
                            eps_d=eps_d,
                            eps=EPS_RANGE,
                            beta_p= 1,
                            U=0.0,
                            fermi_energy=0.0
            )

            newns.self_consistent_calculation()
            if newns.has_localised_occupied_state_positive:
                ax.plot(eps_d, newns.DeltaE, 'v', color='k', alpha=0.25)
            # if newns.has_localised_occupied_state_negative:
            #     ax.plot(eps_d, newns.DeltaE, 'o', color='k',)
            
            energies.append( newns.DeltaE )
        
        ax.plot(eps_ds, energies, label=f'width: {width} eV', lw=3)

    ax.legend(loc='best')
    ax.set_xlabel('$\epsilon_d$ (eV)')
    ax.set_ylabel('$\Delta E$ (eV)')
    fig.savefig('output/NewnsAndersonAnalytical_vary_epsd.png')
