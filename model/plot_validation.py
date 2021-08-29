"""Plot the Newns-Anderson model and compare it against the analytical solution"""

import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonModel
from plot_params import get_plot_params
get_plot_params()


def get_analytical_solution(eps_a):
    """Analytical solution for the Newns-Anderson model for V=0.5 and half the band is filled."""
    energy_ = 2 * eps_a + 1 / 2 / eps_a
    energy_ *= np.arctan(2 * eps_a)
    energy_ += 1
    energy_ /= (2 *np.pi)
    energy_ *= -1
    energy_ += 0.5 * eps_a
    return energy_

if __name__ == '__main__':
    # Define the parameters
    ADSORBATE_ENERGIES = np.linspace(-3, -2, 100)
    COUPLING = 0.5 
    ENERGY_RANGE = np.linspace(-30, 20 , 1000)
    METAL_ENERGIES = 0 

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5), constrained_layout=True)
    
    for i, eps_a in enumerate(ADSORBATE_ENERGIES):
        analytical_half_band = get_analytical_solution(eps_a)
        print(f'eps_a: {eps_a}')
        print(f'Analytical solution for half filled band and V = 0.5 is {analytical_half_band}')
        n = NewnsAndersonModel(
            eps_a=eps_a, 
            coupling=COUPLING,
            eps_d=METAL_ENERGIES, 
            eps=ENERGY_RANGE)
        n.calculate()
        print(f'Numerical solution {n.energy}')
        ax.plot(-1*analytical_half_band, n.energy, 'o', color='tab:blue')
    ax.set_xlabel('Energy - Analytical (eV)')
    ax.set_ylabel('Energy - Numerical (eV)')
    fig.savefig('output/newns_anderson_validation.png')

    