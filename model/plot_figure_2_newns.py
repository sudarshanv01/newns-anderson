"""Recreate Figure 2 of the Newns paper."""

import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':
    """Recreate Figure 2 of the Newns paper.
    The units of the beta_prime and epsilon_sigma are -2beta
    This plot is a colormap of the the number of localised states
    that exist for a given value of beta_prime and epsilon_sigma.
    """
    # Parameters to recreate the figure
    EPSILON_RANGE = np.linspace(-2.5, 2.5, 200) # in units of -2beta 
    BETA_PRIME = np.linspace(-1.25, 1.25, 200) # in units of -2beta 
    EPSILON_SIGMA = np.linspace(-1.5, 1.5, 200) # in units of -2beta
    EPSILON_D = 0 

    # Create the figure
    fige, axe = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    # Store the number of solutions in this array
    solutions = np.zeros((len(BETA_PRIME), len(EPSILON_SIGMA)))

    for i, beta_p in enumerate(BETA_PRIME):
        print('beta prime = {} (units: 2beta)'.format(beta_p))
        # Choose only the right epsilon_{sigma}
        for j, _epsilon_sigma in enumerate(EPSILON_SIGMA):

            # All energy quantities in eV
            epsilon_sigma = _epsilon_sigma * 2 * beta_p 
            epsilon_d = EPSILON_D 
            epsilon_range = EPSILON_RANGE * 2 * beta_p

            analytical = NewnsAndersonAnalytical(beta_p=beta_p,
                                                beta=beta_p, 
                                                eps_sigma=epsilon_sigma, 
                                                eps_d=epsilon_d, 
                                                eps=epsilon_range)
            # Number can only go up
            solutions_ = 0.0 
            # now try to figure out the other solutions
            if analytical.eps_l_sigma_pos is not None:
                solutions_ += 1
            if analytical.eps_l_sigma_neg is not None:
                solutions_ += 1
            if analytical.eps_l_sigma_pos == None and analytical.eps_l_sigma_neg == None:
                if analytical.has_complex_root:
                    solutions_ += 0
                else:
                    solutions_ += 1
            
            solutions[i,j] = solutions_

    solutions = np.array(solutions) 
    cplot = axe.imshow(solutions.T, cmap='viridis', 
        extent=[min(BETA_PRIME), max(BETA_PRIME), min(EPSILON_SIGMA), max(EPSILON_SIGMA)])
    # plot a colorbar for cplot
    cbar = fige.colorbar(cplot, ax=axe)
    axe.set_xlabel(r"$\beta'$ (2$\beta$)")
    axe.set_ylabel(r'$\epsilon_{\sigma}$ (2$\beta$)')
    fige.savefig('output/figure_2_newns.png')
            

            

