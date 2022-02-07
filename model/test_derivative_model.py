"""Test that the derivative of the energy with eps_d is correct."""
import yaml
import json
import numpy as np
from catchemi import NewnsAndersonDerivativeEpsd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from plot_params import get_plot_params
get_plot_params()

ANALYTICAL_COLOR = 'tab:blue'
NUMERICAL_COLOR = 'tab:red'

def function_Vsdsq(x, a):
    """Variation of coupling element with d-band centre."""
    return a / x

def function_Vsdsq_p(x, a):
    """Derivative of Vsd_p with respect to d-band centre."""
    return - a / x**2

def function_wd(x, a, b, c):
    """Variation of width with d-band centre."""
    return b - a * ( x - c) **2

def function_wd_p(x, a, b, c):
    """Derivative of wd_p with respect to d-band centre."""
    return - 2 * a * (x - c) 

if __name__ == '__main__':
    """Check that the derivative and its components is
    computed correctly in the NewnsAnderson code."""

    # Set up the parameters
    eps = np.linspace(-30, 15, 1000)
    # Value at which the derivative of Delta
    # and Lambda are checked.
    range_eps_fixed = [-4, -3, -2, -1]
    # Grid on which numerical differentiation
    # is performed and analytical derivatives
    # are reported.
    diff_grid = np.linspace(-4,-1, 200)
    # Multiprecision, if needed
    use_multiprec = False 
    TEST_TYPE = 'metal_params'

    # Create a gridspec where the first two columns
    # are split into 4 rows and the third column is
    # has just one row
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs = GridSpec(len(range_eps_fixed), 3, width_ratios=[1, 1, 1], figure=fig)
    ax1 = [ fig.add_subplot(gs[i, 0]) for i in range(len(range_eps_fixed)) ]
    ax2 = [ fig.add_subplot(gs[i, 1]) for i in range(len(range_eps_fixed)) ]
    ax1 = np.array(ax1)
    ax2 = np.array(ax2)
    ax3 = fig.add_subplot(gs[:, 2])

    # Read in adsorbate parameters
    COMP_SETUP = yaml.safe_load(stream=open('../analysis/chosen_group.yaml', 'r'))['group'][0]
    with open(f"../analysis/output/O_parameters_{COMP_SETUP}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"../analysis/output/C_parameters_{COMP_SETUP}.json", 'r') as f:
        c_parameters = json.load(f)
    with open(f"../analysis/output/fitting_metal_parameters_{COMP_SETUP}.json", 'r') as f:
        metal_parameters = json.load(f)
    
    # Choose adsorbate parameters
    eps_a = o_parameters['eps_a']
    alpha = o_parameters['alpha']
    beta = o_parameters['beta']
    Delta0_mag = o_parameters['delta0']
    constant_offset = o_parameters['constant_offset']

    # Choose metal parameters
    metal_row = 0
    Vsdsq_fit = metal_parameters['Vsdsq_epsd'][str(metal_row)]
    wd_fit = metal_parameters['width_epsd'][str(metal_row)]
    # Make Vsd and wd functions
    if TEST_TYPE == 'constant':
        f_Vsd = lambda x: 2
        f_wd = lambda x: 2
        f_Vsd_p = lambda x: 0
        f_wd_p = lambda x: 0
    elif TEST_TYPE == 'metal_params':
        f_Vsd = lambda x: np.sqrt(function_Vsdsq(x, *Vsdsq_fit))
        f_Vsd_p = lambda x: np.sqrt(function_Vsdsq_p(x, *Vsdsq_fit))
        f_wd = lambda x: function_wd(x, *wd_fit)
        f_wd_p = lambda x: function_wd_p(x, *wd_fit)


    for i, eps_fixed in enumerate(range_eps_fixed):
        derivative = NewnsAndersonDerivativeEpsd(f_Vsd=f_Vsd,f_Vsd_p=f_Vsd_p,
                                                eps_a=eps_a, eps=eps, f_wd=f_wd,
                                                f_wd_p=f_wd_p, diff_grid=diff_grid,
                                                alpha=alpha, beta=beta, Delta0_mag=Delta0_mag,
                                                constant_offset=constant_offset,
                                                use_multiprec=use_multiprec)

        numerical_Delta_deriv = derivative.get_Delta_prime_epsd_numerical(eps_fixed) 
        analytical_Delta_deriv = derivative.get_Delta_prime_epsd(eps_fixed)
        numerical_Lambda_deriv = derivative.get_Lambda_prime_epsd_numerical(eps_fixed)
        analytical_Lambda_deriv = derivative.get_Lambda_prime_epsd(eps_fixed)

        # Plot the numerical and analytical derivatives
        ax1[i].plot(diff_grid, analytical_Delta_deriv, color=ANALYTICAL_COLOR, label=r'Analytical $\Delta$')
        ax1[i].plot(diff_grid, numerical_Delta_deriv, color=NUMERICAL_COLOR, ls='--', label=r'Numerical $\Delta$')

        ax2[i].plot(diff_grid, analytical_Lambda_deriv, color=ANALYTICAL_COLOR, label=r'Analytical $\Lambda$')
        ax2[i].plot(diff_grid, numerical_Lambda_deriv, color=NUMERICAL_COLOR, label=r'Numerical $\Lambda$', ls='--')

    # Confirm that the energy is also correct
    numerical_hyb_deriv, numerical_hyb = derivative.get_hybridisation_energy_prime_epsd_numerical(get_hyb=True)
    analytical_hyb_deriv = derivative.get_hybridisation_energy_prime_epsd()
    ax3.plot(diff_grid, analytical_hyb_deriv, label=r'Analytical', color=ANALYTICAL_COLOR)
    ax3.plot(diff_grid, numerical_hyb_deriv, label=r'Numerical', ls='--', color=NUMERICAL_COLOR)
    ax32 = ax3.twinx()
    ax32.plot(diff_grid, numerical_hyb, label=r'Numerical', ls='-', color='k')

    fig.savefig('output/test_derivative_Delta_epsd.png')