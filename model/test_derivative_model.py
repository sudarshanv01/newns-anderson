"""Test that the derivative of the energy with eps_d is correct."""
import yaml
import json
import numpy as np
from catchemi import NewnsAndersonDerivativeEpsd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fitting_functions import get_fitted_function
from plot_params import get_plot_params
get_plot_params()

ANALYTICAL_COLOR = 'tab:blue'
NUMERICAL_COLOR = 'tab:red'

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
    diff_grid = np.linspace(-4,-2, 200)
    # Multiprecision, if needed
    use_multiprec = False 
    TEST_TYPE = 'metal_params'

    # Create a gridspec where the first two columns
    # are split into 4 rows and the third column is
    # has just one row
    fig = plt.figure(figsize=(6.75, 6), constrained_layout=True)
    gs = GridSpec(len(range_eps_fixed), 3, width_ratios=[1, 1, 1], figure=fig)
    ax1 = [ fig.add_subplot(gs[i, 0]) for i in range(len(range_eps_fixed)) ]
    ax2 = [ fig.add_subplot(gs[i, 1]) for i in range(len(range_eps_fixed)) ]
    ax1 = np.array(ax1)
    ax2 = np.array(ax2)
    ax3 = fig.add_subplot(gs[0:2, 2])
    ax4 = fig.add_subplot(gs[2:, 2])
    
    # Plot labels
    for a in ax1:
        a.set_ylabel('$\Delta^\prime$')
    a.set_xlabel('$\epsilon_d$ (eV)')
    for a in ax2:
        a.set_ylabel('$\Lambda^\prime$')
    a.set_xlabel('$\epsilon_d$ (eV)')

    ax3.set_ylabel('$E_{\mathrm{hyb}}^\prime$')
    ax3.set_xlabel('$\epsilon_d$ (eV)')

    ax4.set_ylabel('$E_{\mathrm{hyb}}$ (eV)')
    ax4.set_xlabel('$\epsilon_d$ (eV)')

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
    metal_row = 1
    Vsd_fit = metal_parameters['Vsd'][str(metal_row)]
    wd_fit = metal_parameters['width'][str(metal_row)]
    epsd_filling_fit = metal_parameters['epsd_filling'][str(metal_row)]
    kwargs = {'input_epsd':True, 'fitted_epsd_to_filling':epsd_filling_fit}
    kwargs_deriv = {'input_epsd':True, 
                    'fitted_epsd_to_filling':epsd_filling_fit, 
                    'is_derivative':True}

    # Make Vsd and wd functions
    if TEST_TYPE == 'constant':
        f_Vsd = lambda x: 2
        f_wd = lambda x: 2
        f_Vsd_p = lambda x: 0
        f_wd_p = lambda x: 0
    elif TEST_TYPE == 'metal_params':
        function_Vsd, function_Vsd_p = get_fitted_function('Vsd') 
        function_wd, function_wd_p = get_fitted_function('wd')

        f_Vsd = lambda x: function_Vsd(x, *Vsd_fit, **kwargs)
        f_Vsd_p = lambda x: function_Vsd_p(x, *Vsd_fit, **kwargs_deriv)
        f_wd = lambda x: function_wd(x, *wd_fit, **kwargs)
        f_wd_p = lambda x: function_wd_p(x, *wd_fit, **kwargs_deriv)


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
    ax3.legend(loc='best', fontsize=8)
    # ax32 = ax3.twinx()
    ax4.plot(diff_grid, numerical_hyb, label=r'Numerical', ls='-', color='k')

    fig.savefig('output/test_derivative_Delta_epsd.png')