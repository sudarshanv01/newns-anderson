"""Determine the fitted quantities for the delta function."""
import json
import pickle
import collections
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from glob import glob
from plot_params import get_plot_params
get_plot_params()

class FitSemiEllipse:
    """Fit a semi-ellipse to the data."""
    def __init__(self, delta, eps):
        """Initialize the class with delta and eps values."""
        self.delta = delta
        self.eps = eps

    def fit(self):
        """Fit the semi-ellipse."""
        def func(eps, beta, beta_p, eps_d):
            """Define the semi-ellipse, the only variation from the 
            Newns paper is that the center of the d-band might be moved.
            The energies from the DFT calculations are in eV, so all quantities
            from the Newns paper must be multiplied by 2beta."""
            eps_wrt_d = eps - eps_d
            delta_fit = 2 * beta_p**2 * (2*beta - eps_wrt_d**2)**0.5
            delta_fit = np.nan_to_num(delta_fit)
            return delta_fit

        popt, pcov = curve_fit(func, self.eps, self.delta)

        # Store the delta after fit
        self.delta_fit = func(self.eps, *popt)
        self.beta = popt[0]
        self.beta_p = popt[1]
        self.eps_d = popt[2]

        return popt, pcov

if __name__ == '__main__':
    """Fitting the Delta function to a semi-ellipse."""
    # Load the data
    files = glob('output/delta/*.pkl')
    
    # Store the results to be make into a json
    results = collections.defaultdict(dict)

    # Loop over the files for all the metals
    for f in files:
        # Load the data
        data = pickle.load(open(f, 'rb'))

        # Perform the semi-ellsipse fit
        semiellipse = FitSemiEllipse(delta=data['Delta'], eps=data['eigenval_metal'])
        popt, pcov = semiellipse.fit() 

        # Store the results
        results[data['metal']] = { 'beta': popt[0], 'beta_p': popt[1], 'eps_d': popt[2] }

        # Each metal has a figure with the fitted quantities
        figa, axa = plt.subplots(1, 1, figsize=(6,4), constrained_layout=True)

        axa.plot(semiellipse.eps, semiellipse.delta, '-', label='$\Delta$', lw=3) 
        axa.plot(semiellipse.eps, semiellipse.delta_fit, '--', label='$\Delta$ fit', color='black', lw=3)
        axa.set_xlabel('$\epsilon$')
        axa.set_ylabel('$\Delta$')

        # Find the eigenvalues closest to the Fermi energy and report 
        # them with respect to the Fermi level as eps_a
        eigenval_ads_occupied = [a for a in data['eigenval_ads'] if a < data['fermi_energy']]
        # pick the closest eigenvalue to the Fermi energy
        eps_a = min(eigenval_ads_occupied, key=lambda x:abs(x-data['fermi_energy']))
        results[data['adsorbate']][data['metal']]['eps_a'] = eps_a
        

        # Save the figure
        figa.savefig('output/delta/{}_{}_delta.png'.format(a=data['metal'], b=data['adsorbate']))
    
    # Save the results to a json
    with open('output/delta/fit_results.json', 'w') as f:
        json.dump(results, f, indent=4)



