"""Determine the fitted quantities for the delta function."""
import json
import pickle
import collections
import warnings
from pprint import pprint
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from glob import glob
from plot_params import get_plot_params
from ase.dft import get_distribution_moment
get_plot_params()

class FitSemiEllipse:
    """Fit a semi-ellipse to the data."""
    def __init__(self, delta, eps, fermi_energy=0):
        """Initialize the class with delta and eps values."""
        self.delta = np.array(delta)
        self.eps = np.array(eps)
        self.fermi_energy = fermi_energy

    def fit(self):
        """Fit the semi-ellipse."""
        def func(eps, eps_d, width, Vak):
            """Define the semi-ellipse, the only variation from the 
            Newns paper is that the center of the d-band might be moved.
            The energies from the DFT calculations are in eV, so all quantities
            from the Newns paper must be multiplied by 2beta."""
            eps_rel = ( eps - eps_d ) / width
            delta_fit = np.zeros(len(eps))
            # Find elements in eps_rel that have abs values less than 1
            index = np.where(np.abs(eps_rel) < 1)[0]
            delta_fit[index] = np.pi * Vak**2 * (1 - eps_rel[index]**2)**0.5 
            delta_fit /= width
            delta_fit *= 2
            return delta_fit

        delta_guess = self.delta
        # get the initial guesses based on the moments of the distribution
        center, width = get_distribution_moment(self.eps, delta_guess, (1, 2))
        # Guess the Vak based on the maximum value of Delta
        Vak_guess = 1. # np.max(delta_guess)

        # Fit the curve
        popt, pcov = curve_fit(func, self.eps, self.delta, p0=[center, width, Vak_guess])

        # Store the delta after fit
        self.delta_fit = func(self.eps, *popt)
        
        self.eps_d = popt[0]
        self.width = popt[1]
        self.Vak = popt[2]

        return popt, pcov

if __name__ == '__main__':
    """Fitting the Delta function to a semi-ellipse."""

    # Load the data
    files = glob('output/delta/*_lcao_data.pkl')

    # Ignore this elements
    IGNORE_ELEMENTS = []
    
    # Store the results to be make into a json
    results = collections.defaultdict(dict)

    # Loop over the files for all the metals
    for f in files:

        # Load the data
        data = pickle.load(open(f, 'rb'))

        # If the element is in the ignore list, skip it
        if data['metal'] in IGNORE_ELEMENTS:
            print(f"Skipping {data['metal']}")
            continue

        # Perform the semi-ellsipse fit
        semiellipse = FitSemiEllipse(delta=data['Delta'], 
                                     eps=data['eigenval_metal'],
                                     fermi_energy=data['fermi_energy'])
        popt, pcov = semiellipse.fit()

        # Store the results with all energies referenced to the Fermi level
        results[data['adsorbate']][data['metal']] = { 'eps_d': semiellipse.eps_d - data['fermi_energy'], 
                                                      'width': semiellipse.width, 
                                                      'Vak_fit': semiellipse.Vak,
                                                    }

        # Each metal has a figure with the fitted quantities
        figa, axa = plt.subplots(1, 1, figsize=(8,5), constrained_layout=True)

        axa.plot(semiellipse.eps, semiellipse.delta, '-',  lw=3, color='tab:blue', label=r'From LCAO') 
        axa.plot(semiellipse.eps, semiellipse.delta_fit, '--', color='tab:red', lw=3, label=r'Fit $\Delta$')
        axa.axvline(x=data['fermi_energy'], color='black', ls='--', lw=2, label='Fermi Level')
        axa.set_xlabel('$\epsilon$ (eV)')
        axa.set_ylabel('$\Delta$ (eV)')
        axa.legend(loc='best')

        # Find the eigenvalues closest to the Fermi energy and report 
        # them with respect to the Fermi level as eps_a
        eigenval_ads_occupied = [a for a in data['eigenval_ads'] if a < data['fermi_energy']]
        # pick the closest eigenvalue to the Fermi energy
        eps_a = min(eigenval_ads_occupied, key=lambda x:abs(x-data['fermi_energy']))
        # Get the Vak for the eigenstate closest to the Fermi energy
        all_Vak = data['Vak']
        Vak = all_Vak[np.argmin(np.abs(np.array(data['eigenval_ads']) -eps_a))]
        all_Sak = data['Sak']
        Sak = all_Sak[np.argmin(np.abs(np.array(data['eigenval_ads']) -eps_a))]
        # Store the quantity in results
        results[data['adsorbate']][data['metal']]['eps_a'] = eps_a - data['fermi_energy']
        results[data['adsorbate']][data['metal']]['Vak_calc'] = Vak
        results[data['adsorbate']][data['metal']]['Sak_calc'] = Sak
        
        # Save the figure
        figa.savefig('output/delta/fitting_images/{a}_{b}_delta.png'.format(a=data['metal'], b=data['adsorbate']))
        plt.close(figa)
    
    # Save the results to a json
    with open('output/fit_results_lcao.json', 'w') as f:
        json.dump(results, f, indent=4)



