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
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class FitSemiEllipse:
    """Fit a semi-ellipse to the data."""
    def __init__(self, delta, eps, fermi_energy=0):
        """Initialize the class with delta and eps values."""
        self.delta = delta
        self.eps = eps
        self.fermi_energy = fermi_energy

    def fit(self):
        """Fit the semi-ellipse."""
        def func(eps, beta, beta_p, eps_d):
            """Define the semi-ellipse, the only variation from the 
            Newns paper is that the center of the d-band might be moved.
            The energies from the DFT calculations are in eV, so all quantities
            from the Newns paper must be multiplied by 2beta."""
            eps_wrt_d = eps - eps_d
            delta_fit = 2 * beta_p**2 * (beta - eps_wrt_d**2)**0.5
            delta_fit = np.nan_to_num(delta_fit)
            return delta_fit
        delta_guess = self.delta
        delta_guess = np.array(delta_guess)
        self.eps = np.array(self.eps)
        # Set all elements of delta which lie above the Fermi energy to 0.
        delta_guess[self.eps > self.fermi_energy] = 0
        # get the initial guesses based on the moments of the distribution
        center, width = get_distribution_moment(self.eps, delta_guess, (1, 2))
        beta_guess = 4 * np.sqrt(width)
        eps_d_guess = center
        beta_p_guess = np.max(self.delta)

        # Fit the curve
        popt, pcov = curve_fit(func, self.eps, self.delta, p0=[beta_guess, beta_p_guess, eps_d_guess])

        # Store the delta after fit
        self.delta_fit = func(self.eps, *popt)
        self.beta = np.sqrt(popt[0]) # in the fit we are fitting for 2beta in units of 2beta
        self.beta_p = popt[1]
        self.eps_d = popt[2]

        return popt, pcov

if __name__ == '__main__':
    """Fitting the Delta function to a semi-ellipse."""
    # Load the data
    files = glob('output/delta/*_LCAO_lcao_data.pkl')

    # Ignore this elements
    IGNORE_ELEMENTS = [] #['Ni', 'Cu']
    
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
        semiellipse = FitSemiEllipse(delta=data['Delta'], eps=data['eigenval_metal'], fermi_energy=data['fermi_energy'])
        popt, pcov = semiellipse.fit() 

        # Store the results with all energies referenced to the Fermi level
        results[data['adsorbate']][data['metal']] = { 'beta': semiellipse.beta, 
                                                      'beta_p': semiellipse.beta_p, 
                                                      'eps_d': semiellipse.eps_d - data['fermi_energy'], 
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
        # Store the quantity in results
        results[data['adsorbate']][data['metal']]['eps_a'] = eps_a - data['fermi_energy']
        results[data['adsorbate']][data['metal']]['Vak'] = Vak
        
        # Save the figure
        figa.savefig('output/delta/{a}_{b}_delta.png'.format(a=data['metal'], b=data['adsorbate']))
        plt.close(figa)
    
    # Save the results to a json
    with open('output/delta/fit_results.json', 'w') as f:
        json.dump(results, f, indent=4)



