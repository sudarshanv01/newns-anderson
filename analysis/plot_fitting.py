"""Fit the parameters for the Newns-Anderson and d-band model of chemisorption."""
import numpy as np
import json
from dataclasses import dataclass
from NewnsAnderson import NewnsAndersonNumerical
from collections import defaultdict
from scipy.optimize import minimize, least_squares, leastsq
from pprint import pprint
import matplotlib.pyplot as plt
from plot_params import get_plot_params
get_plot_params()

@dataclass
class JensNewnsAnderson:
    Vsd: list
    filling: list
    eps_d: list
    width: list

    eps_a: float

    def __post_init__(self):
        """Extra variables that are needed for the model."""
        self.eps = np.linspace(-20, 10, 20000)
        # convert all lists to numpy arrays
        self.Vsd = np.array(self.Vsd)
        self.filling = np.array(self.filling)
        self.eps_d = np.array(self.eps_d)
        self.width = np.array(self.width)

    def fit_parameters(self, params, dft_energy):
        """Fit the parameters alpha, beta, delta0."""
        alpha, beta, delta0 = params

        Vak = np.sqrt(beta) * self.Vsd

        hybridisation_energy = np.zeros(len(self.eps_d))
        for i, eps_d in enumerate(self.eps_d):
            hybridisation = NewnsAndersonNumerical(
                Vak = Vak[i],
                eps_a = self.eps_a,
                eps_d = self.eps_d[i],
                width = self.width[i],
                eps = self.eps,
                k = delta0, 
            )
            hybridisation.calculate_energy()
            hybridisation_energy[i] = hybridisation.DeltaE

        assert all(hybridisation_energy <= 0), "Hybridisation energy is negative"

        # orthonogonalisation energy
        ortho_energy = self.filling * alpha * np.sqrt(beta) * self.Vsd**2

        assert all(ortho_energy >= 0), "Orthogonalisation energy is positive"

        # Add the orthogonalisation energy to the hybridisation energy
        hybridisation_energy += ortho_energy

        self.hybridisation_energy = hybridisation_energy
        
        # Return the least square error
        return  np.sum((dft_energy - hybridisation_energy)**2)


if __name__ == '__main__':
    """Determine the fitting parameters for a particular adsorbate."""

    REMOVE_LIST = []

    # Choose an adsorbate
    eps_a = -5.0
    ADSORBATE = 'O'
    print(f"Fitting parameters for adsorbate {ADSORBATE} with eps_a {eps_a}")

    # get the width and d-band centre parameters
    data_from_dos_calculation = json.load(open('output/pdos_moments.json')) 
    data_from_energy_calculation = json.load(open('output/adsorption_energies.json'))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    # Store the parameters
    parameters = defaultdict(list)
    # Store the final DFT energies
    dft_energies = []
    metals = []

    for metal in data_from_energy_calculation[ADSORBATE]:
        if metal in REMOVE_LIST:
            continue

        # get the parameters
        width = data_from_dos_calculation[metal]['width']
        d_band_centre = data_from_dos_calculation[metal]['d_band_centre']

        parameters['width'].append(width)
        parameters['d_band_centre'].append(d_band_centre)

        adsorption_energy = data_from_energy_calculation[ADSORBATE][metal]
        dft_energies.append(adsorption_energy)
    
        Vsd = data_from_LMTO['Vsd'][metal]
        parameters['Vsd'].append(Vsd)
        filling = data_from_LMTO['filling'][metal]
        parameters['filling'].append(filling)

        metals.append(metal)

    # Fit the parameters
    fitting_function = JensNewnsAnderson(
        Vsd = parameters['Vsd'],
        filling = parameters['filling'],
        eps_d = parameters['d_band_centre'],
        width = parameters['width'],
        eps_a = eps_a,
    )
    # Make the constraints for minimize such that all parameters
    # are positive
    constraints = ( (0, None), (0, None), (0, None) )
    result = minimize(
        fitting_function.fit_parameters,
        x0 = [0.01, 1.5, 3 ],
        args = dft_energies,
        method='Nelder-Mead',
        bounds=constraints,
    )

    print(f"Finished: {result.success} with residue: {result.fun}")
    print(f"alpha = {result.x[0]}, beta = {result.x[1]}, delta_sp: {result.x[2]}")

    # Plot the Fitted and the real adsorption energies
    optimised_hyb = fitting_function.fit_parameters(result.x, dft_energies)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax.plot(dft_energies, fitting_function.hybridisation_energy, 'o',)
    # plot the parity line
    x = np.linspace(np.min(dft_energies), np.max(dft_energies), 2)
    # ax.plot(x, x, '--', color='black')
    for i, metal in enumerate(metals):
        ax.annotate(metal, (dft_energies[i], fitting_function.hybridisation_energy[i]))
    ax.set_xlabel('DFT energy (eV)')
    ax.set_ylabel('Hybridisation energy (eV)')
    fig.savefig(f'output/{ADSORBATE}_parity_plot.png')




    

    