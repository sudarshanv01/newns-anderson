"""Plot the projected density of states from the DFT calculations."""
import sys
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from ase.dft import get_distribution_moment
import numpy as np
import scipy
from scipy.integrate import simps
from scipy import odr
from scipy import signal
from scipy.optimize import curve_fit
from scipy import interpolate
from pprint import pprint
from dataclasses import dataclass
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical
from plot_params import get_plot_params
get_plot_params()

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 
METALS = [FIRST_ROW, SECOND_ROW, THIRD_ROW]

def semi_ellipse(energies, eps_d, width, amp):
    width = np.abs(width)
    energy_ref = ( energies - eps_d ) / width
    delta = np.zeros(len(energies))
    for i, eps_ in enumerate(energy_ref):
        if np.abs(eps_) < 1:
            delta[i] = amp * (1 - eps_**2)**0.5
    return delta

def normalise_na_quantities(quantity, energies):
    """Make sure that the quantity provided integrates to 1."""
    return quantity / np.trapz(quantity, x=energies)


@dataclass
class FittingVak:
    """Fit Vak based on the features of the (sp) projected
    density of states of the adsorbates."""
    rho_d: list
    eps_a: float
    Delta0: float= 0.1

    def fitting_function(self, args, energies, store_variables=False):
        """Fitting function to determine Vak and Delta0."""
        Vak = np.abs(args[0])

        # Create Delta from rho_d
        Delta = np.pi * Vak**2 * self.rho_d + self.Delta0
        # Find the Hilbert transform of Delta
        Lambda = np.imag(signal.hilbert(Delta))
        # Find the density of states from the Newns-Anderson model
        numerator = Delta 
        denominator = (energies - self.eps_a - Lambda)**2 + Delta**2
        rho_a = numerator / denominator / np.pi

        # If store variables is True, store the variables
        if store_variables:
            self.Lambda = Lambda
            self.Delta = Delta
            self.energy_diff = energies - self.eps_a

        return rho_a

def get_semi_ellipse_dos(Vak, eps_a, eps_d, width, 
                         eps, Delta0_mag, eps_sp_min=-20, eps_sp_max=20):
    """Function to interactively plot the Newns-Anderson model density of states."""

    hybridisation = NewnsAndersonNumerical(
        Vak = Vak,
        eps_a = eps_a,
        eps = eps,
        width = width,
        eps_d = eps_d,
        Delta0_mag = Delta0_mag,
        eps_sp_max = eps_sp_max,
        eps_sp_min = eps_sp_min,
    )
    # Run the calculation
    hybridisation.calculate_energy()
    hybridisation.calculate_occupancy()


    # Get the Newns-Anderson model outputs
    Delta = hybridisation.get_Delta_on_grid()
    Lambda = hybridisation.get_Lambda_on_grid()
    eps_diff = hybridisation.get_energy_diff_on_grid()
    pdos = hybridisation.get_dos_on_grid()

    return pdos

if __name__ == "__main__":
    """Plot the pdos for the metal and of the adsorbates 
    from a DFT calculation and compare the result with 
    the fitted Newns-Anderson Delta."""
    # Choice of functional
    FUNCTIONAL = 'PBE_scf_cold_smearing_0.2eV' 
    
    # Adsorbates to plot 
    adsorbates = {'O':-5, 'C':-1}
    # Store initial guess for Delta, Vak
    initial_guess_adsorbate = {'O':[1.0, 0.1], 'C':[1.0, 0.5]}

    # Parameters to generate initial guesses
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    # Get projected density of states from a DFT calculation
    with open(f'output/pdos_{FUNCTIONAL}.json', 'r') as handle:
        pdos_data = json.load(handle)

    # Plot two figures, one for the metal density of
    # states compared with Delta, second for the C (sp)
    # states and third for the O (sp) states.
    fig, ax = plt.subplots(len(METALS), len(METALS[0]), 
                           figsize=(16,12), sharey=True,
                           constrained_layout=True)
    # Adsorbate pdos together
    figa, axa = plt.subplots(len(METALS), len(METALS[0]), 
                           figsize=(16,12), sharey=True,
                           constrained_layout=True)

    # Store the moments to plot later
    moments = defaultdict(dict)
    # Store the fitted Vak values to use later
    filling_data = defaultdict(dict)
    # Store the fitted Delta0 values to use later
    Delta0_data = defaultdict(dict)
    # Store the fitted Vak values to use later
    Vak_data = defaultdict(dict)

    # Keep track of which elements are computed
    used_ij = []

    for metal in pdos_data['slab']:
        # Get all the pdos
        try:
            energies, pdos, pdos_sp = pdos_data['slab'][metal]
        except ValueError:
            energies, pdos = pdos_data['slab'][metal]
            pdos_sp = np.zeros(len(energies))

        try: 
            energies_c, pdos_c = pdos_data['C'][metal]
            energies_o, pdos_o = pdos_data['O'][metal]
        except KeyError:
            continue

        # make pdos and energies into numpy arrays
        energies = np.array(energies)
        pdos = np.array(pdos)
        pdos_sp = np.array(pdos_sp)
        energies_c = np.array(energies_c)
        pdos_c = np.array(pdos_c)
        energies_o = np.array(energies_o)
        pdos_o = np.array(pdos_o)

        # in case the maximum value of energies is
        # less than +5 eV, pad the pdos with 0's
        if energies[-1] < 5:
            assert energies[0] < energies[-1]
            energy_spacing = energies[1] - energies[0]
            extra_zeros = np.arange(energies[-1], 5, energy_spacing)
            if len(extra_zeros) > 0:
                energies = np.append(energies, extra_zeros)
                energies_o = np.append(energies_o, extra_zeros)
                energies_c = np.append(energies_c, extra_zeros)

                pdos = np.concatenate((pdos, np.zeros(len(extra_zeros)))) 
                pdos_c = np.concatenate((pdos_c, np.zeros(len(extra_zeros)))) 
                pdos_o = np.concatenate((pdos_o, np.zeros(len(extra_zeros)))) 

        # Normalise the pdos for later fitting
        rho_d = np.pi * normalise_na_quantities(pdos, energies)
        assert np.round(np.trapz(rho_d, x=energies),1) == np.round(np.pi,1)
        pdos_c = normalise_na_quantities(pdos_c, energies_c)
        assert np.round(np.trapz(pdos_c, x=energies_c),1) == 1.0
        pdos_o = normalise_na_quantities(pdos_o, energies_o)
        assert np.round(np.trapz(pdos_o, x=energies_o),1) == 1.0


        # Interpolate the oxygen and carbon pdos
        func_c = interpolate.interp1d(energies_c, pdos_c, fill_value='extrapolate') 
        func_o = interpolate.interp1d(energies_o, pdos_o, fill_value='extrapolate')
        pdos_c_interp = func_c(energies)
        pdos_o_interp = func_o(energies)

        # Get the Hilbert transform of the pdos
        Lambda = np.imag(signal.hilbert(pdos))

        # Decide on the index based on the metal
        if metal in METALS[0]:
            i = 0
            j = METALS[0].index(metal)
        elif metal in METALS[1]:
            i = 1
            j = METALS[1].index(metal)
        elif metal in METALS[2]:
            i = 2
            j = METALS[2].index(metal)
        else:
            raise ValueError('Metal not in chosen list of metals.')

        # If we have gotten so far, there is data to plot
        # the metal.
        used_ij.append((i,j))

        # Plot the dft projected density of states
        ax[i,j].plot(pdos, energies, color='k')
        ax[i,j].fill_between(pdos, energies, color='tab:grey', alpha=0.2)
        # Plot the Hilbert transform
        ax[i,j].plot(Lambda, energies, color='tab:green')
        # Plot the p projected density of states for the adsorbates 
        axa[i,j].plot(pdos_o, energies_o, color='tab:red')
        axa[i,j].fill_between(pdos_o, energies_o, color='tab:red', alpha=0.2)
        axa[i,j].plot(pdos_c, energies_c, color='tab:blue')
        axa[i,j].fill_between(pdos_c, energies_c, color='tab:blue', alpha=0.2)
        ax[i,j].plot(pdos_o, energies_o, color='tab:red')
        ax[i,j].fill_between(pdos_o, energies_o, color='tab:red', alpha=0.2)
        ax[i,j].plot(pdos_c, energies_c, color='tab:blue')
        ax[i,j].fill_between(pdos_c, energies_c, color='tab:blue', alpha=0.2)

        # Magnify the region between -5, -20 eV as an inset
        axins = axa[i,j].inset_axes([0.5, 0., 0.47, 0.47])
        axins.plot(pdos_c, energies_c, color='tab:blue')
        axins.fill_between(pdos_c, energies_c, color='tab:blue', alpha=0.2)
        axins.plot(pdos_o, energies_o, color='tab:red')
        axins.fill_between(pdos_o, energies_o, color='tab:red', alpha=0.2)
        # sub region of the original image
        x1, x2, y1, y2 = 0, 0.05, -20, -5
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        # axa[i,j].indicate_inset_zoom(axins, edgecolor="tab:grey")
        

        # Display relevant information on plot
        ax[i,j].set_title(metal)
        axa[i,j].set_title(metal)
        ax[i,j].set_xticks([])
        axa[i,j].set_xticks([])

        # if metal == 'Pt':
        #     # TODO: Something with the BZ-integration?
        #     # Get the index for the second largest peak
        #     index_plot = np.argpartition(pdos, -2)[-2]
        #     ax[i,j].set_xlim([None, pdos[index_plot]])

        # get the d-band center and the width
        # Fit a semi-ellipse to the data by passing in the 
        # moments as an initial guess to the curve fitting procedure
        center, second_moment = get_distribution_moment(energies, pdos, (1, 2)) 
        popt, pcov = curve_fit(semi_ellipse, energies, pdos, p0=[center, 4*np.sqrt(second_moment), 1])
        # The center is just the first element of the popt array
        eps_d = popt[0]
        # Store the width as the 2*width from the fitting procedure
        width =  2 * popt[1] 
        # Store the moments
        moments[metal]['d_band_centre'] = eps_d
        moments[metal]['width'] = np.abs(width)

        # Plot the semi-ellipse and its Hillbert transform
        Lambda_semiellipse = np.imag(signal.hilbert(semi_ellipse(energies, *popt)))

        # Find the index where energies are between eps_d + width/2
        # and eps_d - width/2
        # index_o_filled = np.argwhere((energies_o < eps_d + width/2) & (energies_o > eps_d - width/2)).flatten()
        # index_c_filled = np.argwhere((energies_c < eps_d + width/2) & (energies_c > eps_d - width/2)).flatten()
        index_o_filled = np.argwhere(energies_o <= 0).flatten()
        index_c_filled = np.argwhere(energies_c <= 0).flatten()

        # Get the filling of the pdos of the metal lying between the d-band
        f_c = np.trapz(pdos_c[index_c_filled], energies_c[index_c_filled]) / np.trapz(pdos_c, energies_c)
        f_o = np.trapz(pdos_o[index_o_filled], energies_o[index_o_filled]) / np.trapz(pdos_o, energies_o)
        # Store for later use in the fitting procedure
        filling_data['C'][metal] = f_c
        filling_data['O'][metal] = f_o

        # Fit the Vak for the different adsorbates by making sure
        # that the obtained density of states from the Newns-Anderson 
        # model is close to that from the projected density of states
        # of the DFT calculation.
        # for adsorbate, eps_a in adsorbates.items():
        #     # Get adsorbate speicifc fitted quantities
        #     if adsorbate == 'C':
        #         pdos_dft = pdos_c_interp
        #         ax_p = axc[i,j]
        #     elif adsorbate == 'O':
        #         pdos_dft = pdos_o_interp
        #         ax_p = axo[i,j]
        #     # Get the fitted Vak function
        #     fit_Vak = FittingVak(rho_d, eps_a, Delta0=initial_guess_adsorbate[adsorbate][0])
        #     # Get the initial guesses
        #     Vsd = initial_guess_adsorbate[adsorbate][1] 
        #     initial_guess = [Vsd]
        #     # Fit the data
        #     data = odr.RealData(x=energies, y=pdos_dft)
        #     fitting_model = odr.Model(fit_Vak.fitting_function)
        #     fitting_odr = odr.ODR(data, fitting_model, initial_guess)
        #     fitting_odr.set_job(fit_type=2)
        #     output = fitting_odr.run()
        #     print(f'Metal: {metal} with fitted Vak: {output.beta[0]}') 

            # Plot the fitted result
            # opt_dos = fit_Vak.fitting_function(output.beta, energies, store_variables=True)
            # ax_p.plot(opt_dos, energies, color='tab:orange')
            # Freeze the xlim to what it is currently
            # xlim = ax_p.get_xlim()
            # ax_p.set_xlim(xlim)
            # ax_p.plot(fit_Vak.Lambda, energies, color='tab:green', alpha=0.5)
            # ax_p.plot(fit_Vak.energy_diff, energies, color='tab:grey', alpha=0.5)

            # Get and plot the semi-elliptical dos
            # semi_ellipse_dos = get_semi_ellipse_dos(Vak=output.beta[0],
            #                                         eps_a=eps_a,
            #                                         eps_d=eps_d,
            #                                         width=width,
            #                                         eps=energies,
            #                                         Delta0_mag=initial_guess_adsorbate[adsorbate][0],
            #                                         )
            # ax_p.plot(semi_ellipse_dos, energies, color='tab:green')

            # Store the fitted values
            # Vak_data[adsorbate][metal] = np.abs(output.beta[0])
            # Delta0_data[adsorbate][metal] = fit_Vak.Delta0 

        ax[i,j].axhline(y=center, color='k', linestyle='-')
        ax[i,j].axhline(y=center + width / 2, color='k', linestyle='--')
        ax[i,j].axhline(y=center - width / 2, color='k', linestyle='--')
        axa[i,j].axhline(y=center + width / 2, color='k', linestyle='--')
        axa[i,j].axhline(y=center - width / 2, color='k', linestyle='--')
        axa[i,j].axhline(y=center + width / 2, color='k', linestyle='--')
        axa[i,j].axhline(y=center - width / 2, color='k', linestyle='--')

    # Delete unused axes 
    complete_label = False
    for i in range(len(METALS)):
        for j in range(len(METALS[i])):
            if (i,j) not in used_ij:
                if complete_label==False:
                    axa[i,j].plot([], [], color='tab:blue', label='C*')
                    axa[i,j].plot([], [], color='tab:red', label='O*')
                    axa[i,j].legend(loc='best')
                    complete_label = True
                ax[i,j].axis('off')
                axa[i,j].axis('off')

    ax[1,0].set_ylabel(r'$\epsilon - \epsilon_F$ (eV)')
    axa[1,0].set_ylabel(r'$\epsilon - \epsilon_F$ (eV)')
    fig.savefig(f'output/pdos_{FUNCTIONAL}.png')
    figa.savefig(f'output/pdos_adsorbate_{FUNCTIONAL}.png')

    pprint(moments)
    pprint(Vak_data)
    pprint(Delta0_data)

    with open(f'output/pdos_moments_{FUNCTIONAL}.json', 'w') as handle:
        json.dump(moments, handle, indent=4)
    with open(f'output/filling_data_{FUNCTIONAL}.json', 'w') as handle:
        json.dump(filling_data, handle, indent=4)
    # with open(f'output/Vak_data_{FUNCTIONAL}.json', 'w') as handle:
    #     json.dump(Vak_data, handle, indent=4)
    # with open(f'output/Delta0_data_{FUNCTIONAL}.json', 'w') as handle:
    #     json.dump(Delta0_data, handle, indent=4)