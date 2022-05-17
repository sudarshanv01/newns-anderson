"""Check the validity of VakS approximation."""
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import yaml
import json
import pickle
from catchemi import (  NewnsAndersonLinearRepulsion,
                        NewnsAndersonGrimleyRepulsion
)
import string
from create_coupling_elements import create_coupling_elements

if __name__ == '__main__':
    """Compare the variation of the orthogonalisation energy
    the d-band centre for the modified 2-state problem.""" 

    # Setup the parameters needed in the model.
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = open('chosen_setup', 'r').read() 
    REPULSION_TYPES = ['linear', 'linear_mod', 'grimley']
    REPULSION_LABELS = ['two-state', 'modified two-state', 'grimley']
    COLORS = ['tab:red', 'tab:blue', 'tab:orange']
    METAL_ROW_INDEX = 1 # Index of metal row to be plotted; 0 1 2
    NUMBER_OF_METALS = 100

    # Read in the metal fitting splines
    with open(f"output/spline_objects.pkl", 'rb') as f:
        spline_objects = pickle.load(f)

    # Other input quantities from LMTO
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    no_of_bonds = yaml.safe_load(open('inputs/number_bonds.yaml', 'r'))
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    minmax_parameters = json.load(open('output/minmax_parameters.json'))

    # Parameters for the model
    ADSORBATES = ['O', 'C']
    EPS_A_VALUES = [ -5, -1 ] # eV
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    EPS_VALUES = np.linspace(-30, 10, 1000)

    # Separate figure for each adsorbate
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        fig, ax = plt.subplots(1, 4, figsize=(6.75,2.), constrained_layout=True)
        # The repulsion figures will be plotted together
        for j, repulsion in enumerate(REPULSION_TYPES):

            if repulsion == 'linear':
                add_largeS_contribution = False
                repulsive_method = NewnsAndersonLinearRepulsion
            elif repulsion == 'linear_mod':
                add_largeS_contribution = True
                repulsive_method = NewnsAndersonLinearRepulsion
            elif repulsion == 'grimley':
                add_largeS_contribution = False
                repulsive_method = NewnsAndersonGrimleyRepulsion

            # Read in scaling parameters from the model.
            with open(f"output/O_repulsion_{repulsion}_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
                o_parameters = json.load(f)
            with open(f"output/C_repulsion_{repulsion}_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
                c_parameters = json.load(f)
            adsorbate_params = {'O': o_parameters, 'C': c_parameters}

            # Get the adsorbate specific parameters for each calculation
            print(f"Plotting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
            alpha = adsorbate_params[adsorbate]['alpha']
            beta = adsorbate_params[adsorbate]['beta']
            constant_offest = adsorbate_params[adsorbate]['constant_offset']
            CONSTANT_DELTA0 = adsorbate_params[adsorbate]['delta0']

            # Perform a continuous variation of all metals
            parameters_metal = defaultdict(list)

            # get the metal fitting parameters
            s_fit = spline_objects[METAL_ROW_INDEX]['s']
            Delta_anderson_fit = spline_objects[METAL_ROW_INDEX]['Delta_anderson']
            wd_fit = spline_objects[METAL_ROW_INDEX]['width']
            eps_d_fit = spline_objects[METAL_ROW_INDEX]['eps_d']
            filling_min, filling_max = minmax_parameters[str(j)]['filling']
            eps_d_min, eps_d_max = minmax_parameters[str(j)]['eps_d']
            filling_range = np.linspace(filling_max, filling_min, NUMBER_OF_METALS)
            eps_d_range = np.linspace(eps_d_min, eps_d_max, NUMBER_OF_METALS)

            # First contruct a continuous variation of all parameters
            results = defaultdict(list)
            for k, filling in enumerate(filling_range):
                # Continuous setting of parameters for each 
                # continous variation of the metal
                width = wd_fit(filling) 
                eps_d = eps_d_fit(filling) 
                Vsdsq = create_coupling_elements(s_metal=s_fit(filling),
                                                s_Cu=s_data['Cu'],
                                                anderson_band_width=Delta_anderson_fit(filling),
                                                anderson_band_width_Cu=anderson_band_width_data['Cu'],
                                                r=s_fit(filling),
                                                r_Cu=s_data['Cu'],
                                                normalise_by_Cu=True,
                                                normalise_bond_length=True
                                                )
                Vsd = np.sqrt(Vsdsq)

                kwargs = dict(
                    eps_d = eps_d,
                    eps_sp_min = EPS_SP_MIN,
                    eps_sp_max = EPS_SP_MAX,
                    eps = EPS_VALUES,
                    Delta0_mag = CONSTANT_DELTA0,
                    Vsd = Vsd, 
                    width = width, 
                    eps_a = eps_a,
                    verbose = True,
                    alpha = alpha,
                    beta = beta,
                    constant_offset = constant_offest,
                    add_largeS_contribution = add_largeS_contribution,
                )

                # If grimley repulsion remove the largeS contribution
                if repulsion == 'grimley':
                    kwargs.pop('add_largeS_contribution')

                # Get the energies for the repulsive method
                repulsive_class = repulsive_method(**kwargs)

                chem_energy = repulsive_class.get_chemisorption_energy()
                ortho_energy = repulsive_class.get_orthogonalisation_energy()
                hyb_energy = repulsive_class.get_hybridisation_energy()
                occupancy = repulsive_class.get_occupancy()
                filling = repulsive_class.get_dband_filling()

                results['chem_energy'].append(chem_energy)
                results['ortho_energy'].append(ortho_energy)
                results['hyb_energy'].append(hyb_energy)
                results['eps_d'].append(eps_d)
                results['na_plus_f'].append(occupancy + filling)

            ax[0].plot(results['eps_d'], results['chem_energy'], color=COLORS[j],
                        label=REPULSION_LABELS[j].title())
            ax[1].plot(results['eps_d'], results['ortho_energy'], color=COLORS[j])
            ax[2].plot(results['eps_d'], results['hyb_energy'], color=COLORS[j])
            ax[3].plot(results['eps_d'], results['na_plus_f'], color=COLORS[j])

        ax[0].set_ylabel(r'$E_{\mathrm{chem}}$ (eV)')
        ax[1].set_ylabel(r'$E_{\mathrm{ortho}}$ (eV)')
        ax[2].set_ylabel(r'$E_{\mathrm{hyb}}$ (eV)')
        ax[3].set_ylabel(r'$n_a + f$ (e)')
        for a in ax:
            a.set_xlabel(r'$\epsilon_d$ (eV)')
        # ax[0].legend(loc='best',)
        ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1, fontsize=5.7)

        # Add figure numbers
        alphabet = list(string.ascii_lowercase)
        for i, a in enumerate(ax.T.flatten()):
            a.annotate(alphabet[i]+')', xy=(0.05, 0.5), fontsize=8, xycoords='axes fraction')

        fig.savefig(f'output/compare_validity_twostate_approx_adsorbate_{adsorbate}.png', dpi=300)