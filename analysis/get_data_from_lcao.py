"""Plots the results of LCAO calculations for the Newns Anderson Model."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import index
from aiida import orm
from NewnsAndersonLCAO import NewnsAndersonLCAO
from plot_params import get_plot_params
get_plot_params()
BaseGPAW = WorkflowFactory('ase.gpaw.base')


if __name__ == '__main__':

    GROUP_NAME = 'GPAW/LCAO/surface_structures/scf/O'
    ADSORBATE = GROUP_NAME.split('/')[-1]
    print(f'Parsing data for {ADSORBATE}')
    TYPE_OF_CALC = BaseGPAW 
    POSSIBLE_ADSORBATE_INDEX = list(ADSORBATE) 

    # Query for the data of the calculation
    qb = orm.QueryBuilder()
    qb.append(Group, filters={'label':GROUP_NAME}, tag='Group')
    qb.append(TYPE_OF_CALC, with_group='Group', )


    for results in qb.all(flat=True):
        if not results.is_finished_ok:
            continue

        # Get the structure and get relevant information
        # from that data
        structure = results.inputs.structure
        ase_structure = structure.get_ase()
        adsorbate_index = [atom.index for atom in ase_structure if atom.symbol in POSSIBLE_ADSORBATE_INDEX]
        metal_index = [atom.index for atom in ase_structure if atom.symbol not in POSSIBLE_ADSORBATE_INDEX]
        metal_name = np.unique([atom.symbol for atom in ase_structure if atom.symbol not in POSSIBLE_ADSORBATE_INDEX])[0]

        # Get the relevant quantities for the 
        # Newns Anderson Model
        arrays = results.outputs.array
        H_skMM = arrays.get_array('H_skMM')
        S_kMM = arrays.get_array('S_kMM')
        energies_dos = arrays.get_array('energies_dos')
        weights_dos = arrays.get_array('weights_dos')
        eigenvalues = arrays.get_array('eigenvalues')
        adsorbate_basis_index = arrays.get_array('basis_func_adsorbate')
        fermi_energy = results.outputs.parameters.get_attribute('fermi_energy')

        # Choose the right index for the subdiagonalisation
        # of the Hamiltonian
        H_MM = H_skMM[0,0]
        S_MM = S_kMM[0]

        # Partitition the index
        all_index = np.arange(len(H_MM[0]))
        adsorbate_index = adsorbate_basis_index 
        metal_index = np.delete(all_index, adsorbate_basis_index) 

        # Range of values for which we want delta to be plotted
        delta_range = np.array([-10, 10]) + fermi_energy 
        # Decide if more padding with zero's is required
        PAD_RANGE = np.array([15, 15]) # Pad with zeros 10 eV on either side
        # Perform all the manipulations of the Newns-Anderson model
        newns = NewnsAndersonLCAO(H_MM, S_MM, adsorbate_index, metal_index,
                                    broadening_width=0.1, cutoff_range=delta_range,
                                    pad_range=PAD_RANGE)
        data_to_pickle = {}
        # Need to only store one Delta
        data_to_pickle['Delta'] = newns.delta[0].tolist()
        data_to_pickle['eigenval_ads'] = newns.eigenval_ads.tolist()
        data_to_pickle['eigenval_metal'] = newns.eigenval_metal.tolist()
        data_to_pickle['fermi_energy'] = fermi_energy
        data_to_pickle['metal'] = metal_name
        data_to_pickle['adsorbate'] = ADSORBATE
        data_to_pickle['Vak'] = newns.Vak.tolist()
        data_to_pickle['Sak'] = newns.Sak.tolist()
        with open('output/delta/{a}_{b}_lcao_data.pkl'.format(a=metal_name, b=ADSORBATE), 'wb') as f:
            pickle.dump(data_to_pickle, f)

