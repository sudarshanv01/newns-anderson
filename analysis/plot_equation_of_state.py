"""Plot the equation of state for a particular group and write out relevant structures if needed."""
import collections
import numpy as np
import json
import matplotlib.pyplot as plt
from ase import eos
from ase.build import fcc111, add_adsorbate
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase import io
from aiida import orm
# aiida specific
StructureData = DataFactory('structure')

if __name__ == '__main__':
    """Create the equation of state plots and write out the lattice constants
    for all the metal atoms in a json file. This file will then form the basis
    for creating different surfaces to get Delta from."""

    # Group name for which to get the equation of state data
    GROUP_NAME = 'equation_of_state'

    # Type of calculation, most likely it is commonworkflows
    CommonWorflows = WorkflowFactory('common_workflows.eos')
    node_type = CommonWorflows 
    
    qb = QueryBuilder()
    qb.append(Group, filters={'label':GROUP_NAME}, tag='Group')
    qb.append(node_type, with_group='Group', tag='Screening')

    # Bulk structures for the different elements
    lattices = []
    info = {}

    # Plot the equation of state for each node
    for node in qb.all(flat=True):
        if node.is_finished_ok:
            energies_dict = node.outputs.total_energies
            structure_dict = node.outputs.structures
            energies = []
            volumes = []
            cells = []
            for i, Energy in energies_dict.items():
                energies.append(Energy.value)
                ase_structure = structure_dict[i].get_ase()
                cells.append(ase_structure.cell[:])
                volumes.append(ase_structure.get_volume())
                metal = np.unique(ase_structure.get_chemical_symbols())[0]
            print(f'Plotting for {metal} with node pk: {node.pk}')
            try:
                method = eos.EquationOfState(volumes, energies)
                method.plot(f'output/eos/{metal}.png') 
                v0, e0, B = method.fit()
                print(f'{metal} a = {v0**(1/3)}')
            except ValueError:
                print(f'No lattice found for {metal}')
                continue
            plt.close()

            # Get the cell based on the scaling factor needed to get
            # the desired volume. Then multiply the scaling factor to the original
            # cell and get the new cell.
            scaling_factor = v0 / volumes[0]
            new_cell = np.multiply(scaling_factor**(1/3), cells[0])
            ase_structure.set_cell(new_cell)
            lattices.append(ase_structure)

            # Store the node pk for some iterations if the results need to be improved
            info[metal] = node.pk

    # Write out the lattice constants to a json file
    io.write('output/lattice_constants.json', lattices, format='json')

    # Write out the info into a json file
    with open('output/eos_aiida_info.json', 'w') as f:
        json.dump(info, f, indent=4)
