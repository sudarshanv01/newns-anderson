"""Plot the equation of state for a particular group and write out relevant structures if needed."""

import collections
from aiida import orm
import numpy as np
from ase import eos
import matplotlib.pyplot as plt
from ase.build import fcc111, add_adsorbate
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers

if __name__ == '__main__':
    GROUP_NAME = 'equation_of_state'
    CommonWorflows = WorkflowFactory('common_workflows.eos')
    node_type = CommonWorflows 
    CREATE_SURFACE = True
    STRUCTURES_GROUP_LABEL = 'fcc_111/6x6x4/surfaces/initial_structures'

    CREATE_ADSORBATES = True
    C_STRUCTURE_GROUP_LABEL = 'fcc_111/6x6x4/c_adsorption/initial_structures'
    O_STRUCTURE_GROUP_LABEL = 'fcc_111/6x6x4/o_adsorption/initial_structures'
    
    StructureData = DataFactory('structure')

    qb = QueryBuilder()
    qb.append(Group, filters={'label':GROUP_NAME}, tag='Group')
    qb.append(node_type, with_group='Group', tag='Screening')

    subgroup, _ = orm.Group.objects.get_or_create(label=STRUCTURES_GROUP_LABEL)
    subgroup_c, _ = orm.Group.objects.get_or_create(label=C_STRUCTURE_GROUP_LABEL)
    subgroup_o, _ = orm.Group.objects.get_or_create(label=O_STRUCTURE_GROUP_LABEL)

    for node in qb.all(flat=True):
        # Plot the equation of state for every metal
        if node.is_finished_ok:
            energies_dict = node.outputs.total_energies
            structure_dict = node.outputs.structures
            energies = []
            volumes = []
            for i, Energy in energies_dict.items():
                energies.append(Energy.value)
                ase_structure = structure_dict[i].get_ase()
                volumes.append(ase_structure.get_volume())
                metal = np.unique(ase_structure.get_chemical_symbols())[0]
            print(f'Plotting for {metal}')
            try:
                method = eos.EquationOfState(volumes, energies)
                method.plot(f'output/eos/{metal}.png') 
                v0, e0, B = method.fit()
                print(f'{metal} a = {v0**(1/3)}')
            except ValueError:
                print(f'No lattice found for {metal}')
                continue
            plt.close()
    
        if CREATE_SURFACE:
            a = v0**(1/3)
            surface = fcc111(metal, size=(4,4,4), vacuum=10, a=a)
            surface.write(f'output/eos/{metal}_111.traj')
            structure = StructureData(ase=surface)
            structure.store()

            # Add the structure to the group
            subgroup.add_nodes(structure)
        
        if CREATE_ADSORBATES:
            a = v0**(1/3)
            surface = fcc111(metal, size=(6,6,4), vacuum=10, a=a)

            # Add a carbon
            c_ase_structure = surface.copy()

            height = covalent_radii[atomic_numbers[metal]] + 0.5 
            add_adsorbate(c_ase_structure, 'C', height, 'ontop')

            o_ase_structure = surface.copy()
            add_adsorbate(o_ase_structure, 'O', height, 'ontop')

            c_structure = StructureData(ase=c_ase_structure)
            o_structure = StructureData(ase=o_ase_structure)

            c_structure.store()
            o_structure.store()

            c_ase_structure.write(f'output/eos/{metal}_111_c.traj')
            o_ase_structure.write(f'output/eos/{metal}_111_o.traj')

            subgroup_c.add_nodes(c_structure)
            subgroup_o.add_nodes(o_structure)

