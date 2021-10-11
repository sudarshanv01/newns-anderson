"""View structures for C and O from the DFT calculation."""
import numpy as np
import matplotlib.pyplot as plt
from aiida import orm
from ase import io
# --- AiiDA imports
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
DosWorkflow = WorkflowFactory('quantumespresso.pdos')
StructureData = DataFactory('structure')

if __name__ == '__main__':
    """View structures of the adsorbate on the metal slabs and
    plot the bond length of C and O from the metal centers."""

    # Group name for the adsorbate on the transition metal
    GROUPNAME = 'transition_metals/initial_structures/O'
    type_of_calc = StructureData#PwBaseWorkChain
    ADSORBATE = 'O'

    # Get the nodes from the calculation
    qb = QueryBuilder()
    qb.append(Group, filters={'label':GROUPNAME}, tag='Group')
    qb.append(type_of_calc, with_group='Group', tag='calctype')

    all_structures = []

    for node in qb.all(flat=True):
        # get the output structure
        try:
            output_structure = node.outputs.output_structure
        except AttributeError:
            output_structure = node
        ase_structure = output_structure.get_ase()

        # get the distance between the metal and the adsorbate
        adsorbate_index = [i for i in range(len(ase_structure)) if ase_structure[i].symbol == ADSORBATE][0]
        # metal index to be used is the one closest to the adsorbate
        metal_index = [i for i in range(len(ase_structure)) if ase_structure[i].symbol != ADSORBATE]
        # metal_name
        metal_name = ase_structure[metal_index[0]].symbol

        # get the distance between the metal and the adsorbate
        distances = ase_structure.get_distances(adsorbate_index, metal_index)
        metal_to_choose = metal_index[np.argmin(distances)]

        # get the bond length between the metal and the adsorbate
        bond_length = ase_structure.get_distance(metal_to_choose, adsorbate_index)

        print(f'Bond length for {ADSORBATE} on {metal_name} is {bond_length:1.2f} AA')

        all_structures.append(ase_structure)

    io.write(f'output/all_structures_{ADSORBATE}.xyz', all_structures)  

        



