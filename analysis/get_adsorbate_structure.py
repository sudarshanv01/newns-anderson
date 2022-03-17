"""View structures for C and O from the DFT calculation."""
import numpy as np
import matplotlib.pyplot as plt
from aiida import orm
from ase import io
from collections import defaultdict
import json, yaml

# --- AiiDA imports
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
DosWorkflow = WorkflowFactory('quantumespresso.pdos')
StructureData = DataFactory('structure')

if __name__ == '__main__':
    """View structures of the adsorbate on the metal slabs and
    plot the bond length of C and O from the metal centers."""

    # Group name for the adsorbate on the transition metal
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = open('chosen_setup', 'r').read() + '_groupname' 
    COMP_SETUP = COMP_SETUP[CHOSEN_SETUP]

    ADSORBATES = ['C', 'O'] 
    type_of_calc = PwBaseWorkChain 
    LABEL = COMP_SETUP.replace('/', '_')

    all_bond_lengths = {}

    for ADSORBATE in ADSORBATES:

        # Get the nodes from the calculation
        qb = QueryBuilder()
        qb.append(Group, filters={'label':COMP_SETUP+'/'+ADSORBATE}, tag='Group')
        qb.append(type_of_calc, with_group='Group', tag='calctype')

        all_structures = defaultdict(list)
        all_energies = defaultdict(list)
        bond_lengths = defaultdict(list)

        for node in qb.all(flat=True):
            # get the output structure
            if not node.is_finished_ok:
                continue

            if type_of_calc == DosWorkflow:
                output_structure = node.inputs.structure
                energy = None
            elif type_of_calc == PwBaseWorkChain:
                output_structure = node.outputs.output_structure
                energy = node.outputs.output_parameters.get_dict()['energy']

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
            all_structures[metal_name].append(ase_structure)

            # If available store the energies
            if energy is not None:
                all_energies[metal_name].append(energy)

            # Store the bond lengths for later plotting
            bond_lengths[metal_name].append(bond_length)
        
        most_stable_structure = []
        most_stable_bond_length = {}
        # Get the lowest energy for each metal
        for metal, energies_list in all_energies.items():
            argmin_energies = np.argmin(energies_list)
            # most_stable_structure.append(all_structures[metal][argmin_energies])
            most_stable_structure.extend(all_structures[metal])
            most_stable_bond_length[metal] = bond_lengths[metal][argmin_energies]
        all_bond_lengths[ADSORBATE] = most_stable_bond_length
        io.write(f'output/all_structures_{ADSORBATE}_{LABEL}.xyz', most_stable_structure)

    # Store the bond lengths
    with open(f'output/bond_lengths_{LABEL}.json', 'w') as f:
        json.dump(all_bond_lengths, f, indent=4)


        



