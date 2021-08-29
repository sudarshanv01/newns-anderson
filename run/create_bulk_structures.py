
import numpy as np
from ase.build import bulk
from ase.data import covalent_radii, atomic_numbers
from aiida import orm

if __name__ == '__main__':
    STRUCTURES_GROUP_LABEL = 'initial_bulk_structures'

    metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

    for metal in metals:
        print('Creating lattice for {m}'.format(m=metal))
        a = covalent_radii[atomic_numbers[metal]] * 2 * np.sqrt(2) 

        # Store the structure
        try:
            ase_structures = bulk(metal, a=a, crystalstructure='fcc', cubic=True)
        except:
            print('Failed to create structure for {m}'.format(m=metal))
            continue
        StructureData = DataFactory('structure')
        structure = StructureData(ase=ase_structures)
        structure.store()

        structure.set_extra('metal', metal)
        structure.set_extra('crystalstructure', 'fcc')

        subgroup, _ = orm.Group.objects.get_or_create(label=STRUCTURES_GROUP_LABEL)
        subgroup.add_nodes(structure)