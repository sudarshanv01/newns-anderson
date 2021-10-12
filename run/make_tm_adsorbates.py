"""Make the slab from the bulk structure."""

from ase import build
from aiida import orm
import numpy as np
from ase.data import atomic_numbers, atomic_masses, covalent_radii
StructureData = DataFactory('structure')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')

if __name__ == '__main__':

    STRUCTURES_FULL_GROUP_LABEL = 'support_bulk'
    ADSORBATE  = 'C'
    MOL_INDEX = 0
    STRUCTURES_GROUP_LABEL = f'transition_metals/constant_height_initial_structures/{ADSORBATE}'
    subgroup, _ = orm.Group.objects.get_or_create(label=STRUCTURES_GROUP_LABEL)
    DRY_RUN = False
    all_metal_list = []

    facets = {'Sc':'001', 'Ti':'001', 'V':'110', 'Cr':'110', 
              'Fe':'110', 'Co':'001', 'Ni':'111', 'Cu':'111',
              'Y':'001', 'Zr':'001', 'Nb':'110', 'Mo':'110', 
              'Ru':'001', 'Rh':'111', 'Pd':'111', 'Ag':'111', 
              'Hf':'001', 'Ta':'110', 'W':'110', 'Re':'001',
              'Os':'001', 'Ir':'111', 'Pt':'111', 'Au':'111',
              'Al': '111'}

    query = orm.QueryBuilder()
    query.append(orm.Group, tag='group', filters={'label': STRUCTURES_FULL_GROUP_LABEL})
    query.append(PwRelaxWorkChain, tag='pwrelax', with_group='group')
    results = query.all(flat=True)

    for res in results:
        bulk_structure = res.outputs.output_structure.get_ase()

        metal = bulk_structure.get_chemical_symbols()[0]
        if metal in all_metal_list:
            continue
        else:
            all_metal_list.append(metal)

        if facets[metal] == '001':
            layers = 2
            repeats = [5, 5, 1]
        elif facets[metal] == '110':
            layers = 4
            repeats = [2, 3, 1]
        elif facets[metal] == '100':
            layers = 3
            repeats = [3, 3, 1]
        else:
            layers = 3
            repeats = [2, 2, 1]

        a = np.linalg.norm(bulk_structure.cell[0])
        facet_name = [int(a) for a in list(facets[metal])]
        surface = build.surface(bulk_structure, facet_name , layers=layers, vacuum=10)
        surface = surface.repeat(repeats)
        surface.set_tags(None)

        # create adsorbate
        height = 1.8 # covalent_radii[atomic_numbers[metal]] + covalent_radii[atomic_numbers[list(ADSORBATE)[MOL_INDEX]]]
        adsorbate = build.molecule(ADSORBATE)
        surface_index = np.argmax(surface.positions[:, 2])
        # adsorbate.rotate(180, 'x')
        build.add_adsorbate(surface, adsorbate, height, position=surface[surface_index].position[:2], mol_index=MOL_INDEX) 

        if DRY_RUN:
            print(f'Dry run: {surface}')
            surface.write(f'output/transition_metals/{ADSORBATE}_{metal}_{facets[metal]}.cif') 
            # bulk_structure.write(f'outputs/transition_metals/{metal}_bulk.cif')
        
        else:
            structure = StructureData(ase=surface)
            structure.store()

            structure.set_extra('metal', metal)
            structure.set_extra('facet', facets[metal])

            subgroup.add_nodes(structure)

            print(f"Structures added to group '{STRUCTURES_GROUP_LABEL}'")
            print(f"Current group size: {len(subgroup.nodes)}")
