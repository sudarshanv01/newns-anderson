"""Make the slab from the bulk structure."""

from ase import build
from aiida import orm
import numpy as np
from ase.data import atomic_numbers, atomic_masses, covalent_radii
StructureData = DataFactory('structure')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')

if __name__ == '__main__':

    STRUCTURES_FULL_GROUP_LABEL = 'PBE/SSSP_precision/gauss_smearing_0.1eV/bulk_structures'
    ADSORBATE  = 'CO'
    MOL_INDEX = 1

    if ADSORBATE:
        STRUCTURES_GROUP_LABEL = f'PBE/SSSP_precision/gauss_smearing_0.1eV/initial/{ADSORBATE}' 
    else:
        STRUCTURES_GROUP_LABEL = f'PBE/SSSP_precision/gauss_smearing_0.1eV/initial/slab'

    subgroup, _ = orm.Group.objects.get_or_create(label=STRUCTURES_GROUP_LABEL)

    DRY_RUN = False 
    all_metal_list = []

    facets = {'Sc':'001', 'Ti':'001', 'V':'110', 'Cr':'110', 
              'Fe':'110', 'Co':'001', 'Ni':'111', 'Cu':'111',
              'Y':'001', 'Zr':'001', 'Nb':'110', 'Mo':'110', 
              'Ru':'001', 'Rh':'111', 'Pd':'111', 'Ag':'111', 
              'Hf':'001', 'Ta':'110', 'W':'110', 'Re':'001',
              'Os':'001', 'Ir':'111', 'Pt':'111', 'Au':'111',
              'Al': '111', 'Mg': '001'}

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
            repeats = [4, 4, 1]
        elif facets[metal] == '110':
            layers = 4
            repeats = [2, 2, 1]
        elif facets[metal] == '100':
            layers = 3
            repeats = [3, 3, 1]
        else:
            layers = 4
            repeats = [2, 2, 1]

        alat = np.linalg.norm(bulk_structure.cell[0])
        if facets[metal] == '111':
            surface = build.fcc111(metal, a=alat, size=(repeats[:-1] + [layers]), vacuum=10)
        elif facets[metal] == '110':
            surface = build.bcc110(metal, a=alat, size=(repeats[:-1] + [layers]), vacuum=10)
        else:
            facet_name = [int(a) for a in list(facets[metal])]
            surface = build.surface(bulk_structure, facet_name , layers=layers, vacuum=10)

        surface = surface.repeat(repeats)
        surface.set_tags(None)

        # create adsorbate
        if ADSORBATE:
            height = covalent_radii[atomic_numbers[metal]] + covalent_radii[atomic_numbers[list(ADSORBATE)[MOL_INDEX]]]
            adsorbate = build.molecule(ADSORBATE)
            surface_index = np.argmax(surface.positions[:, 2])
            build.add_adsorbate(surface, adsorbate, height, position=surface[surface_index].position[:2], mol_index=MOL_INDEX) 

        if DRY_RUN:
            print(f'Dry run: {surface}')
            surface.write(f'output/transition_metals/{ADSORBATE}_{metal}_{facets[metal]}.cif') 
            bulk_structure.write(f'output/transition_metals/{metal}_bulk.cif')
        
        else:
            structure = StructureData(ase=surface)
            structure.store()

            structure.set_extra('metal', metal)
            structure.set_extra('facet', facets[metal])

            subgroup.add_nodes(structure)

            print(f"Structures added to group '{STRUCTURES_GROUP_LABEL}'")
            print(f"Current group size: {len(subgroup.nodes)}")
