# -*- coding: utf-8 -*-

from ase import build
from aiida.orm import load_code
from aiida import orm, engine

def runner(structure):

    BaseGPAW = WorkflowFactory('ase.gpaw.base')
    builder = BaseGPAW.get_builder()

    # Structure specifications
    builder.structure = structure

    # Code specifications
    code = load_code('gpaw.21.6.0@dtu_xeon8')
    builder.gpaw.code = code

    # k-point information
    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1])
    builder.kpoints = kpoints

    # Parameters for an LCAO calculation
    parameters = { 
        'calculator': {
            'name': 'gpaw',
            'args':{
                'mode': 'lcao',
                'basis': 'dzp',
                'occupations':
                    {
                        'name' : 'marzari-vanderbilt',
                        'width' : 0.1,
                    },
                'xc': 'RPBE',
            },
        },
        'extra_imports':[
            ['gpaw.lcao.tools', 'get_lcao_hamiltonian'],
            ['gpaw.utilities.dos', 'RestartLCAODOS'],
            ['ase.constraints', 'FixAtoms'],
                    ],
        'post_lines':[
            "H_skMM, S_kMM = get_lcao_hamiltonian(calculator)",
            "results['S_kMM'] = S_kMM.tolist()",
            "results['H_skMM'] = H_skMM.tolist()",
            "dos = RestartLCAODOS(calculator)",
            "energies, weights = dos.get_atomic_subspace_pdos(range(len(atoms)))",
            "results['energies_dos'] = energies.tolist()",
            "results['weights_dos'] = weights.tolist()",
        ],
    }

    builder.gpaw.parameters = orm.Dict(dict=parameters)

    # Running the calculation using gpaw python
    settings = {'CMDLINE': ['python']}
    builder.gpaw.settings = orm.Dict(dict=settings)

    # Specifications of the time and resources
    builder.gpaw.metadata.options.resources = {'num_machines': 1}
    builder.gpaw.metadata.options.max_wallclock_seconds = 5 * 60 * 60

    calculation = engine.submit(BaseGPAW, **builder)
    
    subgroup.add_nodes(calculation)

if __name__ == '__main__':
    GROUP_NAME = 'fcc_111/surfaces/initial_structures'
    CALC_GROUPNAME = 'fcc_111/surfaces/scf_calculation'
    subgroup, _ = orm.Group.objects.get_or_create(label=CALC_GROUPNAME)

    qb = QueryBuilder()
    qb.append(Group, filters={'label':GROUP_NAME}, tag='Group')
    qb.append(Node, with_group='Group', tag='Screening')

    for i, node in enumerate(qb.all(flat=True)):
        if i > 0:
            runner(node)