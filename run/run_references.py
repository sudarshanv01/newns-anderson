
from aiida import orm
from aiida.engine import run, submit
from pprint import pprint
import numpy as np
import json
from aiida.tools.groups import GroupPath
from ase import Atoms
from ase.build import bulk, molecule

def calculator(ecutwf, ecutrho):
    param_dict =  {
    "CONTROL":{
        'calculation':'relax',
        'etot_conv_thr':2e-5,
        'forc_conv_thr':1e-4,
        'tprnfor':True,
        'tstress':True,
        'nstep':200,
                },
    "SYSTEM": {
        "ecutwfc": ecutwf,
        "ecutrho": ecutrho,
        "occupations": 'smearing',
        'smearing':'cold',
        'degauss':1.46997236e-2,
        'nosym':False,
                },
    "ELECTRONS": {
        "conv_thr": 4e-10,
        'electron_maxstep': 200,
        'mixing_beta': 0.2,
        'mixing_ndim': 15,
        'diagonalization': 'david',
                },
    }

    return param_dict

def runner(structure):
    RelaxWorkflow = WorkflowFactory('quantumespresso.pw.relax')

    family = load_group('SSSP/1.1/PBE/precision')
    pseudos = family.get_pseudos(structure=structure)
    cutoffs = family.get_recommended_cutoffs(structure=structure)  

    parameters = calculator(ecutwf=cutoffs[0], ecutrho=cutoffs[1])

    StructureData = DataFactory('structure')

    code = load_code('pw_6-7@localhost')
    builder = RelaxWorkflow.get_builder()
    builder.metadata.label = 'Reference Calculation'
    builder.metadata.description = 'Relaxing for generating vacancies'
    builder.structure = structure 

    builder.meta_convergence = orm.Bool(False)
    builder.clean_workdir = orm.Bool(True)

    builder.base.pw.parameters = orm.Dict(dict=parameters)

    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1])
    builder.base.kpoints = kpoints

    builder.base.pw.pseudos = family.get_pseudos(structure=structure)

    builder.base.pw.metadata.options.resources = {'num_machines': 1}
    builder.base.pw.metadata.options.max_wallclock_seconds = 1 * 60 * 60

    builder.base.pw.code = code

    calculation = submit(builder)
    path = GroupPath()
    path["references/PBE/SSSP_precision"].get_group().add_nodes(calculation)


if __name__ == '__main__':
    
    system = Atoms('H') 
    system.set_cell([10, 10, 10])
    StructureData = DataFactory('structure')
    runner(StructureData(ase=system))
