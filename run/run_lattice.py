
from aiida import orm
from aiida.engine import run, submit
from pprint import pprint
import numpy as np
import json
from aiida.tools.groups import GroupPath
from ase import Atoms
from ase import build
import time

def calculator(ecutwf, ecutrho):
    param_dict =  {
    "CONTROL":{
        'calculation':'vc-relax',
        'etot_conv_thr':2e-5,
        'forc_conv_thr':1e-4,
        'tprnfor':True,
        'tstress':True,
        'nstep':200,
                },
    "SYSTEM": {
        "ecutwfc": ecutwf,
        "ecutrho": ecutrho,
        "occupations":'smearing',
        "smearing":'cold',
        "degauss":0.01,
                },
    "ELECTRONS": {
        "conv_thr": 1e-9,
        'electron_maxstep': 200,
        'mixing_beta': 0.2,
        'mixing_ndim': 15,
        'diagonalization': 'david',
                },
    "CELL":{
            'cell_dofree':'xyz',
            }
    }

    return param_dict

def runner(structure):
    ase_structure = structure.get_ase()
    chemical_symbols = np.unique(ase_structure.get_chemical_symbols())

    family = load_group('SSSP/1.1/PBE/precision')
    cutoffs = family.get_recommended_cutoffs(structure=structure)  
    ecutwf = max(90, cutoffs[0])
    ecutrho = max(600, cutoffs[1])

    StructureData = DataFactory('structure')
    RelaxWorkflow = WorkflowFactory('quantumespresso.pw.relax')
    builder = RelaxWorkflow.get_builder()

    code = load_code('pw_6-7@dtu_xeon24')
    builder.metadata.label = 'Lattice Relaxation Calculation'
    builder.metadata.description = 'All cell vectors to relax to get the lattice constants.'
    builder.structure = structure 

    builder.meta_convergence = orm.Bool(False)
    builder.clean_workdir = orm.Bool(False)
    builder.base.pw.parameters = orm.Dict(dict=calculator(ecutwf, ecutrho))

    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([12, 12, 12])
    builder.base.kpoints = kpoints

    builder.base.pw.pseudos = family.get_pseudos(structure=structure)

    builder.base.pw.metadata.options.resources = {'num_machines': 1}
    builder.base.pw.metadata.options.max_wallclock_seconds =  1 * 60 * 60

    builder.base.pw.code = code

    calculation = submit(builder)
    path = GroupPath()
    path["bulk_structures/PBE/SSSP_precision"].get_group().add_nodes(calculation)


if __name__ == '__main__':

    StructureData = DataFactory('structure')

    metals = [ 'Sc', 'Ti', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu',
               'Zr', 'Nb', 'Mo', 'Ru', 'Hf', 'Ta', 'W', 'Y',
               'Re', 'Os', 'Ir', 'Ag', 'Au', 'Pt', 'Pd', 'Rh' ]

    for metal in metals:
        try:
            atoms = build.bulk(metal, cubic=True)
        except RuntimeError:
            atoms = build.bulk(metal)

        structure = StructureData(ase=atoms)
        time.sleep(3)
        runner(structure)
