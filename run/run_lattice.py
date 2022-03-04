"""Get the lattice constants of the different metals."""
from aiida import orm
from aiida.engine import run, submit
import numpy as np
from aiida.tools.groups import GroupPath
from ase import build
import time

def calculator(ecutwf, ecutrho, nbnd=None):
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
        "smearing":'gauss',
        "degauss":0.0075,
        "nosym":True,
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
    if nbnd is not None:
        param_dict["SYSTEM"]["nbnd"] = int(nbnd)

    return param_dict

def get_nbands_data(metal, atoms, family, extra=50):
    """Given the metal atom, ase atoms object and
    family of pseudopotentials, find the number of bands
    to set in the calculation."""
    upf_data = family.get_pseudo(metal)
    valence_electrons = upf_data.get_attribute('z_valence')
    number_electrons = valence_electrons * len(atoms)
    return number_electrons / 2 + extra 

def runner(structure, metal):
    """Run the calculation based on the settings that we will use throughout
    the work."""

    family = load_group('SSSP/1.1/PBE/precision')
    cutoffs = family.get_recommended_cutoffs(structure=structure)  
    ecutwf = max(80, cutoffs[0])
    ecutrho = max(600, cutoffs[1])
    nbnd = get_nbands_data(metal, structure.get_ase(), family)

    RelaxWorkflow = WorkflowFactory('quantumespresso.pw.relax')
    builder = RelaxWorkflow.get_builder()

    code = load_code('pw_6-7_stage2022@juwels_scr')
    builder.metadata.label = 'Lattice Relaxation Calculation'
    builder.metadata.description = 'All cell vectors to relax to get the lattice constants.'
    builder.structure = structure 

    builder.meta_convergence = orm.Bool(False)
    builder.clean_workdir = orm.Bool(False)
    builder.base.pw.parameters = orm.Dict(dict=calculator(ecutwf, ecutrho, nbnd))

    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([12, 12, 12])
    builder.base.kpoints = kpoints

    builder.base.pw.pseudos = family.get_pseudos(structure=structure)

    builder.base.pw.metadata.options.resources = {'num_machines': 2}
    builder.base.pw.metadata.options.max_wallclock_seconds = 2 * 60 * 60

    builder.base.pw.code = code

    settings = {'cmdline': ['-nk', '2']}
    builder.base.pw.settings = orm.Dict(dict=settings)

    calculation = submit(builder)
    path = GroupPath()
    path["PBE/SSSP_precision/gauss_smearing_0.1eV/bulk_structures"].get_group().add_nodes(calculation)


if __name__ == '__main__':

    StructureData = DataFactory('structure')

    # metals = [ 'Sc', 'Ti', 'V' , 'Cr', 'Fe', 'Co', 'Ni', 'Cu',
    #            'Zr', 'Nb', 'Mo', 'Ru', 'Hf', 'Ta', 'W', 'Y',
    #            'Re', 'Os', 'Ir', 'Ag',  'Pt', 'Pd', 'Rh', 
    #            'Al', ]
    metals = ['Fe']

    for metal in metals:
        try:
            atoms = build.bulk(metal, cubic=True)
        except RuntimeError:
            atoms = build.bulk(metal)

        structure = StructureData(ase=atoms)
        runner(structure, metal)
        time.sleep(3)
