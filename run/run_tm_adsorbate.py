
import sys
from copy import deepcopy
from pprint import pprint
import numpy as np
from aiida_submission_controller import FromGroupSubmissionController
from aiida.plugins import DataFactory, WorkflowFactory
from aiida import orm
from aiida.engine import submit

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')

def calculator(ecutwf, ecutrho):
    param_dict =  {
    "CONTROL":{
        'calculation':'scf',
        'tprnfor':True,
        'tstress':True,
        # 'nstep':200,
        'tefield':True,
        'dipfield':True,
                },
    "SYSTEM": {
        "ecutwfc": ecutwf,
        "ecutrho": ecutrho,
        "occupations":'smearing',
        "smearing":'cold',
        "degauss":0.01,
        "nspin": 1,
        "edir": 3,
        "emaxpos": 0.05,
        "eopreg": 0.025,
        "eamp": 0.0,
                },
    "ELECTRONS": {
        "conv_thr": 1e-8,
        'electron_maxstep': 200,
        'mixing_beta': 0.2,
        'mixing_ndim': 15,
        'diagonalization': 'david',
        'mixing_mode': 'local-TF',
                },
    }

    return param_dict

class AdsorbateSubmissionController(FromGroupSubmissionController):
    """A SubmissionController for submitting Adsorbates with Quantum ESPRESSO base workflows."""
    def __init__(self, code_label, *args, **kwargs):
        """Pass also a code label, that should be a code associated to an `quantumespresso.pw` plugin."""
        super().__init__(*args, **kwargs)
        self._code = orm.load_code(code_label)
        self._process_class = WorkflowFactory('quantumespresso.pw.base')

    def get_extra_unique_keys(self):
        """Return a tuple of the keys of the unique extras that will be used to uniquely identify your workchains.
        """
        return ['metal', 'facet']

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """Return inputs and process class for the submission of this specific process.
        """
        structure = self.get_parent_node_from_extras(extras_values)
        ase_structure = structure.get_ase()
        family = load_group('SSSP/1.1/PBE/efficiency')
        pseudos = family.get_pseudos(structure=structure)
        cutoffs = family.get_recommended_cutoffs(structure=structure)  

        inputs = PwBaseWorkChain.get_builder()
        inputs.pw.code = self._code

        KpointsData = DataFactory('array.kpoints')
        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([4, 4, 1])
        inputs.kpoints = kpoints

        ecutwf = max(60, cutoffs[0])
        ecutrho = max(480, cutoffs[1])

        parameters = calculator(ecutwf=ecutwf, ecutrho=ecutrho)

        inputs.pw.parameters = orm.Dict(dict=parameters)

        inputs.pw.structure = structure
        inputs.pw.pseudos = pseudos

        # Constrain the bottom two layers of the metal
        molecule_symbol = ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']
        fixed_coords = []
        for atom in ase_structure:
            if atom.symbol in molecule_symbol: 
                fixed_coords.append([False, False, False])
            else:
                fixed_coords.append([True, True, True])

        settings = {'fixed_coords': fixed_coords, 
                    'cmdline': ['-nk', '4']}
        inputs.pw.settings = orm.Dict(dict=settings)

        inputs.pw.metadata.options.resources = {'num_machines': 4}
        inputs.pw.metadata.options.max_wallclock_seconds = 10 * 60 * 60

        return inputs, self._process_class

if __name__ == '__main__':

    # For the calculation
    COMPUTER = sys.argv[1]
    ADSORBATE = sys.argv[2]

    # For the submission controller
    DRY_RUN = False
    MAX_CONCURRENT = 25
    CODE_LABEL = f'pw_6-7@{COMPUTER}'
    STRUCTURES_GROUP_LABEL = f'transition_metals/initial_structures/{ADSORBATE}'
    WORKFLOWS_GROUP_LABEL = f'transition_metals/scf/{ADSORBATE}'

    controller = AdsorbateSubmissionController(
        parent_group_label=STRUCTURES_GROUP_LABEL,
        code_label=CODE_LABEL,
        group_label=WORKFLOWS_GROUP_LABEL,
        max_concurrent=MAX_CONCURRENT)
    
    print('Already run    :', controller.num_already_run)
    print('Max concurrent :', controller.max_concurrent)
    print('Available slots:', controller.num_available_slots)
    print('Active slots   :', controller.num_active_slots)
    print('Still to run   :', controller.num_to_run)
    print()

    run_processes = controller.submit_new_batch(dry_run=DRY_RUN)
    for run_process_extras, run_process in run_processes.items():
        if run_process is None:
            print(f'{run_process_extras} --> To be run')    
        else:
            print(f'{run_process_extras} --> PK = {run_process.pk}')

    print()
