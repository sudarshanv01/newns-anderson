
import sys
from copy import deepcopy
from pprint import pprint
import numpy as np
from aiida_submission_controller import FromGroupSubmissionController
from aiida.plugins import DataFactory, WorkflowFactory
from aiida import orm
from aiida.engine import submit

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')

def calculator(ecutwf, ecutrho, metal, nbnds=None):
    param_dict =  {
    "CONTROL":{
        'calculation':'relax',
        'tprnfor':True,
        'tstress':True,
        'tefield':True,
        'dipfield':True,
                },
    "SYSTEM": {
        "ecutwfc": ecutwf,
        "ecutrho": ecutrho,
        "occupations":'smearing',
        "smearing":'gauss',
        "degauss":0.0075,
        "nspin": 1,
        "edir": 3,
        "emaxpos": 0.05,
        "eopreg": 0.025,
        "eamp": 0.0,
        "nspin":2,
        "starting_magnetization":{metal: 0.5},
                },
    "ELECTRONS": {
        "conv_thr": 1e-7,
        'electron_maxstep': 200,
        'mixing_beta': 0.1,
        'mixing_ndim': 15,
        'diagonalization': 'david',
        'mixing_mode': 'local-TF',
                },
    }
    if nbnds:
        param_dict['SYSTEM']['nbnd'] = nbnds

    return param_dict

def get_nbands_data(metal, atoms, family, extra=50):
    """Given the metal atom, ase atoms object and
    family of pseudopotentials, find the number of bands
    to set in the calculation."""
    upf_data = family.get_pseudo(metal)
    valence_electrons = upf_data.get_attribute('z_valence')
    number_electrons = valence_electrons * len(atoms)
    return number_electrons / 2 + extra 


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
        return ['metal', 'facets', 'sampled_index']

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """Return inputs and process class for the submission of this specific process.
        """
        structure = self.get_parent_node_from_extras(extras_values)
        ase_structure = structure.get_ase()
        family = load_group('SSSP/1.1/PBE/precision')
        pseudos = family.get_pseudos(structure=structure)
        cutoffs = family.get_recommended_cutoffs(structure=structure)  

        inputs = PwBaseWorkChain.get_builder()
        inputs.pw.code = self._code

        # k-points to be used in the calculation
        KpointsData = DataFactory('array.kpoints')
        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([4, 4, 1])
        inputs.kpoints = kpoints

        # Get the cutoff for the calculation
        ecutwf = max(80, cutoffs[0])
        ecutrho = max(600, cutoffs[1])

        # Get the number of bands of the calculation
        nbands = get_nbands_data(extras_values[0], ase_structure, family, extra=50)

        # Get the parameters for the calculation
        parameters = calculator(ecutwf=ecutwf, ecutrho=ecutrho, nbnds=nbands, metal=extras_values[0])
        inputs.pw.parameters = orm.Dict(dict=parameters)

        inputs.pw.structure = structure
        inputs.pw.pseudos = pseudos

        # Constrain metal atoms
        molecule_symbol = ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']
        fixed_coords = []
        for atom in ase_structure:
            if atom.symbol in molecule_symbol: 
                fixed_coords.append([False, False, False])
            else:
                fixed_coords.append([True, True, True])

        settings = {'fixed_coords': fixed_coords, 
                    'cmdline': ['-nk', '2']}

        inputs.pw.settings = orm.Dict(dict=settings)

        inputs.pw.metadata.options.resources = {'num_machines': 4}
        inputs.pw.metadata.options.max_wallclock_seconds = 15 * 60 * 60

        return inputs, self._process_class

if __name__ == '__main__':

    # For the calculation
    ADSORBATE = sys.argv[1]

    # For the submission controller
    DRY_RUN = False
    MAX_CONCURRENT = 50
    CODE_LABEL = f'pw_6-7_stage2022@juwels_scr'
    STRUCTURES_GROUP_LABEL = f'PBE_spin/SSSP_precision/gauss_smearing_0.1eV/sampling/initial/{ADSORBATE}'
    WORKFLOWS_GROUP_LABEL = f'PBE_spin/SSSP_precision/gauss_smearing_0.1eV/sampling/relax/{ADSORBATE}'

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
