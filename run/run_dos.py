"""Run the dos workchain to get the single point energies and the dos."""
import sys
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida import orm
from copy import deepcopy
from aiida.engine import run, submit
from aiida.tools.groups import GroupPath
from aiida_submission_controller import FromGroupSubmissionController
from aiida.plugins import DataFactory, WorkflowFactory

PpDosChain = WorkflowFactory('quantumespresso.pdos')

def calculator(ecutwf, ecutrho, metal, nbnd=None):
    param_dict =  {
    "CONTROL":{
        'calculation':'scf',
        'tprnfor':True,
        'tstress':True,
        'tefield':True,
        'dipfield':True,
                },
    "SYSTEM": {
        "ecutwfc": ecutwf,
        "ecutrho": ecutrho,
        "occupations":'smearing',
        "smearing": 'gauss',
        "degauss": 0.0075,
        "nspin": 1,
        "edir": 3,
        "emaxpos": 0.05,
        "eopreg": 0.025,
        "eamp": 0.0,
        "nspin":2,
        "starting_magnetization":{metal: 0.5},
                },
    "ELECTRONS": {
        "conv_thr": 1e-8,
        'electron_maxstep': 250,
        'mixing_beta': 0.1,
        'mixing_ndim': 15,
        'diagonalization': 'david',
        'mixing_mode': 'local-TF',
                },
    }

    if nbnd is not None:
        param_dict['SYSTEM']['nbnd'] = int(nbnd)

    return param_dict

def get_nbands_data(metal, atoms, family, extra=50):
    """Given the metal atom, ase atoms object and
    family of pseudopotentials, find the number of bands
    to set in the calculation."""
    upf_data = family.get_pseudo(metal)
    valence_electrons = upf_data.get_attribute('z_valence')
    number_electrons = valence_electrons * len(atoms)
    return number_electrons / 2 + extra 

class DOSSubmissionController(FromGroupSubmissionController):
    """A SubmissionController for submitting DOS calculations for Adsorbates with Quantum ESPRESSO base workflows."""
    def __init__(self, code_label, *args, **kwargs):
        """Pass also a code label, that should be a code associated to an `quantumespresso.pw` plugin."""
        super().__init__(*args, **kwargs)
        self._code = orm.load_code(code_label)
        self._process_class = WorkflowFactory('quantumespresso.pdos')

    def get_extra_unique_keys(self):
        """Return a tuple of the keys of the unique extras that will be used to uniquely identify your workchains.
        """
        return ['metal', 'facet']

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """Return inputs and process class for the submission of this specific process.
        """
        # Create the pdos chain builder
        builder = PpDosChain.get_builder()
        
        # Only the structure is passed
        structure = self.get_parent_node_from_extras(extras_values)
        # relax_node = self.get_parent_node_from_extras(extras_values)
        # structure = relax_node.outputs.output_structure

        # Get cutoff information
        family = load_group('SSSP/1.1/PBE/precision')
        cutoffs = family.get_recommended_cutoffs(structure=structure)  
        ecutwf = max(80, cutoffs[0])
        ecutrho = max(600, cutoffs[1])

        # Get the number of bands
        nbnd = get_nbands_data(extras_values[0], structure.get_ase(), family, 300)

        # get the scf information
        # parameters_scf = calculator(ecutwf, ecutrho)
        parameters_scf = calculator(ecutwf, ecutrho, extras_values[0], nbnd)

        # Get the nscf information
        parameters_nscf = deepcopy(parameters_scf)
        parameters_nscf['CONTROL']['calculation'] = 'nscf'
        # parameters_nscf['SYSTEM']['occupations'] = 'tetrahedra'

        # Code related information
        code_pw = load_code(f'pw_6-7_stage2022{COMPUTER}')
        code_dos = load_code(f'dos_6-7_stage2022{COMPUTER}')
        code_projwfc = load_code(f'projwfc_6-7_stage2022{COMPUTER}')

        settings = {'cmdline': ['-nk', '2']}

        # k-points related information
        kpt_mesh_scf = [4, 4, 1]
        kpt_mesh_nscf = [int(2*kpt_mesh_scf[0]), int(2*kpt_mesh_scf[1]), 1]
        KpointsData = DataFactory('array.kpoints')

        kpoints_scf = KpointsData()
        kpoints_nscf = KpointsData()
        kpoints_scf.set_kpoints_mesh(kpt_mesh_scf)
        kpoints_nscf.set_kpoints_mesh(kpt_mesh_nscf)

        # First level of inputs to the Pp workchain 
        builder.structure = structure

        builder.serial_clean = orm.Bool(False)
        builder.clean_workdir = orm.Bool(False)
        builder.align_to_fermi = orm.Bool(True)

        builder.scf.kpoints = kpoints_scf
        builder.scf.pw.pseudos = family.get_pseudos(structure=structure)
        builder.scf.pw.parameters = orm.Dict(dict=parameters_scf)
        builder.scf.pw.code = code_pw
        builder.scf.pw.metadata.options.resources = {'num_machines': 4}
        builder.scf.pw.metadata.options.max_wallclock_seconds =  10 * 60 * 60
        builder.scf.pw.settings = orm.Dict(dict=settings)

        ## NSCF inputs to the Pp workchain
        builder.nscf.kpoints = kpoints_nscf
        builder.nscf.pw.pseudos = family.get_pseudos(structure=structure) 
        builder.nscf.pw.parameters = orm.Dict(dict=parameters_nscf)
        builder.nscf.pw.code = code_pw
        builder.nscf.pw.metadata.options.resources = {'num_machines': 4}
        builder.nscf.pw.metadata.options.max_wallclock_seconds = 10 * 60 * 60

        ## dos inputs to Pp workchain
        dos_parameters = {'DOS':
                                {'Emin':-20,
                                'Emax':20, 
                                'DeltaE':0.01,
                                }
                        }
        builder.dos.parameters = orm.Dict(dict=dos_parameters) 
        builder.dos.code = code_dos
        builder.dos.metadata.options.resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 10}
        builder.dos.metadata.options.max_wallclock_seconds = 10 * 60

        ## projwfc inputs to the Pp workchain
        projwfc_parameters = {'PROJWFC':
                                    {'Emin':-20,
                                    'Emax':20, 
                                    'DeltaE':0.01},
                                    }
        builder.projwfc.parameters = orm.Dict(dict=projwfc_parameters) 
        builder.projwfc.code = code_projwfc
        builder.projwfc.metadata.options.resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 20}
        builder.projwfc.metadata.options.max_wallclock_seconds = 60 * 60

        return builder, self._process_class
    

if __name__ == '__main__':
    # For the calculation
    SYSTEM = sys.argv[1]
    COMPUTER = '@juwels_scr' # sys.argv[2]

    # For the submission controller
    DRY_RUN = False
    MAX_CONCURRENT = 10
    CODE_LABEL = f'pw_6-7_stage2022{COMPUTER}'
    STRUCTURES_GROUP_LABEL = f'PBE_spin/SSSP_precision/gauss_smearing_0.1eV/initial/{SYSTEM}'
    WORKFLOWS_GROUP_LABEL = f'PBE_spin/SSSP_precision/gauss_smearing_0.1eV/dos_scf/{SYSTEM}' 

    controller = DOSSubmissionController(
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

