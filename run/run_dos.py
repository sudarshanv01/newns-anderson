import time
import sys
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida import orm
from copy import deepcopy
from aiida.engine import run, submit
from aiida.tools.groups import GroupPath
from aiida_submission_controller import FromGroupSubmissionController
from aiida.plugins import DataFactory, WorkflowFactory

PpDosChain = WorkflowFactory('quantumespresso.pdos')

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

        builder = PpDosChain.get_builder()

        relax_workchain = self.get_parent_node_from_extras(extras_values)
        structure = relax_workchain.outputs.output_structure
        parameters_scf = relax_workchain.inputs.pw.parameters.get_dict()
        parameters_scf['CONTROL']['calculation'] = 'scf'
        # Choose at least a ecutwfc of 60
        parameters_scf['SYSTEM']['ecutwfc'] = max(60, parameters_scf['SYSTEM']['ecutwfc'])
        # Choose an ecutrho of at least 480
        parameters_scf['SYSTEM']['ecutrho'] = max(480, parameters_scf['SYSTEM']['ecutrho'])
        # Remove spin
        parameters_scf['SYSTEM'].pop('nspin')
        parameters_scf['SYSTEM'].pop('starting_magnetization')

        parameters_nscf = deepcopy(parameters_scf)
        parameters_nscf['CONTROL']['calculation'] = 'nscf'
        parameters_nscf['SYSTEM']['occupations'] = 'tetrahedra'

        pseudos = relax_workchain.inputs.pw.pseudos

        code_pw = load_code(f'pw_6-7@{COMPUTER}')
        code_dos = load_code(f'dos_6-7@{COMPUTER}')
        code_projwfc = load_code(f'projwfc_6-7@{COMPUTER}')

        KpointsData = DataFactory('array.kpoints')
        family = load_group('SSSP/1.1/PBE/efficiency')

        ## First level of inputs to the Pp workchain 
        builder.structure = structure

        builder.serial_clean = orm.Bool(True)
        builder.clean_workdir = orm.Bool(True)
        builder.align_to_fermi = orm.Bool(True)

        kpoints_old = relax_workchain.inputs.kpoints.attributes['mesh']
        kpts_new = [int(2*kpoints_old[0]), int(2*kpoints_old[1]), 1]
        KpointsData = DataFactory('array.kpoints')
        kpoints_scf = KpointsData()
        kpoints_scf.set_kpoints_mesh(kpoints_old)

        kpoints_nscf = KpointsData()
        kpoints_nscf.set_kpoints_mesh(kpts_new)

        builder.scf.kpoints = kpoints_scf
        builder.scf.pw.pseudos = family.get_pseudos(structure=structure)
        builder.scf.pw.parameters = orm.Dict(dict=parameters_scf)
        builder.scf.pw.code = code_pw
        builder.scf.pw.metadata.options.resources = {'num_machines': 2}
        builder.scf.pw.metadata.options.max_wallclock_seconds =  5 * 60 * 60

        ## NSCF inputs to the Pp workchain
        builder.nscf.kpoints = kpoints_nscf
        builder.nscf.pw.pseudos = family.get_pseudos(structure=structure) 
        builder.nscf.pw.parameters = orm.Dict(dict=parameters_nscf)
        builder.nscf.pw.code = code_pw
        builder.nscf.pw.metadata.options.resources = {'num_machines': 1}
        builder.nscf.pw.metadata.options.max_wallclock_seconds = 4 * 60 * 60

        ## dos inputs to Pp workchain
        dos_parameters = {'DOS':
                                {'Emin':-20,
                                'Emax':20, 
                                'DeltaE':0.01,
                                }
                        }
        builder.dos.parameters = orm.Dict(dict=dos_parameters) 
        builder.dos.code = code_dos
        builder.dos.metadata.options.resources = {'num_machines': 1}
        builder.dos.metadata.options.max_wallclock_seconds = 10 * 60

        ## projwfc inputs to the Pp workchain
        projwfc_parameters = {'PROJWFC':
                                    {'Emin':-20,
                                    'Emax':20, 
                                    'DeltaE':0.01},
                                    }
        builder.projwfc.parameters = orm.Dict(dict=projwfc_parameters) 
        builder.projwfc.code = code_projwfc
        builder.projwfc.metadata.options.resources = {'num_machines': 1, 'num_mpiprocs_per_machine':12}
        builder.projwfc.metadata.options.max_wallclock_seconds = 10 * 60

        return builder, self._process_class
    

if __name__ == '__main__':
    # For the calculation
    COMPUTER = 'juwels_scr' 
    SYSTEM = sys.argv[1]

    # For the submission controller
    DRY_RUN = False
    MAX_CONCURRENT = 10
    CODE_LABEL = f'pw_6-7@{COMPUTER}'
    STRUCTURES_GROUP_LABEL = f'transition_metals/relaxed/{SYSTEM}'
    WORKFLOWS_GROUP_LABEL = f'transition_metals/relaxed/density_of_states/{SYSTEM}'

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

