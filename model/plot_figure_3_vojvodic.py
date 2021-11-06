"""Plot part of Figure 3 of the Vojvodic et al. paper."""
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from NewnsAnderson import NewnsAndersonNumerical, NewnsAndersonAnalytical

if __name__ == '__main__':

    eps_a = -5 # eV 
    EPS_RANGE = np.linspace(-20, 20, 10000)
    k = 0

    # Load data from vojvodic_parameters.yaml
    with open('vojvodic_parameters.yaml', 'r') as f:
        vojvodic_parameters = yaml.safe_load(f)

    metals = []
    energies = []
    for metal in vojvodic_parameters['epsd']:
        eps_d = vojvodic_parameters['epsd'][metal]
        second_moment = vojvodic_parameters['mc'][metal]
        Vaksq = vojvodic_parameters['Vaksq'][metal]
        Vak = np.sqrt(Vaksq)
        # Get the width based on the second moment
        width = 4 * np.sqrt(second_moment)

        # Get the energy from the Newns Anderson model
        newns = NewnsAndersonNumerical(
            width = width,
            Vak = Vak, 
            eps_a = eps_a,
            eps_d = eps_d,
            eps = EPS_RANGE,
            k = k, 
        )
        newns.calculate_energy()

        energies.append(newns.DeltaE)
        metals.append(metal)

    data = np.array([metals, energies]).T
    pprint(data)
