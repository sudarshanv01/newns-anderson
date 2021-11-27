"""Compare energies for DFT calculations with and without relaxation."""
import matplotlib.pyplot as plt
import json
from adjustText import adjust_text
from plot_params import get_plot_params
get_plot_params()

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 

if __name__ == "__main__":
    """Compare the adsorption energies with and without relaxation 
    for the same functional."""

    # Read the data from the json file
    without_relax = json.load(open("output/adsorption_energies_PBE_scf.json"))
    with_relax = json.load(open("output/adsorption_energies_PBE.json"))

    # Plot the data according to the metal row
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    # Store text for adjustable plots
    text_C = []
    text_O = []

    # Iterate over row
    for row_index, row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        if row_index == 0:
            color ='tab:blue'
        elif row_index == 1:
            color ='tab:orange'
        elif row_index == 2:
            color ='tab:green'

        for metal in row:
            if metal in without_relax['C'] and metal in with_relax['C']:
                # Plot the adsorption energies without relaxation
                ax[0].plot(without_relax['C'][metal], with_relax['C'][metal],
                            marker='o', color=color)
            if metal in without_relax['O'] and metal in with_relax['O']:
                # Plot the adsorption energies without relaxation
                ax[1].plot(without_relax['O'][metal], with_relax['O'][metal],
                            marker='o', color=color)
                # Add text to the plot
                text_C.append(ax[0].text(without_relax['C'][metal], 
                                          with_relax['C'][metal], metal,
                                          color=color, fontsize=12))
                text_O.append(ax[1].text(without_relax['O'][metal], 
                                          with_relax['O'][metal], metal,
                                          color=color, fontsize=12))

    adjust_text(text_C, ax=ax[0])
    adjust_text(text_O, ax=ax[1])
    
    # Set the labels
    ax[0].set_xlabel('$E_C$ without relaxation (eV)')
    ax[1].set_xlabel('$E_O$ without relaxation (eV)')
    ax[0].set_ylabel('$E_C$ with relaxation (eV)')
    ax[1].set_ylabel('$E_O$ with relaxation (eV)')

    fig.savefig('output/adsorption_energies_PBE_scf_vs_PBE.png', dpi=300)


