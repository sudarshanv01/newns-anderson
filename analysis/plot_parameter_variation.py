"""Plot the variation of the parameters with elements."""
import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint
from adjustText import adjust_text
from collections import defaultdict
from plot_params import get_plot_params
get_plot_params()
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 
if __name__ == "__main__":
    """Plot the variation of epsilon_d, width and Vsd with elements."""
    # Load the moments
    with open("output/pdos_moments.json", 'r') as handle:
        moments = json.load(handle)
    pprint(moments)
    # Load the Vsd and filling data
    with open("inputs/data_from_LMTO.json", 'r') as handle:
        data_from_LMTO = json.load(handle)
    
    Vsd_data = data_from_LMTO["Vsd"]
    filling_data = data_from_LMTO["filling"]

    # Plot the epsilon_d, width and Vsd
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)

    texts = defaultdict(list)
    for metal in moments:
        # Get the epsilon_d
        epsilon_d = moments[metal]["d_band_centre"]
        # Get the width
        width = moments[metal]["width"]
        # Get the filling
        filling = filling_data[metal]
        # Get the Vsd
        Vsd = Vsd_data[metal]

        # Decide on the color based on the metal
        if metal in FIRST_ROW:
            color = "red"
        elif metal in SECOND_ROW:
            color = "orange"
        elif metal in THIRD_ROW:
            color = "green"

        ax[0].plot(filling, epsilon_d, 'o', color=color)
        # ax[0].annotate(metal, (filling, epsilon_d), color=color)
        texts[0].append(ax[0].text(filling, epsilon_d, metal, color=color))

        ax[1].plot(filling, width, 'o', color=color)
        # ax[1].annotate(metal, (filling, width), color=color)
        texts[1].append(ax[1].text(filling, width, metal, color=color))

        ax[2].plot(filling, Vsd, 'o', color=color)
        # ax[2].annotate(metal, (filling, Vsd), color=color)
        texts[2].append(ax[2].text(filling, Vsd, metal, color=color))

    for i, text in texts.items():
        adjust_text(text, ax=ax[i])
    
    # Set the labels
    ax[0].set_xlabel("Idealised filling of $d$")
    ax[0].set_ylabel("$\epsilon_d$ (eV)")
    ax[1].set_xlabel("Idealised filling of $d$")
    ax[1].set_ylabel("Width (eV)")
    ax[2].set_xlabel("Idealised filling of $d$")
    ax[2].set_ylabel("$V_{sd}$ (eV)")

    # Save the figure
    fig.savefig("output/parameter_variation.png")






    

