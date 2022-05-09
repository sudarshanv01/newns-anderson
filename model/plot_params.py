

def get_plot_params():
    """Create the plot parameters used in the plotting 
    all the figures in the paper
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rc
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # plt.rcParams['xtick.major.size'] = 6
    # plt.rcParams['xtick.major.width'] = 2
    # plt.rcParams['xtick.minor.size'] = 5
    # plt.rcParams['xtick.minor.width'] = 2
    # plt.rcParams['ytick.major.size'] = 10
    # plt.rcParams['ytick.major.width'] = 2
    # plt.rcParams['ytick.minor.size'] = 5
    # plt.rcParams['ytick.minor.width'] = 2



    COLOR = 'k'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR



def get_plot_params_Times():
    """Create the plot parameters used in the plotting 
    all the figures in the paper
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rc
    # mpl.rcParams['text.usetex']=True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times New Roman'
    # rc('text.latex', preamble=r'\usepackage{cmbright}')
    mpl.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22

    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.minor.width'] = 2

    plt.rcParams['axes.labelsize'] = 26
    COLOR = 'k'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

def get_plot_params_Arial():
    """Create the plot parameters used in the plotting 
    all the figures in the paper
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rc
    # mpl.rcParams['text.usetex']=True
    mpl.rcParams['font.family'] = 'Arial'
    # mpl.rcParams['font.sans-serif'] = 'Helvetica'
    rc('text.latex', preamble=r'\usepackage{cmbright}')
    rc('text.latex', preamble=r'\usepackage{color}')
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.minor.width'] = 2

    plt.rcParams['axes.labelsize'] = 18


    COLOR = 'k'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR