import matplotlib.pylab as plt

def plot_ncp_vol(conductance_vs_vol):
    """
    Plots the Network Community Profile, i.e., minimum conductance vs volume.
    """    
    lists = sorted(conductance_vs_vol.items())
    x, y = zip(*lists)

    fig = plt.figure()
    ax = fig.add_subplot(111)
            
    plt.loglog(x, y)
            
    ax.set_xlabel('Volume')
    ax.set_ylabel('Minimum conductance')  
            
    ax.set_title('Min. Conductance vs. Volume NCP')
            
    plt.show()
    
def plot_ncp_size(conductance_vs_size):
    """
    Plots the Network Community Profile, i.e., minimum conductance vs size.
    """    
    lists = sorted(conductance_vs_size.items())
    x, y = zip(*lists)

    fig = plt.figure()
    ax = fig.add_subplot(111)
            
    plt.loglog(x, y)
            
    ax.set_xlabel('Size')
    ax.set_ylabel('Minimum conductance')  
            
    ax.set_title('Min. Conductance vs. Size NCP for component ')
            
    plt.show()

