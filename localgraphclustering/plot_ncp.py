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
    
def plot_ncp_size(isoperimetry_vs_size):
    """
    Plots the Network Community Profile, i.e., minimum isoperimetry vs size.
    """    
    lists = sorted(isoperimetry_vs_size.items())
    x, y = zip(*lists)

    fig = plt.figure()
    ax = fig.add_subplot(111)
            
    plt.loglog(x, y)
            
    ax.set_xlabel('Size')
    ax.set_ylabel('Minimum isoperimetry')  
            
    ax.set_title('Min. Isoperimetry vs. Size NCP')
            
    plt.show()
    
def plot_ncp_MQI_size(isoperimetry_R_vs_size_R,isoperimetry_MQI_vs_size_R):
    """
    Plots the MQI Network Community Profile, i.e., minimum isoperimetry vs size.
    """    
    lists = sorted(isoperimetry_R_vs_size_R.items())
    x, y = zip(*lists)

    fig = plt.figure()
    ax = fig.add_subplot(111)
            
    plt.loglog(x, y)
    
    lists = sorted(isoperimetry_MQI_vs_size_R.items())
    x, y = zip(*lists)
    
    plt.loglog(x, y)
    
    ax.legend(["For given sets","After applying MQI"])
            
    ax.set_xlabel('Size')
    ax.set_ylabel('Minimum isoperimetry')  
            
    ax.set_title('Min. Isoperimetry vs. Size NCP')
            
    plt.show()
    
def plot_ncp_MQI_vol(conductance_R_vs_vol_R,conductance_MQI_vs_vol_R):
    """
    Plots the Network Community Profile, i.e., minimum conductance vs volume.
    """    
    lists = sorted(conductance_R_vs_vol_R.items())
    x, y = zip(*lists)

    fig = plt.figure()
    ax = fig.add_subplot(111)
            
    plt.loglog(x, y)
            
    lists = sorted(conductance_MQI_vs_vol_R.items())
    x, y = zip(*lists)
    
    plt.loglog(x, y)
        
    ax.legend(["For given sets","After applying MQI"])    
        
    ax.set_xlabel('Volume')
    ax.set_ylabel('Minimum conductance')  
            
    ax.set_title('Min. Conductance vs. Volume NCP')
            
    plt.show()

def plot_ncp_conductance_node(conductance_vs_node_R,conductance_vs_node_MQI):
    """
    Plots the minimum conductance vs node number.
    """    
    lists = sorted(conductance_vs_node_R.items())
    x, y = zip(*lists)

    fig = plt.figure()
    ax = fig.add_subplot(111)
            
    plt.plot(x, y)
    
    lists = sorted(conductance_vs_node_MQI.items())
    x, y = zip(*lists)
    
    plt.plot(x, y)
    
    ax.legend(["For given sets","After applying MQI"])
            
    ax.set_xlabel('Node number')
    ax.set_ylabel('Minimum isoperimetry')  
            
    ax.set_title('Min. Isoperimetry vs. node number')
            
    plt.show()
    
def plot_ncp_isoperimetry_node(isoperimetry_vs_node_R,isoperimetry_vs_node_MQI):
    """
    Plots the minimum conductance vs node number.
    """    
    lists = sorted(isoperimetry_vs_node_R.items())
    x, y = zip(*lists)

    fig = plt.figure()
    ax = fig.add_subplot(111)
            
    plt.plot(x, y)
    
    lists = sorted(isoperimetry_vs_node_MQI.items())
    x, y = zip(*lists)
    
    plt.plot(x, y)
    
    ax.legend(["For given sets","After applying MQI"])
            
    ax.set_xlabel('Node number')
    ax.set_ylabel('Minimum isoperimetry')  
            
    ax.set_title('Min. Isoperimetry vs. node number')
            
    plt.show()