import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import localgraphclustering.ncp
   
def _ncp_min(grp, feature):
    if len(grp[feature]) > 0: 
        minj = grp[feature].idxmin()
        result = grp.loc[minj]
        result["best"] = minj
        return result
        
def ncp_min_feature_by_group(df, feature, group):
    return df.groupby(group).apply(lambda x: _ncp_min(x, feature))
    
def ncp_min_feature_by_group_binned(df, feature, group, nbins=50, log=False): 
    xs = df[group].as_matrix().copy()
    xs.sort()
    if log is True:
        xs = np.log10(xs)
        edges = np.power(10.0,np.histogram(xs, bins=nbins)[1]) # second output
    else:
        edges = np.histogram(xs, bins=nbins)[1]
    buckets = pd.cut(df[group], edges)
    return df.groupby(buckets).apply(lambda x: _ncp_min(x, feature))

class NCPPlots:
    def __init__(self, var, method_name="", selected_rows=[]):
        print(type(var))
        if type(var) is localgraphclustering.ncp.NCPData:
            self.df = var.as_data_frame()
        elif type(var) is pd.DataFrame:
            self.df = var
        else:
            raise Exception(
                "Invalid argument to NCPPlots, need NCPData or DataFrame not %s"%(
                type(var).__name__))
        if selected_rows != []:
            self.df = self.df.iloc[selected_rows,:]
        if method_name != "":
            available_methods = list(set(self.df["method"]))
            if np.sum([method_name in i for i in available_methods]) == 0:
                raise Exception("Method name is not available. Options are %s"%"".join(i+" " for i in list(available_methods)))
            self.df = self.df[self.df["method"].str.contains(method_name)]

    def feature_by_group(self, feature, group):
        ax = self.df.plot.scatter(x=group, y=feature)
        ncp_min_feature_by_group(self.df, feature, group).plot.line(
                x=group, y=feature, ax=ax)
        plt.show()
        return ax
  
        
    def mqi_input_output_cond_plot(self, nbins=50):
        ncpdata = self.df
        fig, ax = plt.subplots()
        ax.hexbin(ncpdata["input_cond"], ncpdata["output_cond"], 
                      gridsize=nbins, cmap="magma", bins='log', mincnt=1)
        ax.set_xlabel("input conductance")
        ax.set_ylabel("output conductance")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_aspect('equal', 'box')
        #axins = inset_axes(ax, width="40%", height="40%", loc=2)
        axins = fig.add_axes([0.3,0.6,0.2,0.2], xscale='log', yscale='log')
        axins.hist(ncpdata["output_cond"]/ncpdata["input_cond"])
        axins.yaxis.set_ticks_position('none') 
        axins.spines['right'].set_visible(False)
        axins.spines['top'].set_visible(False)
        axins.spines['left'].set_visible(False)
        axins.set_title("improvement")
        axins.set_xticks([0.1, 0.3, 1.0])
        axins.set_yticks([])
        return fig, ax, axins
        
    def feature_by_group_histogram(self, feature, group, nbins=50, log=True):
        ncpdata = self.df
        fig,ax = plt.subplots()
        if log:
            ax.hexbin(ncpdata[group], ncpdata[feature], 
              gridsize=50, cmap="magma", bins='log', mincnt=1, xscale='log', yscale='log')
        else:
            ax.hexbin(ncpdata[group], ncpdata[feature], 
              gridsize=50, cmap="magma",  mincnt=1)
        return fig, ax
        
    def feature_by_group_histogram_and_min_line(self, feature, group, 
                                    nbins=50, nbinsx=100, log=True):
        ncpdata = self.df
        fig, ax = self.feature_by_group_histogram(
            feature, group, nbins=nbins, log=True)
        dfmin = ncp_min_feature_by_group_binned(ncpdata, "output_cond", "output_voleff",
            nbins=nbinsx, log=log).dropna(axis=0)
        y = dfmin[feature]
        x = dfmin[group]
        tmp = list(zip(x,y))
        tmp.sort(key = lambda x: x[0])
        x = [i[0] for i in tmp]
        y = [i[1] for i in tmp]
        ax.plot(x, y)
        return fig, ax
        
    
    #plot conductance vs volume
    def cond_by_vol(self, **kwargs):
        fig, ax = self.feature_by_group_histogram_and_min_line(
            "output_cond", "output_voleff", **kwargs)
        ax.set_xlabel("effective volume")
        ax.set_ylabel("conductance")
        return fig, ax
        
      
    #plot conductance vs size
    def cond_by_size(self, **kwargs):
        fig, ax = self.feature_by_group_histogram_and_min_line(
            "output_cond", "output_sizeeff", **kwargs)
        ax.set_xlabel("effective size")
        ax.set_ylabel("conductance")
        return fig, ax
            
        
    def isop_by_size(self, nbins=50, nbinsx=100):
        ncpdata = self.df
        fig, ax = self.feature_by_group_histogram(
            "output_isop", "output_sizeeff", nbins=nbins, log=True)
        dfmin = ncp_min_feature_by_group_binned(ncpdata, "output_isop", "output_sizeeff",
            nbins=nbinsx).dropna(axis=0)
        y = dfmin["output_isop"]
        x = dfmin["output_sizeeff"]
        ax.set_xlabel("effective size")
        ax.set_ylabel("isoperimetry")
        ax.plot(x, y)
        return fig, ax
