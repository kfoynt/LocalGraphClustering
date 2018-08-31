import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

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
    xs = df[group].values.copy()
    xs.sort()
    xs = xs.astype(np.float64)
    if log is True:
        xs = np.log10(xs)
        edges = np.power(10.0,np.histogram(xs, bins=nbins)[1]) # second output
    else:
        edges = np.histogram(xs, bins=nbins)[1]
    buckets = pd.cut(df[group], edges)
    return df.groupby(buckets).apply(lambda x: _ncp_min(x, feature))

class NCPPlots:
    def __init__(self, var, method_name="", selected_rows=[]):
        init_notebook_mode(connected=True)
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
                raise Exception("Method name is not available. Options are %s"%"\n".join(i+"\n" for i in list(available_methods)))
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
        dfmin = ncp_min_feature_by_group_binned(ncpdata, feature, group,
            nbins=nbinsx, log=log).dropna(axis=0)
        y = dfmin[feature]
        x = dfmin[group]
        pos = dfmin["best"]
        tmp = list(zip(x,y))
        tmp.sort(key = lambda x: x[0])
        x = [i[0] for i in tmp]
        y = [i[1] for i in tmp]
        ax.plot(x, y)
        return fig, ax, list(zip(x,y,pos))


    #plot conductance vs volume
    def cond_by_vol(self, **kwargs):
        fig, ax, min_tuples = self.feature_by_group_histogram_and_min_line(
            "output_cond", "output_voleff", **kwargs)
        ax.set_xlabel("effective volume")
        ax.set_ylabel("conductance")
        return fig, ax, min_tuples


    #plot conductance vs size
    def cond_by_size(self, **kwargs):
        fig, ax, min_tuples = self.feature_by_group_histogram_and_min_line(
            "output_cond", "output_sizeeff", **kwargs)
        ax.set_xlabel("effective size")
        ax.set_ylabel("conductance")
        return fig, ax, min_tuples


    def isop_by_size(self, nbins=50, nbinsx=100, log=True):
        ncpdata = self.df
        fig, ax = self.feature_by_group_histogram(
            "output_isop", "output_sizeeff", nbins=nbins, log=log)
        dfmin = ncp_min_feature_by_group_binned(ncpdata, "output_isop", "output_sizeeff",
            nbins=nbinsx).dropna(axis=0)
        y = dfmin["output_isop"]
        x = dfmin["output_sizeeff"]
        pos = dfmin["best"]
        ax.set_xlabel("effective size")
        ax.set_ylabel("expansion")
        ax.plot(x, y)
        return fig, ax, list(zip(x,y,pos))

    def interactive(self,feature, group, min_tuples, alpha=0.3, ratio=1.0, log=True, filename=""):
        sample_size = np.int(self.df.shape[0]*ratio)
        sample_indices = np.random.choice(self.df.index.values, sample_size, replace=False)
        trace1 = go.Scattergl(
            x = self.df.iloc[sample_indices][group],
            y = self.df.iloc[sample_indices][feature],
            mode = 'markers',
            marker = dict(
                opacity=alpha,
                size= 10,
                line = dict(
                    width = 1)
                ),
            name = "Points",
            text = list(map(lambda z: 'index: {}'.format(int(z)), sample_indices))
        )

        trace2 = go.Scattergl(
            x = [i[0] for i in min_tuples],
            y = [i[1] for i in min_tuples],
            mode = 'lines',
            name = "Line",
            text = list(map(lambda z: 'index: {}'.format(int(z)), [i[2] for i in min_tuples]))
        )

        layout = go.Layout(
            xaxis=dict(
                type='log' if log else 'line',
                autorange=True,
                title = group
            ),
            yaxis=dict(
                type='log' if log else 'line',
                autorange=True,
                title = feature
            ),
            showlegend=False
        )

        data = [trace1, trace2]

        # Plot and embed in ipython notebook!
        fig = go.Figure(data=data, layout=layout)
        if filename == "":
            iplot(fig)
        else:
            plot(fig,filename=filename)

        return fig

    def cond_by_vol_itrv(self, nbinsx=100, **kwargs):
        dfmin = ncp_min_feature_by_group_binned(self.df, "output_cond", "output_voleff",
            nbins=nbinsx).dropna(axis=0)
        y = dfmin["output_cond"]
        x = dfmin["output_voleff"]
        pos = dfmin["best"]
        min_tuples = list(zip(x,y,pos))
        return self.interactive("output_cond", "output_voleff", min_tuples, **kwargs)

    def cond_by_size_itrv(self, nbinsx=100, **kwargs):
        dfmin = ncp_min_feature_by_group_binned(self.df, "output_cond", "output_sizeeff",
            nbins=nbinsx).dropna(axis=0)
        y = dfmin["output_cond"]
        x = dfmin["output_sizeeff"]
        pos = dfmin["best"]
        min_tuples = list(zip(x,y,pos))
        return self.interactive("output_cond", "output_sizeeff", min_tuples, **kwargs)

    def isop_by_size_itrv(self, nbinsx=100, **kwargs):
        dfmin = ncp_min_feature_by_group_binned(self.df, "output_isop", "output_sizeeff",
            nbins=nbinsx).dropna(axis=0)
        y = dfmin["output_isop"]
        x = dfmin["output_sizeeff"]
        pos = dfmin["best"]
        min_tuples = list(zip(x,y,pos))
        return self.interactive("output_isop", "output_sizeeff", min_tuples, **kwargs)
