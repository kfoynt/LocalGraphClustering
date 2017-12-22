import matplotlib.pyplot as plt

def ncp_min_cond_func(grp):
    minj = grp["output_cond"].idxmin()
    #print(grp["output_cond"])
    result = grp.loc[minj]
    result["best"] = minj
    return result
    #return grp
    
def ncp_min(grp, feature):
    minj = grp[feature].idxmin()
    result = grp.loc[minj]
    result["best"] = minj
    return result

def ncp_min_feature_by_group(df, feature, group):
    return df.groupby(group).apply(lambda x: ncp_min(x, feature))

class NCPPlots:
    def __init__(self, df):
        self.df = df
        
    def feature_by_group(self, feature, group):
        ax = self.df.plot.scatter(x=group, y=feature)
        ncp_min_feature_by_group(self.df, feature, group).plot.line(
                x=group, y=feature, ax=ax)
        plt.show()
        return ax
    #plot conductance vs size
    def cond_by_size(self):
        return self.feature_by_group("output_cond", "output_sizeeff")
    #plot conductance vs volume
    def cond_by_vol(self):
        ax = self.df.plot.scatter(x="output_voleff", y="output_cond")
        ncp_min_feature_by_group(self.df, "output_cond", "output_voleff").plot.line(
                x="output_voleff", y="output_cond", ax=ax)
        plt.show()        
    #plot isoperimetry vs size
    def isop_by_size(self):
        ax = self.df.plot.scatter(x="output_sizeeff", y="output_isop")
        ncp_min_feature_by_group(self.df, "output_isop", "output_sizeeff").plot.line(
                x="output_sizeeff", y="output_isop", ax=ax)
        plt.show()