import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import to_rgb,to_rgba
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class GraphDrawing:
    def __init__(self,G,coords,ax=None,groups=None,figsize=None):
        self.G = G
        self.coords = coords
        self.is_3d = (len(coords[0]) == 3)
        self.edge_pos,self.edge_mapping = self._build_edgepos(G,coords)
        if ax is None:
            fig = plt.figure(figsize=figsize)
            if len(coords[0]) == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        ax.set_axis_off()
        self.fig = fig
        self.ax = ax
        self.nodes_collection = None
        self.edge_collection = None
        self.groups = groups

    
    @staticmethod
    def _push_edges_for_node(center,points,pos,edge_pos,edge_mapping):
        for i,p in enumerate(points):
            if p >= center:
                edge_mapping[(center,p)] = len(edge_pos)
                edge_pos.append([pos[center],pos[p]])

    @staticmethod
    def _build_edgepos(G,pos):
        edge_pos = []
        edge_mapping = {}
        for i in range(G._num_vertices):
            GraphDrawing._push_edges_for_node(i,G.aj[G.ai[i]:G.ai[i+1]],pos,edge_pos,edge_mapping)
        edge_pos = np.asarray(edge_pos)
        return edge_pos,edge_mapping

    def show(self):
        return self.fig

    def highlight(self,nodelist,othernodes=False,otheredges=False,filled=False,alpha=0):
        nodeset = set(nodelist)
        if not othernodes or not filled:
            node_out = list(set(range(self.G._num_vertices)) - nodeset)
            if not othernodes:
                self.nodecolor(node_out,alpha=alpha)
            if not filled:
                self.only_circle_nodes(nodelist)
        if not otheredges:
            for (i,j) in self.edge_mapping.keys():
                if i not in nodeset or j not in nodeset:
                    self.edgecolor(i,j,alpha=alpha)
    
    def only_circle_nodes(self,nodeset):
        facecolors = self.nodes_collection.get_facecolor()
        facecolors[nodeset] = [0,0,0,0]

    def between_group_alpha(self,alpha):
        if self.groups is not None:
            if self.groups.ndim == 2:
                node_mapping = np.zeros(self.G._num_vertices,dtype=self.G.aj.dtype)
                for idx,grp in enumerate(self.groups):
                    node_mapping[grp] = idx
            else:
                node_mapping = self.groups
            for edge in self.edge_mapping.keys():
                if node_mapping[i] != node_mapping[j]:
                    self.edgecolor(edge[0],edge[1],alpha=alpha)


    def nodecolor(self,node,c=None,edgecolor=None,facecolor=None,alpha=None):
        if c is not None:
            edgecolor = c
            facecolor = c
        if facecolor is not None or alpha is not None:
            colors = self.nodes_collection.get_facecolor()
            self._update_color(colors,node,facecolor,alpha)
        if edgecolor is not None or alpha is not None:
            colors = self.nodes_collection.get_edgecolor()
            self._update_color(colors,node,edgecolor,alpha)

        return self.nodes_collection.get_facecolor()[node]

    def edgecolor(self,i,j,c=None,alpha=None):
        colors = self.edge_collection.get_edgecolor()
        idx = self.edge_mapping[(i,j)]
        return self._update_color(colors,idx,c,alpha)

    def nodesize(self,node,nodesize):
        sizes = self.nodes_collection.get_sizes()
        if len(sizes) == 1:
            sizes = np.array([sizes[0]]*self.G._num_vertices)
        if isinstance(nodesize,float) or isinstance(nodesize,int):
            sizes[node] = nodesize
        else:
            sizes[node] = np.reshape(nodesize,len(nodesize))
        self.nodes_collection.set_sizes(sizes)
        return sizes[node]

    def nodewidth(self,node,width):
        widths = np.asarray(self.nodes_collection.get_linewidths())
        if len(widths) == 1:
            widths = np.array([widths[0]]*self.G._num_vertices)
        if isinstance(width,float) or isinstance(width,int):
            widths[node] = width
        else:
            widths[node] = np.reshape(width,len(width))
        self.nodes_collection.set_linewidths(widths)
        return widths[node]
    
    @staticmethod
    def _update_color(container,key,c,alpha):
        if c is not None:
            if c == "none":
                container[key] = c
            else:
                if alpha is not None:
                    c = to_rgba(c,alpha)
                    container[key] = c
                else:
                    c = to_rgb(c)
                    container[key,0:3] = c
        else:
            if alpha is not None:
                container[key,3] = alpha
        return container[key]

    def scatter(self,**kwargs):
        coords = self.coords
        if len(self.coords[0]) == 2:
            self.nodes_collection = self.ax.scatter([p[0] for p in coords],[p[1] for p in coords],**kwargs)
        else:
            self.nodes_collection = self.ax.scatter([p[0] for p in coords],[p[1] for p in coords],[p[2] for p in coords],**kwargs)
        
    def plot(self,**kwargs):
        if len(self.coords[0]) == 2:
            self.edge_collection = LineCollection(self.edge_pos,**kwargs)
        else:
            self.edge_collection = Line3DCollection(self.edge_pos,**kwargs)
        #make sure edges are at the bottom
        self.edge_collection.set_zorder(1)
        self.ax.add_collection(self.edge_collection)
    

