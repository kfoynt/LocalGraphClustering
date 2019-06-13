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
    """
    This class implements all the drawing related methods for a GraphLocal
    instance. These methods include changing colors, highlighting a set etc.
    It is not designed to be used individually. Its purpose is to change all
    kinds of drawing properties after calling standard drawing functions,
    "draw" and "draw_groups" in GraphLocal.

    Attributes
    ----------
    G : GraphLocal instance

    coords : a n-by-2 or n-by-3 array with coordinates for each node of the graph.

    ax,fig : None,None (default)
            by default it will create a new figure, or this will plot in axs if not None.

    is_3d : True when it is a 3D graph

    nodes_collection : a matplotlib PathCollection instance containing all nodes

    edge_collection : a matplotlib LineCollection instance containing all edges

    groups : list[list] or list, for the first case, each sublist represents a cluster
            for the second case, list must have the same length as the number of nodes and
            nodes with the number are in the same cluster

    """
    def __init__(self,G,coords,ax=None,groups=None,figsize=None):
        self.G = G
        self.coords = coords
        self.is_3d = (len(coords[0]) == 3)
        self.edge_pos,self.edge_mapping = self._plotting_build_edgepos(G,coords)
        if ax is None:
            fig = plt.figure(figsize=figsize)
            if len(coords[0]) == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        ax.set_axis_off()

        # this gives us no border
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        self.fig = fig
        self.ax = ax
        self.nodes_collection = None
        self.edge_collection = None
        self.groups = groups


    @staticmethod
    def _plotting_push_edges_for_node(center,points,pos,edge_pos,edge_mapping):
        for i,p in enumerate(points):
            if p >= center:
                edge_mapping[(center,p)] = len(edge_pos)
                edge_pos.append([pos[center],pos[p]])

    @staticmethod
    def _plotting_build_edgepos(G,pos):
        edge_pos = []
        edge_mapping = {}
        for i in range(G._num_vertices):
            GraphDrawing._plotting_push_edges_for_node(i,G.aj[G.ai[i]:G.ai[i+1]],pos,edge_pos,edge_mapping)
        edge_pos = np.asarray(edge_pos)
        return edge_pos,edge_mapping

    def show(self):
        """
        show the graph
        """
        return self.fig

    def highlight(self,nodelist,othernodes=False,otheredges=False,circled=True,alpha=0.1):
        """
        highlight a set of nodes

        Parameters
        ----------

        nodelist: a list of nodes to be highlighted

        Optional parameters
        ------------------

        othernodes: bool (False by default)
            whether to hide nodes that is not in the nodelist

        otheredges: bool (False by default)
            whether to hide edges that doesn't connect two nodes in the nodelist

        circled: bool (False by default)
            set to True to circle nodes in the nodelist

        alpha: float (1.0 by default)
            change alpha for nodes that are not in the nodelist
        """
        nodeset = set(nodelist)
        if not othernodes or circled:
            node_out = list(set(range(self.G._num_vertices)) - nodeset)
            if not othernodes:
                self.nodecolor(node_out,alpha=alpha)
            if circled:
                self.nodecolor(nodelist,facecolor='r',edgecolor='b',alpha=1)
                curr_size = self.nodes_collection.get_sizes()[0]
                self.nodesize(nodelist,[(curr_size*1.5)**2]*len(nodelist))
                curr_width = self.nodes_collection.get_linewidths()[0]
                self.nodewidth(nodelist,[(curr_width*1.5)**2]*len(nodelist))
                #self.only_circle_nodes(nodelist)
        if not otheredges:
            for (i,j) in self.edge_mapping.keys():
                if i not in nodeset or j not in nodeset:
                    self.edgecolor(i,j,alpha=alpha)

    def only_circle_nodes(self,nodeset):
        """
        only circle the nodes in nodeset
        """
        facecolors = self.nodes_collection.get_facecolor()
        facecolors[nodeset] = [0,0,0,0]

    def between_group_alpha(self,alpha):
        """
        change the edge alpha value for edges that connect nodes from different groups
        """
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
        """
        change node color

        Parameters
        ----------
        node: integer or list[integer]

        c: string or rgb or rgba (None by default)
            when set to be None, this function just returns the current colors for the node

        edgecolor,facecolor: (None by default)
            used when you want different edgecolor and facecolor for the node
            when set to be None, it will be same as "c"

        alpha: float (None by default)
            when set to be None, alpha will not be changed

        Returns
        -------
        list of two lists, where the first is new face color and the second is new edge color, if face color
        is not changed, the first is None, if edge color is not changed, the second is None
        """
        if c is not None:
            edgecolor = c
            facecolor = c
        ret_facecolor,ret_edgecolor = None,None
        if facecolor is not None or alpha is not None:
            colors = self.nodes_collection.get_facecolor()
            # This means right now, all nodes have the same facecolor
            if colors.shape[0] == 1:
                # Firstly, we need to expand the color array so that every node has an independant facecolor
                self.nodes_collection.set_facecolor([colors[0] for i in range(self.G._num_vertices)])
                colors = self.nodes_collection.get_facecolor()
            ret_facecolor = self._plotting_update_color(colors,node,facecolor,alpha)
        if edgecolor is not None or alpha is not None:
            colors = self.nodes_collection.get_edgecolor()
            if colors.shape[0] <= 1:
                # This means right now, all nodes have hidden edges
                if colors.shape[0] == 0:
                    # Use facecolor as edgecolor
                    colors = self.nodes_collection.get_facecolor()
                # This means right now, all nodes have the same edgecolor
                if colors.shape[0] == 1:
                    # Firstly, we need to expand the color array so that every node has an independant edgecolor
                    colors = [colors[0] for i in range(self.G._num_vertices)]
                self.nodes_collection.set_edgecolor(colors)
                colors = self.nodes_collection.get_edgecolor()
            ret_edgecolor = self._plotting_update_color(colors,node,edgecolor,alpha)

        return [ret_facecolor,ret_edgecolor]

    # The better way here might be diectly modifying self.edge_collection._paths
    def edgecolor(self,i,j,c=None,alpha=None):
        """
        change edge color

        Parameters
        ----------
        i,j: integer, start and end node of the edge

        c: string or rgb or rgba (None by default)
            when set to be None, this function just returns the current colors for the edge

        alpha: float (None by default)
            when set to be None, alpha will not be changed

        Returns
        -------
        current edge color

        """
        colors = self.edge_collection.get_edgecolor()
        if len(colors) == 1:
            colors = np.array([colors[0]]*self.G._num_edges)
            self.edge_collection.set_edgecolor(c=colors)
        idx = self.edge_mapping[(i,j)]
        return self._plotting_update_color(colors,idx,c,alpha)

    def nodesize(self,node,nodesize):
        """
        change node size

        Parameters
        ----------
        node: integer or list[integer]

        nodesize: float, int, list[int] or list[float]
            in the latter two cases, the length of nodesize must
            be the same as the length of node

        Returns
        -------
        current node size

        """
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
        """
        change line width of node

        Parameters
        ----------
        node: integer or list[integer]

        width: float, int, list[int] or list[float]
            in the latter two cases, the length of nodesize must
            be the same as the length of node

        """
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
    def _plotting_update_color(container,key,c,alpha):
        if c is not None:
            if c == "none":
                # only circle the nodes
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
        """
        a wrapper of standard matplotlib scatter function

        Parameters
        ----------
        **kwargs: same as the parameters in matplotlib scatter
        """            
        coords = self.coords
        if len(self.coords[0]) == 2:
            if kwargs['zorder'] == 2:
                self.nodes_collection = self.ax.scatter([p[0] for p in coords],[p[1] for p in coords],**kwargs)
            else:
                for _s, _m, _c, _zorder, _x, _y in zip(kwargs['s'], kwargs['marker'], kwargs['c'], kwargs['zorder'], coords[:,0], coords[:,1]):
                    self.nodes_collection = self.ax.scatter(_x, _y, marker=_m, s=_s, c=_c, zorder=_zorder, cmap=kwargs['cmap'], vmin=kwargs['vmin'], vmax=kwargs['vmax'], alpha=kwargs['alpha'], edgecolors=kwargs['edgecolors'])
        else:
            self.nodes_collection = self.ax.scatter([p[0] for p in coords],[p[1] for p in coords],[p[2] for p in coords],**kwargs)
        self.ax.add_collection(self.nodes_collection)
        self.ax._sci(self.nodes_collection)

    def plot(self,**kwargs):
        """
        a wrapper of standard matplotlib plot function

        Parameters
        ----------
        **kwargs: same as the parameters in matplotlib plot
        """
        if len(self.coords[0]) == 2:
            self.edge_collection = LineCollection(self.edge_pos,**kwargs)
        else:
            self.edge_collection = Line3DCollection(self.edge_pos,**kwargs)
        #make sure edges are at the bottom
        self.edge_collection.set_zorder(1)
        self.ax.add_collection(self.edge_collection)
        self.ax._sci(self.edge_collection)
