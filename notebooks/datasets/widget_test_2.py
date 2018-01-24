#! /usr/bin/env python
# -*- coding: utf-8 -*-

# This simple example on how to do animations using graph-tool, where the layout
# changes dynamically. We start with some network, and randomly rewire its
# edges, and update the layout dynamically, where edges are rewired only if
# their euclidean distance is reduced. It is thus a very simplistic model for
# spatial segregation.

from graph_tool.all import *
from numpy.random import *
from numpy.linalg import norm
import numpy as np
import sys, os, os.path

from scipy import sparse as sp
from scipy import linalg as sp_linalg

from localgraphclustering import *

# We need some Gtk and gobject functions
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject

# Read graph. This also supports gml and graphml format.
g_gtool = load_graph("Colgate88_reduced.graphml")

remove_self_loops(g_gtool)

# Load pre-computed coordinates for nodes.
g_gtool_x = g_gtool.vertex_properties.x
g_gtool_y = g_gtool.vertex_properties.y

pos = g_gtool.new_vertex_property("vector<double>")
for i in g_gtool.get_vertices():
    pos[i] = [g_gtool_x[i],g_gtool_y[i]]

print("Done!")

# If True, the frames will be dumped to disk as images.
offscreen = sys.argv[1] == "offscreen" if len(sys.argv) > 1 else False
max_count = 5000
if offscreen and not os.path.exists("./frames"):
    os.mkdir("./frames")

# This creates a GTK+ window with the initial graph layout
if not offscreen:
    win = GraphWindow(g_gtool, pos, geometry=(1000, 1000))
else:
    win = Gtk.OffscreenWindow()
    win.set_default_size(1000, 1000)
    win.graph = GraphWidget(g_gtool, pos)
    win.add(win.graph)

# We will give the user the ability to stop the program by closing the window.
win.connect("delete_event", Gtk.main_quit)

# Actually show the window, and start the main loop.
win.show_all()
Gtk.main()
