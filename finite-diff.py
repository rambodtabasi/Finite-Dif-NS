#! /usr/bin/env python
# -*- coding: utf-8 -*-
import math
import sys

import numpy as np
import numpy.ma as ma
import scipy.spatial
import matplotlib.path as path
import matplotlib.pyplot as plt

from ensight import Ensight
#from tempfile import TemporaryFile

from PyTrilinos import Epetra
from PyTrilinos import EpetraExt
from PyTrilinos import Teuchos
from PyTrilinos import Isorropia
from PyTrilinos import NOX

#import matplotlib.pyplot as plt
import pylab
import time as ttt


#np.set_printoptions(threshold=np.nan)
## class ##
class PD(NOX.Epetra.Interface.Required,
	NOX.Epetra.Interface.Jacobian):
    """
       Class that inherits from `NOX.Epetra.Interface.Required
       <http://trilinos.sandia.gov/packages/pytrilinos/development/NOX.html>`_
       to produce the problem interface to NOX for solving steady-state
       peridynamic problems.
    """
    def __init__(self, num_nodes, length, width=10.0e-3, bc_regions=None,
            bc_values=None, symm_bcs=False, horizon=None, verbose=None):
        """Instantiate the problem object"""
        NOX.Epetra.Interface.Required.__init__(self)
        NOX.Epetra.Interface.Jacobian.__init__(self)

        #Epetra communicator attributes
        self.comm = Epetra.PyComm()
        self.rank = self.comm.MyPID()
        self.size = self.comm.NumProc()
        self.nodes_numb = num_nodes
        self.width = width
        self.length = length
	#Print version statement
        if self.rank == 0: print("PDD.py version 0.4.0zzz\n")
	# Domain properties
        #self.iteration = 0
        self.num_nodes = num_nodes
        self.time_stepping =1e-2
        self.pressure_const = 1e-2
        self.grid_spacing = float(length) / (num_nodes - 1)
        self.bc_values = bc_values
        self.symm_bcs = symm_bcs

        #self.aspect_ratio = 45.0 / num_nodes
        self.aspect_ratio = 1.0
        width = length * self.aspect_ratio
        #width = 0.0
        self.width = width

        #Default settings and attributes
        if horizon != None:
            self.horizon = horizon
        else:
            self.horizon =1.001 * self.grid_spacing


        if verbose != None:
            self.verbose = True
        else:
            self.verbose = False

        #Flow properties

        #Setup problem grid
        self.create_grid(length, width)
        #Find the global family array
        self.get_neighborhoods()
        #Initialize the neighborhood graph
        #check to see how the neighbors match
        self.__init_neighborhood_graph()
        #Load balance
        self.__load_balance()
        #Initialize jacobian
        self.__init_jacobian()
	#self.__init_overlap_import_export()
        self.__init_overlap_import_export()
        #Initialize grid data structures
        self.__init_grid_data()
    def isinteger(x):
        return np.equal(np.mod(x, 1), 0)

    def create_grid(self, length, width):
        """Private member function that creates initial rectangular grid"""

        if self.rank == 0:
            #Create grid, if width == 0, then create a 1d line of nodes
            j = np.complex(0,1)
            if width > 0.0:
                grid = np.mgrid[0:length:self.num_nodes*j,
                        0:width:self.aspect_ratio*self.num_nodes*j]
                self.nodes = np.asarray(zip(grid[0].ravel(),grid[1].ravel()),
                        dtype=np.double)
            else:
                x = np.r_[0.0:length:self.num_nodes*j]
                y = np.r_[[0.0] * self.num_nodes]
                self.nodes = np.asarray(zip(x, y),dtype=np.double)

            my_num_nodes = len(self.nodes)

        else:
            self.nodes = np.array([],dtype=np.double)
            my_num_nodes = len(self.nodes)

        self.__global_number_of_nodes = self.comm.SumAll(my_num_nodes)

        return

    def get_neighborhoods(self):
	""" cKDTree implemented for neighbor search """

        if self.rank == 0:
            #Create a kdtree to do nearest neighbor search
            tree = scipy.spatial.cKDTree(self.nodes)

            #Get all neighborhoods
            self.neighborhoods = tree.query_ball_point(self.nodes,
                    r=self.horizon, eps=0.0, p=2)
	else:
            #Setup empty data on other ranks
            self.neighborhoods = []

        return


    def __init_neighborhood_graph(self):
        """
           Creates the neighborhood ``connectivity'' graph.  This is used to
           load balanced the problem and initialize Jacobian data.
        """

        #Create the standard unbalanced map to instantiate the Epetra.CrsGraph
        #This map has all nodes on the 0 rank processor.
        standard_map = Epetra.Map(self.__global_number_of_nodes,
                len(self.nodes), 0, self.comm)
	#Compute a list of the lengths of each neighborhood list
        num_indices_per_row = np.array([ len(item)
            for item in self.neighborhoods ], dtype=np.int32)
	#Instantiate the graph
        self.neighborhood_graph = Epetra.CrsGraph(Epetra.Copy, standard_map,
                num_indices_per_row, True)
        #Fill the graph
        for rid,row in enumerate(self.neighborhoods):
	    self.neighborhood_graph.InsertGlobalIndices(rid,row)
        #Complete fill of graph
        self.neighborhood_graph.FillComplete()

	return

    def __load_balance(self):
        """Load balancing function."""

        #Load balance
        if self.rank == 0: print "Load balancing neighborhood graph...\n"
        #Create Teuchos parameter list to pass parameters to ZOLTAN for load
        #balancing
        parameter_list = Teuchos.ParameterList()
        parameter_list.set("Partitioning Method","block")
        if not self.verbose:
            parameter_sublist = parameter_list.sublist("ZOLTAN")
            parameter_sublist.set("DEBUG_LEVEL", "0")
        #Create a partitioner to load balance the graph
        partitioner = Isorropia.Epetra.Partitioner(self.neighborhood_graph,
                parameter_list)
        #And a redistributer
        redistributer = Isorropia.Epetra.Redistributor(partitioner)

        #Redistribute graph and store the map
        self.balanced_neighborhood_graph = redistributer.redistribute(
                self.neighborhood_graph)
        self.balanced_map = self.balanced_neighborhood_graph.Map()
        self.g_nodes = self.__global_number_of_nodes
        """Assign displacement and velocity_x indices for each node"""
        Number_of_Global_Variables = 2 * self.g_nodes
        Global_Indices = self.balanced_map.MyGlobalElements()

        XY_Global_Indices = np.zeros(2*len(Global_Indices),dtype = np.int32)

        for index in range(len(Global_Indices)):
            XY_Global_Indices[2*index] = 2*Global_Indices[index]
            XY_Global_Indices[2*index+1]= 2*Global_Indices[index]+1

        XY_list = XY_Global_Indices.tolist()

        #create Epetra Map based on node degrees of Freedom
        self.xy_balanced_map = Epetra.Map(Number_of_Global_Variables,
                XY_list, 0, self.comm)
	#Instantiate the corresponding graph
        self.xy_balanced_neighborhood_graph = Epetra.CrsGraph(Epetra.Copy,
                self.xy_balanced_map,True)
        #fill the XYP vaiable graph
        ### form: [Node N] >>> [X_disp_N, Y_disp_N, velocity_x_N] ###
        for index in range(len(Global_Indices)):
            #array of Global indices in neighborhood of each node
            Global_Index = np.asarray(self.balanced_neighborhood_graph
                    .ExtractGlobalRowCopy(Global_Indices[index]))
            #convert global node indices to appropriate xyp indices
            x_index = 2*Global_Index
            x_index = np.array(x_index, dtype=np.int32)
            y_index = 2*Global_Index +1
            y_index = np.array(y_index, dtype=np.int32)

            #Group and sort xyp indices in 1 array
            xy_col_indices = np.sort(np.array([x_index,y_index],
                dtype=np.int32).flatten())
            #insert colums into balanced graph per appropriate rows
            self.xy_balanced_neighborhood_graph.InsertGlobalIndices(
                    2*Global_Indices[index],xy_col_indices)
            self.xy_balanced_neighborhood_graph.InsertGlobalIndices(
                    (2*Global_Indices[index]+1),xy_col_indices)
            #completer fill of balanced grpah per appropriate rows

	self.xy_balanced_neighborhood_graph.FillComplete()
	#create balanced xyp map form balanced xyp neighborhood graph
	self.xy_balanced_map = self.xy_balanced_neighborhood_graph.Map()
        return

    def __init_jacobian(self):
        """
           Initialize Jacobian based on the row and column maps of the balanced
           neighborhood graph.
        """
        xy_graph = self.get_xy_balanced_neighborhood_graph()
	self.__jac = Epetra.CrsMatrix(Epetra.Copy, xy_graph)
        return

    def __init_overlap_import_export(self):
        """
           Initialize Jacobian based on the row and column maps of the balanced
           neighborhood graph.
        """

        balanced_map = self.get_balanced_map()
	ps_balanced_map = self.get_balanced_xy_map()

	overlap_map = self.get_overlap_map()
        vel_overlap_map = self.get_xy_overlap_map()

	self.overlap_importer = Epetra.Import( balanced_map, overlap_map)
	self.overlap_exporter = Epetra.Export(overlap_map,balanced_map)
	self.xy_overlap_importer = Epetra.Import( ps_balanced_map, vel_overlap_map)
        self.xy_overlap_exporter = Epetra.Export(vel_overlap_map , ps_balanced_map)

        return

    def __init_grid_data(self):
        """
           Create data structure needed for doing computations
        """
        #Create some local (to function) convenience variables
        balanced_map = self.get_balanced_map()
	ps_balanced_map = self.get_balanced_xy_map()

	overlap_map = self.get_overlap_map()
        vel_overlap_map = self.get_xy_overlap_map()

	overlap_importer = self.get_overlap_importer()
	vel_overlap_importer = self.get_xy_overlap_importer()

        neighborhood_graph = self.get_balanced_neighborhood_graph()
        xy_neighborhood_graph = self.get_xy_balanced_neighborhood_graph()

        nodes_numb = self.nodes_numb
        horizon = self.horizon


        #Store the unbalanced nodes in temporary x and y position vectors
        if self.rank == 0:
            my_x_temp = self.nodes[:,0]
            my_y_temp = self.nodes[:,1]

	    my_xy_temp = np.vstack( (my_x_temp, my_y_temp) ).T.flatten()
	    my_ps_temp = np.vstack( (0*my_x_temp, 0*my_y_temp) ).T.flatten()
	    #sat = np.linspace(0.4,0.6,len(my_x_temp))
	    #my_ps_temp[1::2] = sat

        else:
            my_x_temp = np.array([],dtype=np.double)
            my_y_temp = np.array([],dtype=np.double)
            my_xy_temp = np.array([],dtype=np.double)
            my_ps_temp = np.array([],dtype=np.double)

        #Create a temporary unbalanced map
        unbalanced_map = Epetra.Map(self.__global_number_of_nodes,
                len(self.nodes), 0, self.comm)

	""" Needed to build the combined unbalanced map to export values
		from head node to all nodes """
        ps_unbalanced_map = Epetra.Map(2*self.__global_number_of_nodes,
                2*len(self.nodes), 0, self.comm)

        #Create the unbalanced Epetra vectors that will only be used to import
        #to the balanced x and y vectors
        my_x_unbalanced = Epetra.Vector(unbalanced_map, my_x_temp)
	my_y_unbalanced = Epetra.Vector(unbalanced_map, my_y_temp)
	my_xy_unbalanced = Epetra.Vector(ps_unbalanced_map, my_xy_temp)
	# ADDED Jason#
	my_ps_unbalanced = Epetra.Vector(ps_unbalanced_map, my_ps_temp)


	#Create the balanced x and y vectors
	my_xy = Epetra.Vector(ps_balanced_map)

	#Create an importer
	ps_importer = Epetra.Import( ps_balanced_map, ps_unbalanced_map )

	#Import the unbalanced data to balanced data
        my_xy.Import(my_xy_unbalanced, ps_importer, Epetra.Insert)

	my_xy_overlap = Epetra.Vector(vel_overlap_map)
        my_xy_overlap.Import(my_xy, vel_overlap_importer, Epetra.Insert)

	#Query the graph to get max indices of any neighborhood graph row on
        #processor (the -1 will make the value correct after the diagonal
        #entries have been removed) from the graph
        my_row_max_entries = neighborhood_graph.MaxNumIndices() - 1

        #Query the number of rows in the neighborhood graph on processor
        my_num_rows = neighborhood_graph.NumMyRows()
	#Allocate the neighborhood array, fill with -1's as placeholders
        my_neighbors_temp = np.ones((my_num_rows, my_row_max_entries),
                dtype=np.int32) * -1
	#Extract the local node ids from the graph (except on the diagonal)
        #and fill neighborhood array
        for rid in range(my_num_rows):
            #Extract the row and remove the diagonal entry
            row = np.setdiff1d(neighborhood_graph.ExtractMyRowCopy(rid),
                    [rid], True)
	    #Compute the length of this row
            row_length = len(row)
            #Fill the neighborhood array
            my_neighbors_temp[rid, :row_length] = row

        #Convert the neighborhood array to a masked array.  This allows for
        #fast computations using numpy. Ragged Python neighborhood lists would
        #prevent this.
        self.my_neighbors = ma.masked_equal(my_neighbors_temp, -1)
        self.my_neighbors.harden_mask()
	#Create distributd vectors needed for the residual calculation
        #(owned only)

	""" velocity_x and velocity_y combined and set for import routine """

	my_ps = Epetra.Vector( ps_balanced_map)
	self.F_fill = Epetra.Vector( ps_balanced_map)

	ps_importer = Epetra.Import( ps_balanced_map, ps_unbalanced_map )
	my_ps.Import( my_ps_unbalanced, ps_importer, Epetra.Insert )

	my_vel_overlap = Epetra.Vector( vel_overlap_map )
	self.vel_overlap = Epetra.Vector( vel_overlap_map )
	my_vel_overlap.Import( my_ps, vel_overlap_importer, Epetra.Insert )
	self.F_fill_overlap = Epetra.Vector( vel_overlap_map)

        #List of Global xyp overlap indices on each rank
        ps_global_overlap_indices = vel_overlap_map.MyGlobalElements()
        #Indices of Local x, y, & p overlap indices based on Global indices
        ux_local_overlap_indices = np.where(ps_global_overlap_indices%2==0)
        uy_local_overlap_indices = np.where(ps_global_overlap_indices%2==1)

        #Extract x,y, and p overlap [owned+ghost] vectors
        my_ux_overlap = my_vel_overlap[ux_local_overlap_indices]
        my_uy_overlap = my_vel_overlap[uy_local_overlap_indices]

        #List of Global xyp indices on each rnak
        ps_global_indices = ps_balanced_map.MyGlobalElements()
        #Indices of Local x,y,& p indices based on Global indices
        ux_local_indices = np.where(ps_global_indices%2==0)
        uy_local_indices = np.where(ps_global_indices%2==1)

	my_x = my_xy[ux_local_indices]
	my_y = my_xy[uy_local_indices]

	my_x_overlap = my_xy_overlap[ux_local_overlap_indices]
	my_y_overlap = my_xy_overlap[uy_local_overlap_indices]

	#Compute reference position state of all nodes
        self.my_ref_pos_state_x = ma.masked_array(
                my_x_overlap[[self.my_neighbors]] -
                my_x_overlap[:my_num_rows,None],
                mask=self.my_neighbors.mask)
	#
	self.my_ref_pos_state_y = ma.masked_array(
                my_y_overlap[[self.my_neighbors]] -
                my_y_overlap[:my_num_rows,None],
                mask=self.my_neighbors.mask)

        #self.right_neighb = np.where(self.my_ref_pos_state_x > 0, 1, 0)
        #self.left_neighb = np.where(self.my_ref_pos_state_x<0, 1, 0)
        #self.top_neighb = np.where(self.my_ref_pos_state_y > 0, 1, 0)
        #self.bott_neighb = np.where(self.my_ref_pos_state_y < 0, 1, 0)

        width = self.width
        for i in range(len(self.my_ref_pos_state_y[:,1])):
            for j in range(len(self.my_ref_pos_state_y[1,:])):
                if self.my_ref_pos_state_y[i,j] > (self.horizon):
                    self.my_ref_pos_state_y[i,j] =  self.my_ref_pos_state_y[i,j] - width - self.grid_spacing
                if self.my_ref_pos_state_y[i,j] < -(self.horizon):
                    self.my_ref_pos_state_y[i,j] =  width + self.grid_spacing + self.my_ref_pos_state_y[i,j]
	self.my_ref_pos_state_y = ma.masked_array(self.my_ref_pos_state_y,
                mask=self.my_neighbors.mask)
	#Compute reference magnitude state of all nodes
        self.my_ref_mag_state = (self.my_ref_pos_state_x *
                self.my_ref_pos_state_x + self.my_ref_pos_state_y *
                self.my_ref_pos_state_y) ** 0.5
	#Initialize the volumes
        if self.width==0:
            self.my_volumes = np.ones_like(my_x_overlap,
                dtype=np.double) * self.grid_spacing
	    self.vol = self.grid_spacing
        else:
            self.my_volumes = np.ones_like(my_x_overlap,
                dtype=np.double) * self.grid_spacing * self.grid_spacing
	    self.vol = self.grid_spacing * self.grid_spacing


        #Extract x,y, and p [owned] vectors
        neighbor = self.my_neighbors

        my_p = my_ps[ux_local_indices]
        my_s = my_ps[uy_local_indices]
        self.velocity_y_n = my_vel_overlap[uy_local_overlap_indices]
        self.pressure = self.velocity_y_n
        self.velocity_x_n = self.velocity_y_n

	self.my_x = my_x
	self.my_y = my_y
	self.my_x_overlap = my_x_overlap
	self.my_y_overlap = my_y_overlap

	self.my_velocity_x = my_p
	self.my_velocity_y = my_s
	self.my_velocity_x_overlap = my_ux_overlap
	self.my_velocity_y_overlap = my_uy_overlap

	self.my_ps = my_ps
	self.my_vel_overlap = my_vel_overlap


	self.my_ux_resi = Epetra.Vector(balanced_map)
        self.my_ux_resi_overlap = Epetra.Vector(overlap_map)


	"ux_resi equiv. for velocity_y "
	self.my_uy_resi = Epetra.Vector(balanced_map)
        self.my_uy_resi_overlap = Epetra.Vector(overlap_map)


        self.ux_local_indices = ux_local_indices
        self.uy_local_indices = uy_local_indices
        self.ux_local_overlap_indices = ux_local_overlap_indices
        self.uy_local_overlap_indices = uy_local_overlap_indices

	self.i = 0

	# Establish Boundary Condition #
	balanced_nodes = zip(self.my_x,self.my_y)
	hgs = 0.5 * self.grid_spacing
        gs = self.grid_spacing
	l = self.length
        w = self.width
	num_elements = balanced_map.NumMyElements()


        """Right BC with one horizon thickness"""
        x_min_right = np.where(self.my_x >= l-(3.0*gs+hgs))
        x_max_right = np.where(self.my_y <= l+hgs)
        x_min_right = np.array(x_min_right)
        x_max_right = np.array(x_max_right)
	BC_Right_Edge = np.intersect1d(x_min_right,x_max_right)
	BC_Right_Index = np.sort( BC_Right_Edge )
	BC_Right_fill = np.zeros(len(BC_Right_Edge), dtype=np.int32)
	BC_Right_fill_ux = np.zeros(len(BC_Right_Edge), dtype=np.int32)
	BC_Right_fill_uy = np.zeros(len(BC_Right_Edge), dtype=np.int32)
	for item in range(len( BC_Right_Index ) ):
	    BC_Right_fill[item] = BC_Right_Index[item]
	    BC_Right_fill_ux[item] = 2*BC_Right_Index[item]
	    BC_Right_fill_uy[item] = 2*BC_Right_Index[item]+1
	self.BC_Right_fill = BC_Right_fill
	self.BC_Right_fill_ux = BC_Right_fill_ux
	self.BC_Right_fill_uy = BC_Right_fill_uy

        """ Left BC with one horizon thickness"""
        x_min_left= np.where(self.my_x >= -hgs)[0]
        x_max_left= np.where(self.my_x <= (3.0*gs+hgs))[0]
	BC_Left_Edge = np.intersect1d(x_min_left,x_max_left)
        BC_Left_Index = np.sort( BC_Left_Edge )
	BC_Left_fill = np.zeros(len(BC_Left_Edge), dtype=np.int32)
	BC_Left_fill_ux = np.zeros(len(BC_Left_Edge), dtype=np.int32)
	BC_Left_fill_uy = np.zeros(len(BC_Left_Edge), dtype=np.int32)
	for item in range(len(BC_Left_Index)):
	    BC_Left_fill[item] = BC_Left_Index[item]
	    BC_Left_fill_ux[item] = 2*BC_Left_Index[item]
	    BC_Left_fill_uy[item] = 2*BC_Left_Index[item]+1
	self.BC_Left_fill = BC_Left_fill
	self.BC_Left_fill_ux = BC_Left_fill_ux
	self.BC_Left_fill_uy = BC_Left_fill_uy
        """ Left BC with two horizon thickness"""
        x_min_left= np.where(self.my_x >= -hgs)[0]
        x_max_left= np.where(self.my_x <= (0.1))[0]
	BC_Left_Edge_double = np.intersect1d(x_min_left,x_max_left)
        BC_Left_Index_double = np.sort( BC_Left_Edge_double )
	BC_Left_fill_double = np.zeros(len(BC_Left_Edge_double), dtype=np.int32)
	BC_Left_fill_ux_double = np.zeros(len(BC_Left_Edge_double), dtype=np.int32)
	BC_Left_fill_uy_double = np.zeros(len(BC_Left_Edge_double), dtype=np.int32)
	for item in range(len(BC_Left_Index_double)):
	    BC_Left_fill_double[item] = BC_Left_Index_double[item]
	    BC_Left_fill_ux_double[item] = 2*BC_Left_Index_double[item]
	    BC_Left_fill_uy_double[item] = 2*BC_Left_Index_double[item]+1
	self.BC_Left_fill_double = BC_Left_fill_double
	self.BC_Left_fill_ux_double = BC_Left_fill_ux_double
	self.BC_Left_fill_uy_double = BC_Left_fill_uy_double
        """ inner left BC to simulate disturbance"""
        x_min_left_dist= np.where(self.my_x >= (-3.0*gs+hgs))[0]
        x_middle_left_dist= np.where(self.my_x <= (0.105))[0]
        x_max_left_dist= np.where(self.my_x <= (0.1))[0]
	#x_left_dist = np.intersect1d(x_min_left_dist,x_max_left_dist)
        x_column = np.intersect1d(x_min_left_dist, x_max_left_dist)
        y_left_dist_min = np.where(self.my_y>= 0.0*gs)
        y_left_dist_max = np.where(self.my_y<= ((l*self.aspect_ratio)-0.0*gs))
        y_left_dist_min = np.array(y_left_dist_min)
        y_left_dist_max = np.array(y_left_dist_max)
        y_column = np.intersect1d(y_left_dist_max,y_left_dist_min)
        onecolumn = np.intersect1d(y_column,x_column)
        BC_Left_Edge_dist = []
        #number of waves
        n=5.0
        for items in onecolumn:
            current_y = self.my_y[items]
            my_sin = np.sin(current_y*(n/width)*np.pi)*1.0
            my_sin = (np.absolute(my_sin)) + (0.11)
            x_max =np.where(self.my_x<=my_sin)[0]
            for everynode in x_max:
                if current_y == self.my_y[everynode]:
                    BC_Left_Edge_dist = np.append(BC_Left_Edge_dist, everynode)
        #BC_Left_Edge_dist = np.intersect1d(BC_Left_Edge_dist , x_middle_left_dist)
        BC_Left_Index_dist = np.sort( BC_Left_Edge_dist )
	BC_Left_fill_dist = np.zeros(len(BC_Left_Edge_dist), dtype=np.int32)
	BC_Left_fill_ux_dist = np.zeros(len(BC_Left_Edge_dist), dtype=np.int32)
	BC_Left_fill_uy_dist = np.zeros(len(BC_Left_Edge_dist), dtype=np.int32)
	for item in range(len(BC_Left_Index_dist)):
	    BC_Left_fill_dist[item] = BC_Left_Index_dist[item]
	    BC_Left_fill_ux_dist[item] = 2*BC_Left_Index_dist[item]
	    BC_Left_fill_uy_dist[item] = 2*BC_Left_Index_dist[item]+1
	self.BC_Left_fill_dist = BC_Left_fill_dist
	self.BC_Left_fill_ux_dist = BC_Left_fill_ux_dist
	self.BC_Left_fill_uy_dist = BC_Left_fill_uy_dist
        """Bottom BC with one horizon thickness"""
        ymin_bottom = np.where(self.my_y >= (-hgs))[0]
        ymax_bottom = np.where(self.my_y <= (3.0*gs+hgs))[0]
        BC_Bottom_Edge = np.intersect1d(ymin_bottom,ymax_bottom)
        BC_Bottom_fill = np.zeros(len(BC_Bottom_Edge), dtype=np.int32)
	BC_Bottom_fill_ux = np.zeros(len(BC_Bottom_Edge), dtype=np.int32)
	BC_Bottom_fill_uy = np.zeros(len(BC_Bottom_Edge), dtype=np.int32)
        for item in range(len( BC_Bottom_Edge)):
	    BC_Bottom_fill[item] = BC_Bottom_Edge[item]
	    BC_Bottom_fill_ux[item] = 2*BC_Bottom_Edge[item]
	    BC_Bottom_fill_uy[item] = 2*BC_Bottom_Edge[item]+1
        self.BC_Bottom_fill = BC_Bottom_fill
	self.BC_Bottom_fill_ux = BC_Bottom_fill_ux
	self.BC_Bottom_fill_uy = BC_Bottom_fill_uy

        #TOP BC with one horizon thickness
        ymin_top = np.where(self.my_y >= w-(3.0*gs+hgs))[0]
        ymax_top= np.where(self.my_y <= w+hgs)[0]
        BC_Top_Edge = np.intersect1d(ymin_top,ymax_top)
        BC_Top_fill = np.zeros(len(BC_Top_Edge), dtype=np.int32)
	BC_Top_fill_ux = np.zeros(len(BC_Top_Edge), dtype=np.int32)
        BC_Top_fill_uy = np.zeros(len(BC_Top_Edge), dtype=np.int32)
        for item in range(len( BC_Top_Edge ) ):
	    BC_Top_fill[item] = BC_Top_Edge[item]
	    BC_Top_fill_ux[item] = 2*BC_Top_Edge[item]
	    BC_Top_fill_uy[item] = 2*BC_Top_Edge[item]+1
        self.BC_Top_fill = BC_Top_fill
	self.BC_Top_fill_ux = BC_Top_fill_ux
	self.BC_Top_fill_uy = BC_Top_fill_uy
        """#center  bc with two horizon radius"""
        center = 10.0 * (((nodes_numb /2.0)-1.0)/(nodes_numb-1.0))
        size = np.size(self.my_x)
        hl = l/10
        hw = self.width/2 + self.width/20
        r = w/20
        rad = np.ones(size)
        rad = np.sqrt((self.my_x-hl)**2+(self.my_y-hw)**2)
        central_nodes = np.where(rad<=r)
        #central_nodes = np.array(central_nodes)
        #c_2 = central_nodes
        #print c_2
        #ttt.sleep(1)
        xmin_center_2=np.where((0)< self.my_x)[0]
        xmax_center_2=np.where((l) > self.my_x)[0]
        ymin_center_2=np.where((0)< self.my_y)[0]
        ymax_center_2=np.where((w)> self.my_y)[0]
        c1_2=np.intersect1d(xmin_center_2,xmax_center_2)
        c2_2= np.intersect1d(ymin_center_2,ymax_center_2)
        c_2= np.intersect1d(c1_2,c2_2)
        c_2=np.intersect1d(c_2,central_nodes)

        center_2_neighb_fill_ux = np.zeros(len(c_2), dtype=np.int32)
	center_2_neighb_fill_uy = np.zeros(len(c_2), dtype=np.int32)
        for item in range(len(c_2)):
            center_2_neighb_fill_ux[item] = c_2[item]*2.0
            center_2_neighb_fill_uy[item]=c_2[item]*2.0+1.0
        self.center_fill_uy = center_2_neighb_fill_uy
        self.center_fill_ux = center_2_neighb_fill_ux
        self.center_nodes_2 =c_2

        """ Left side of grid """
        x_min= np.where(self.my_x >=-hgs)[0]
        x_max= np.where(self.my_x <= (l - (4.0 * gs + hgs)))[0]
	BC_Edge = np.intersect1d(x_min,x_max)
        BC_Index = np.sort( BC_Edge )
	BC_fill = np.zeros(len(BC_Edge), dtype=np.int32)
	BC_fill_ux = np.zeros(len(BC_Edge), dtype=np.int32)
	BC_fill_uy = np.zeros(len(BC_Edge), dtype=np.int32)
	for item in range(len(BC_Index)):
	    BC_fill[item] = BC_Index[item]
	    BC_fill_ux[item] = 2*BC_Index[item]
	    BC_fill_uy[item] = 2*BC_Index[item]+1
	self.BC_fill_left_end = BC_fill
	self.BC_fill_left_end_p = BC_fill_ux
	self.BC_fill_left_end_s = BC_fill_uy
        """ Left side of grid """
        x_min= np.where(self.my_x >=(4.0*gs+hgs))[0]
        x_max= np.where(self.my_x <= (l - (4.0* gs+hgs)))[0]
	BC_Edge = np.intersect1d(x_min,x_max)
        BC_Index = np.sort( BC_Edge )
	BC_fill = np.zeros(len(BC_Edge), dtype=np.int32)
	BC_fill_ux = np.zeros(len(BC_Edge), dtype=np.int32)
	BC_fill_uy = np.zeros(len(BC_Edge), dtype=np.int32)
	for item in range(len(BC_Index)):
	    BC_fill[item] = BC_Index[item]
	    BC_fill_ux[item] = 2*BC_Index[item]
	    BC_fill_uy[item] = 2*BC_Index[item]+1
	self.BC_fill_right_end = BC_fill
	self.BC_fill_right_end_p = BC_fill_ux
	self.BC_fill_right_end_s = BC_fill_uy
        return

    def compute_velocity_x_y(self, velocity_x, ux_resi, velocity_y, uy_resi, flag):
        """
            Computes the peridynamic ux_resi due to non-local velocity_x
            differentials. Uses the formulation from Kayitar, Foster, & Sharma.
        """
        #print "3"
        comm = self.comm
        node_col = self.n_in_col
        neighbors = self.my_neighbors

        visc =  1.79e-3
        rho = 1000.0
        pressure = self.pressure

        #updating previous step's velocity_y values
        velocity_y_n = self.velocity_y_n
        velocity_x_n = self.velocity_x_n


        #use this to apply upwinded as needed
        #upwind_indicator_state = np.sign(velocity_state_x * ref_pos_state_x + velocity_state_y * ref_pos_state_y)


        #setup empty arrays for velocity residual calc
        residual_x = np.zeros(len(self.nodes))
        residual_y = np.zeros(len(self.nodes))


        """ finite differencing calculation starts"""

        for n in range(len(self.nodes)):
            nodex = self.nodes[n][0]
            nodey = self.nodes[n][1]
            if nodex != 0 and nodex !=self.length and nodey !=0 and nodey != self.width:
                residual_x[n] += (velocity_x[n] - velocity_x_n[n])/self.time_stepping
                residual_x[n] = velocity_x[n]*(velocity_x[n+node_col]-velocity_x[n-node_col])/(2.0*self.grid_spacing) + velocity_y[n]*(velocity_x[n+node_col]-velocity_x[n-node_col])/(2.0*self.grid_spacing)
                residual_x[n] += (-1.0/rho) * (pressure[n+node_col]- pressure[n-node_col])/(2.0 * self.grid_spacing)
                residual_x[n] -= visc * (velocity_x[n+node_col]-2*velocity_x[n]+velocity_x[n-node_col])/(self.grid_spacing **2.0)
                residual_y[n] += (velocity_y[n] - velocity_y_n[n])/self.time_stepping
                residual_y[n] = velocity_y[n]*(velocity_y[n+1]-velocity_y[n-1])/(2.0*self.grid_spacing) + velocity_x[n]*(velocity_y[n+1]-velocity_y[n-1])/(2.0*self.grid_spacing)
                residual_y[n] += (-1.0/rho) * (pressure[n+1]- pressure[n-1])/(2.0 * self.grid_spacing)
                residual_y[n] -= visc * (velocity_y[n+1]-2*velocity_y[n]+velocity_y[n-1])/(self.grid_spacing **2.0)



        #Sum the flux contribution from j nodes to i node
        uy_resi[:] = 0.0
        ux_resi[:] = 0.0
	uy_resi[:num_owned] += residual_y
	ux_resi[:num_owned] += residual_x
	return

    ###########################################################################
    ####################### NOX Required Functions ############################
    ###########################################################################

    def computeF(self, x, F, flag):
        """
           Implements the residual calculation as required by NOX.
        """
        try:

            volumes = self.my_volumes
            ref_mag_state = self.my_ref_mag_state
            ref_pos_state_x = self.my_ref_pos_state_x
            ref_pos_state_y = self.my_ref_pos_state_y
            overlap_importer = self.get_overlap_importer()
	    vel_overlap_importer = self.get_xy_overlap_importer()
            neighbors = self.my_neighbors
            omega = self.omega
            gamma_p = self.gamma_p
	    neighborhood_graph = self.get_balanced_neighborhood_graph()
	    num_owned = neighborhood_graph.NumMyRows()
	    ux_local_indices = self.ux_local_indices
	    uy_local_indices = self.uy_local_indices
	    ux_local_overlap_indices = self.ux_local_overlap_indices
	    uy_local_overlap_indices = self.uy_local_overlap_indices
            #Communicate the velocity_x (previous or boundary condition imposed)
            #to the worker vectors to be used in updating the flow


	    if self.jac_comp == True:
		self.vel_overlap = x
		x = x[:int(2.0*num_owned)]

	    if self.jac_comp == False:
		self.vel_overlap.Import(x, vel_overlap_importer,
			Epetra.Insert)

	    my_ux_overlap = self.vel_overlap[ux_local_overlap_indices]
            my_uy_overlap = self.vel_overlap[uy_local_overlap_indices]
            velocity_x = my_ux_overlap
            velocity_y = my_uy_overlap

	    n = self.balanced_map.NumMyElements()







            self.compute_velocity_x_y(my_ux_overlap, self.my_ux_resi_overlap,
                    my_uy_overlap, self.my_uy_resi_overlap, flag)
	    self.my_ux_resi.Export(self.my_ux_resi_overlap, overlap_importer,
		    Epetra.Add)
            self.my_uy_resi.Export(self.my_uy_resi_overlap,overlap_importer,
                   Epetra.Add)



	    ### velocity_x BOUNDARY CONDITION & RESIDUAL APPLICATION ###
            self.F_fill_overlap[ux_local_overlap_indices]=self.my_ux_resi_overlap
            self.F_fill_overlap[uy_local_overlap_indices]=self.my_uy_resi_overlap
            #Export F fill from [ghost+owned] to [owned]
            # Epetra.Add adds off processor contributions to local nodes
            self.F_fill.Export(self.F_fill_overlap, vel_overlap_importer, Epetra.Add)
            ##update residual F with F_fill
            F[:] = self.F_fill[:]
            #F[self.BC_Right_fill_ux] = x[self.BC_Right_fill_ux] - 0.157
            #F[self.BC_Right_fill_uy] = x[self.BC_Right_fill_uy] - 0.0
            F[self.BC_Left_fill_ux] = x[self.BC_Left_fill_ux] -  0.15
            F[self.BC_Left_fill_uy] = x[self.BC_Left_fill_uy] -  0.0
            F[self.center_fill_ux] = x[self.center_fill_ux] -0.0
            F[self.center_fill_uy] = x[self.center_fill_uy] -0.0
            F[self.BC_Right_fill_uy] = x[self.BC_Right_fill_uy] -0.0



            self.i = self.i + 1

        except Exception, e:
            print "Exception in PD.computeF method"
            print e

            return False

        return True


    # Compute Jacobian as required by NOX
    def computeJacobian(self, x, Jac):
	try:
	    #print " Jacobian called "
            pass

	except Exception, e:
	    print "Exception in PD.computeJacobian method"
	    print e
	    return False

	return True


    #Getter functions
    def get_balanced_map(self):
        return self.balanced_map

    # ADDED Jason#
    def get_balanced_xy_map(self):
        return self.xy_balanced_map

    def get_overlap_map(self):
        return self.balanced_neighborhood_graph.ColMap()
    def get_xy_overlap_map(self):
        return self.xy_balanced_neighborhood_graph.ColMap()

    def get_overlap_importer(self):
        return self.balanced_neighborhood_graph.Importer()
        #return self.overlap_importer

    def get_xy_overlap_importer(self):
        return self.xy_balanced_neighborhood_graph.Importer()
        #return self.xy_overlap_importer

    def get_overlap_exporter(self):
        return self.balanced_neighborhood_graph.Exporter()
        #return self.overlap_exporter
    def get_xy_overlap_exporter(self):
        return self.balanced_neighborhood_graph.Exporter()
        #return self.vel_overlap_exporter
    def get_balanced_neighborhood_graph(self):
        return self.balanced_neighborhood_graph
    def get_xy_balanced_neighborhood_graph(self):
        return self.xy_balanced_neighborhood_graph
    def get_neighborhood_graph(self):
        return self.neighborhood_graph
    def get_jacobian(self):
        return self.__jac
    def get_solution_velocity_x(self):
        return self.my_velocity_x
    #rambod
    def get_solution_velocity_y(self):
        return self.my_velocity_y
    def get_x(self):
        return self.my_x
    def get_y(self):
	return self.my_y
    def get_ps_init(self):
	return self.my_ps
    def get_comm(self):
        return self.comm


if __name__ == "__main__":

    def main():
	#Create the PD object
        i=0
        nodes = 10
	problem = PD(nodes,10.0)
        problem.n_in_col = nodes
        pressure_const = problem.pressure_const
        comm = problem.comm
	#Define the initial guess
	init_ps_guess = problem.get_ps_init()
   	ps_graph = problem.get_xy_balanced_neighborhood_graph()
        ux_local_indices = problem.ux_local_indices
        uy_local_indices = problem.uy_local_indices
        time_stepping = problem.time_stepping
        uy_local_overlap_indices = problem.uy_local_overlap_indices
        ux_local_overlap_indices = problem.ux_local_overlap_indices
        #problem.velocity_y_n = problem.vel_overlap[uy_local_overlap_indices]
        ref_pos_state_x = problem.my_ref_pos_state_x
        ref_pos_state_y = problem.my_ref_pos_state_y
        ref_mag_state = problem.my_ref_mag_state
	neighborhood_graph = problem.get_balanced_neighborhood_graph()
        num_owned = neighborhood_graph.NumMyRows()
        neighbors = problem.my_neighbors
        node_number = neighbors.shape[0]
        neighb_number = neighbors.shape[1]
        size_upscaler = (node_number , neighb_number)
        problem.up_scaler = np.ones(size_upscaler)
        horizon = problem.horizon
        volumes = problem.my_volumes
        size = ref_mag_state.shape
        one = np.ones(size)



        ref_mag_state = problem.my_ref_mag_state
        if problem.width ==0:
            ref_mag_state_invert = (ref_mag_state ** ( 1.0)) ** -1.0
        else:
            ref_mag_state_invert = (ref_mag_state ** ( 2.0)) ** -1.0


        """ ################ choose the right kernel function ####### """
        """ for omega = 1/ (r/horizon) """
        #omega = one
        #omega = one - (ref_mag_state/horizon)
        #problem.omega = omega
        #linear = 1
        #print omega.shape
        #plt.plot(omega[:,10])
        #plt.show()
        """ omega = 1 """
        omega =one
        problem.omega = omega
        problem.omega = omega
        linear = 0
        """ omega from delgosha """
        #x = ref_mag_state / horizon
        #omega = 34.53* (x**6) +-87.89*(x**5) + 66.976 * (x**4) - 3.9475 * (x**3) - 11.756 * (x**2) + 1.1364 * x + 0.9798
        #problem.omega = omega
        #linear = 2

	vel_overlap_importer = problem.get_xy_overlap_importer()
        vel_overlap_map = problem.get_xy_overlap_map()
        my_vel_overlap = problem.my_vel_overlap
	#Initialize and change some NOX settings
	nl_params = NOX.Epetra.defaultNonlinearParameters(problem.comm,2)
	nl_params["Line Search"]["Method"] = "Polynomial"
	ls_params = nl_params["Linear Solver"]
	ls_params["Preconditioner Operator"] = "Use Jacobian"
	ls_params["Preconditioner"] = "New Ifpack"
        #Establish parameters for ParaView Visualization
        VIZ_PATH='/Applications/paraview.app/Contents/MacOS/paraview'
        vector_variables = ['displacement']
        scalar_variables = ['velocity_x','velocity_y']
        outfile = Ensight('output',vector_variables, scalar_variables,
        problem.comm, viz_path=VIZ_PATH)
        """implement upwinding"""
        if linear ==0 :
            if problem.width==0:
                problem.gamma_c = 2.0 / ((horizon**2.0))
                problem.gamma_p = 1.0 / ((horizon**2.0))

            else:
                problem.gamma_c = 6.0 /(np.pi *(horizon**4.0))
                problem.gamma_p = 2.0 /(np.pi *(horizon**2.0))
        if linear == 1:
            if problem.width==0:
                problem.gamma_c = 9.0 / ((horizon**2.0))
                problem.gamma_p = 3.0 / ((horizon**2.0))

            else:
                problem.gamma_c = 18.0 /(np.pi *(horizon**2.0))
                problem.gamma_p = 6.0 /(np.pi *(horizon**2.0))
        if linear == 2:
            if problem.width==0:
                problem.gamma_c = 15.772870 / ((horizon**2.0))
                problem.gamma_p = 7.886435 / ((horizon**2.0))

            else:
                problem.gamma_c = 31.54574 /(np.pi *(horizon**2.0))
                problem.gamma_p = 15.77287 /(np.pi *(horizon**2.0))
        gamma_c = problem.gamma_c
        gamma_p = problem.gamma_p
        ############ Reading simulations results from previous run ##########
        #if i==0:
        #    for j in range(problem.size):
        #        if problem.rank == j:
        #            pre_sol = np.load('sol_out'+'-'+str(j)+'.npy')
        #            pre_sat = np.load('sat_out'+'-'+str(j)+'.npy')
        #    init_ps_guess[ux_local_indices]=pre_sol[ux_local_indices]
        #    init_ps_guess[uy_local_indices]= pre_sol[uy_local_indices]
        #    problem.velocity_y_n = pre_sat
        #
        end_range=5000000
        for i in range(end_range):
            print i
            """ USE Finite Difference Coloring to compute jacobian.  Distinction is made
                    between fdc and solver, as fdc handles export to overlap automatically """
	    #if i>5:
	    #	problem.time_stepping =0.5*0.000125
            problem.jac_comp = True
            fdc_velocity_x = NOX.Epetra.FiniteDifferenceColoring(
                   nl_params, problem, init_ps_guess,
                    ps_graph, False, False)
            fdc_velocity_x.computeJacobian(init_ps_guess)
            jacobian = fdc_velocity_x.getUnderlyingMatrix()
            jacobian.FillComplete()
            problem.jac_comp = False
            #Create NOX solver object, solve for velocity_x and velocity_y
	    if i<1:
            	solver = NOX.Epetra.defaultSolver(init_ps_guess, problem,
                        problem, jacobian,nlParams = nl_params, maxIters=1,
                    wAbsTol=None, wRelTol=None, updateTol=None, absTol = 8.0e-7, relTol = None)
	    else:
            	solver = NOX.Epetra.defaultSolver(init_ps_guess, problem,
                    problem, jacobian,nlParams = nl_params, maxIters=100,
                    wAbsTol=None, wRelTol=None, updateTol=None, absTol = 1.0e-6, relTol = None)
            solveStatus = solver.solve()
            finalGroup = solver.getSolutionGroup()
            solution = finalGroup.getX()
            #resetting the initial conditions
            init_ps_guess[ux_local_indices]=solution[ux_local_indices]
            #start from the initial guess of zero
            init_ps_guess[uy_local_indices]= solution[uy_local_indices]
            #velocity_y_n = solution[uy_local_indices]
            my_vel_overlap.Import( solution, vel_overlap_importer, Epetra.Insert )
            problem.velocity_y_n = my_vel_overlap[uy_local_overlap_indices]
            velocity_x_n = my_vel_overlap[ux_local_overlap_indices]
            problem.velocity_x_n = velocity_x_n
            #plotting the results
            sol_velocity_x = solution[ux_local_indices]
            sol_velocity_y = solution[uy_local_indices]
            velocity_x = problem.velocity_x_n
            velocity_y = problem.velocity_y_n

            #velocity_state_y = ma.masked_array(velocity_y[neighbors]
            #    -velocity_y[:num_owned,None], mask=neighbors.mask)
            #velocity_state_x = ma.masked_array(velocity_x[neighbors] -
            #        velocity_x[:num_owned,None], mask=neighbors.mask)
            #grad_pressure_x = pressure_const * gamma_p * omega * velocity_state_x * (ref_pos_state_x ) * ref_mag_state_invert
            #integ_grad_pressure_x = (grad_pressure_x * volumes[neighbors]).sum(axis=1)
            #grad_pressure_y = pressure_const * gamma_p * omega * velocity_state_y * (ref_pos_state_y ) * ref_mag_state_invert
            #integ_grad_pressure_y = (grad_pressure_y * volumes[neighbors]).sum(axis=1)
            #int_pressure =-1.0* (integ_grad_pressure_x + integ_grad_pressure_y)
            #problem.pressure[:num_owned] = int_pressure #+ 101325

            p_out = comm.GatherAll(problem.int_pressure).flatten()
            x_out = comm.GatherAll(problem.my_x).flatten()
            y_out = comm.GatherAll(problem.my_y).flatten()
            v_x = comm.GatherAll(sol_velocity_x).flatten()
            v_y = comm.GatherAll(sol_velocity_y).flatten()

            velocity_total = (sol_velocity_y **2 + sol_velocity_x**2 )** 0.5
            v_out = comm.GatherAll(velocity_total).flatten()

            if i%100==0:
                if problem.rank==0:
                    #plt.quiver(x_out,y_out,v_x,v_y)
                    plt.scatter(x_out,y_out,c=v_out)
                    #plt.colorbar()
                    file_name= "pressure"+"_"+str(i)
                    plt.savefig(file_name+".png")

                    #plt.show()

            ################ Write Date to Ensight Outfile #################
            time = i * problem.time_stepping
            outfile.write_geometry_file_time_step(problem.my_x, problem.my_y)

            outfile.write_vector_variable_time_step('displacement',
                                                   [0.0*problem.my_x,0.0*problem.my_y], time)
            outfile.write_scalar_variable_time_step('velocity_y',
                                                   sol_velocity_y, time)
            outfile.write_scalar_variable_time_step('velocity_x',
                                                   sol_velocity_x, time)
            outfile.append_time_step(time)
            outfile.write_case_file(comm)

            ################################################################
        outfile.finalize()

        #for i in range(problem.size):
        #    if problem.rank == i:
        #        np.save('sol_out'+'-'+str(i),solution)
        #        np.save('sat_out'+'-'+str(i),problem.velocity_y_n)
        ##plotting the results
        #x = problem.get_x()
        #s_out = comm.GatherAll(sol_velocity_y).flatten()
        #p_out = comm.GatherAll(sol_velocity_x).flatten()
        #x_out = comm.GatherAll(x).flatten()
        #if problem.rank==0:
        #    np.save('s_out'+'-'+str(i),s_out)
        #    np.save('p_out'+'-'+str(i),p_out)
        #    np.save('x_out'+'-'+str(i),x_out)
    main()

