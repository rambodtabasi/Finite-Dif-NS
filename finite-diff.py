#! /usr/bin/env python
# -*- coding: utf-8 -*-
#from __future__ import print_function
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

def isinteger(x):
    return np.equal(np.mod(x, 1), 0)

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
        if self.rank == 0: print("FD.py version 0.0.1\n")
        # Domain properties
        #self.iteration = 0
        self.num_nodes = num_nodes
        self.time_stepping =1e-2
        self.pressure_const = 1e-2
        self.UX = 0.157
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


        #Setup problem grid
        self.create_grid(length, width)
        #Find the global family array
        self.get_neighborhoods()
        #Initialize the neighborhood graph
        #check to see how the neighbors match
        self.__init_neighborhood_graph()
        #Load balance
        self.__load_balance()
        #Initialize the field graph
        self.__init_field_graph()
        #Initialize jacobian
        self.__init_jacobian()
        #self.__init_overlap_import_export()
        self.__init_overlap_import_export()
        #Initialize grid data structures
        self.__init_grid_data()


    def create_grid(self, length, width):
        """Private member function that creates initial rectangular grid"""

        if self.rank == 0:
            #Create grid, if width == 0, then create a 1d line of nodes
            j = np.complex(0,1)
            if width > 0.0:
                grid = np.mgrid[0:length:self.num_nodes*j,
                        0:width:self.aspect_ratio*self.num_nodes*j]
                self.nodes = np.array([grid[0].ravel(),grid[1].ravel()],
                                      dtype=np.double).T
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

        # Load balance
        if self.rank == 0:
            print("Load balancing neighborhood graph...\n")
        # Create Teuchos parameter list to pass parameters to ZOLTAN for load
        # balancing
        parameter_list = Teuchos.ParameterList()
        parameter_list.set("Partitioning Method", "RCP")
        if not self.verbose:
            parameter_sublist = parameter_list.sublist("ZOLTAN")
            parameter_sublist.set("DEBUG_LEVEL", "0")
        # Create a partitioner to load balance the graph
        partitioner = Isorropia.Epetra.Partitioner(self.neighborhood_graph,
                                                   parameter_list)
        # And a redistributer
        redistributer = Isorropia.Epetra.Redistributor(partitioner)

        # Redistribute graph and store the map
        self.balanced_neighborhood_graph = redistributer.redistribute(
                self.neighborhood_graph)
        self.balanced_map = self.balanced_neighborhood_graph.Map()

        return

    def __init_field_graph(self):

        # Assign velocity_x (ux) and velocity_y (uy) indices for each node
        self.number_of_field_variables = 2 * self.__global_number_of_nodes
        global_indices = self.balanced_map.MyGlobalElements()

        field_global_indices = np.empty(2 * global_indices.shape[0],
                                        dtype=np.int32)

        field_global_indices[0:-1:2] = 2 * global_indices
        field_global_indices[1::2] = 2 * global_indices + 1

        # create Epetra Map based on node degrees of Freedom
        self.field_balanced_map = Epetra.Map(self.number_of_field_variables,
                                             field_global_indices.tolist(),
                                             0, self.comm)
        # Instantiate the corresponding graph
        self.balanced_field_graph = Epetra.CrsGraph(Epetra.Copy,
                                                    self.field_balanced_map,
                                                    True)
        # fill the field graph
        for i in global_indices:
            # array of global indices in neighborhood of each node
            global_index_array = (self.balanced_neighborhood_graph
                                      .ExtractGlobalRowCopy(i))
            # convert global node indices to appropriate field indices
            field_index_array = (np.sort(np.r_[2*global_index_array,
                                               2*global_index_array + 1])
                                   .astype(np.int32))

            # insert rows into balanced graph per appropriate rows
            self.balanced_field_graph.InsertGlobalIndices(
                    2 * i, field_index_array)
            self.balanced_field_graph.InsertGlobalIndices(
                    2 * i + 1, field_index_array)

        # complete fill of balanced graph
        self.balanced_field_graph.FillComplete()
        # create balanced field map from balanced field neighborhood graph
        self.balanced_field_map = self.balanced_field_graph.Map()

        return


    def __init_jacobian(self):
        """
           Initialize Jacobian based on the row and column maps of the balanced
           field graph.
        """
        field_graph = self.get_balanced_field_graph()
        self.__jac = Epetra.CrsMatrix(Epetra.Copy, field_graph)

        return

    def __init_overlap_import_export(self):
        """
           Initialize Jacobian based on the row and column maps of the balanced
           neighborhood graph.
        """

        balanced_map = self.get_balanced_map()
        field_balanced_map = self.get_balanced_field_map()

        overlap_map = self.get_overlap_map()
        field_overlap_map = self.get_field_overlap_map()

        self.overlap_importer = Epetra.Import(balanced_map, overlap_map)
        self.overlap_exporter = Epetra.Export(overlap_map, balanced_map)
        self.field_overlap_importer = Epetra.Import(field_balanced_map,
                                                    field_overlap_map)
        self.field_overlap_exporter = Epetra.Export(field_overlap_map,
                                                    field_balanced_map)

        return

    def __init_grid_data(self):
        """
           Create data structures needed for doing computations
        """
        # Create some local (to function) convenience variables
        global_indices = self.balanced_map.MyGlobalElements()
        balanced_map = self.get_balanced_map()
        field_balanced_map = self.get_balanced_field_map()

        overlap_map = self.get_overlap_map()
        field_overlap_map = self.get_field_overlap_map()

        nodes_numb = self.nodes_numb

        # Store the unbalanced nodes in temporary vectors
        if self.rank == 0:
            my_x_temp = self.nodes[:, 0]
            my_y_temp = self.nodes[:, 1]
            my_field_temp = np.zeros(self.number_of_field_variables,
                                     dtype=np.double)

        else:
            my_x_temp = np.array([], dtype=np.double)
            my_y_temp = np.array([], dtype=np.double)
            my_field_temp = np.array([], dtype=np.double)

        # Create a temporary unbalanced map
        unbalanced_map = Epetra.Map(self.__global_number_of_nodes,
                                    self.__global_number_of_nodes, 0,
                                    self.comm)

        # Needed to build the combined unbalanced map to export values
        # from head node to all nodes
        field_unbalanced_map = Epetra.Map(self.number_of_field_variables,
                                          self.number_of_field_variables,
                                          0, self.comm)

        # Create the unbalanced Epetra vectors that will only be used to import
        # to the balanced x and y vectors
        my_x_unbalanced = Epetra.Vector(unbalanced_map, my_x_temp)
        my_y_unbalanced = Epetra.Vector(unbalanced_map, my_y_temp)
        my_field_unbalanced = Epetra.Vector(field_unbalanced_map,
                                            my_field_temp)

        # Create the balanced vectors
        my_x = Epetra.Vector(balanced_map)
        my_y = Epetra.Vector(balanced_map)
        my_field = Epetra.Vector(field_balanced_map)

        # Create importers
        grid_importer = Epetra.Import(balanced_map,
                                      unbalanced_map)
        field_importer = Epetra.Import(field_balanced_map,
                                       field_unbalanced_map)

        grid_overlap_importer = self.get_overlap_importer()
        field_overlap_importer = self.get_field_overlap_importer()

        # Import the unbalanced data to balanced and overlap data
        my_x.Import(my_x_unbalanced, grid_importer, Epetra.Insert)
        my_y.Import(my_y_unbalanced, grid_importer, Epetra.Insert)
        my_field.Import(my_field_unbalanced, field_importer, Epetra.Insert)

        my_x_overlap = Epetra.Vector(overlap_map)
        my_y_overlap = Epetra.Vector(overlap_map)
        my_field_overlap = Epetra.Vector(field_overlap_map)

        my_x_overlap.Import(my_x, grid_overlap_importer, Epetra.Insert)
        my_y_overlap.Import(my_y, grid_overlap_importer, Epetra.Insert)
        my_field_overlap.Import(my_field, field_overlap_importer,
                                Epetra.Insert)

        # Residual vector
        self.F_fill = Epetra.Vector(field_balanced_map)
        self.F_fill_overlap = Epetra.Vector(field_overlap_map)

        # Data for sorting/reshaping overlap field vectors
        self.global_overlap_indices = (self.get_overlap_map()
                                           .MyGlobalElements())


        #rambod
        self.ux_overlap_indices = np.where(self.global_overlap_indices%2==0)
        self.uy_overlap_indices = np.where(self.global_overlap_indices%2==1)
        self.ux_local_indices = np.where(global_indices%2==0)
        self.uy_local_indices = np.where(global_indices%2==1)



        self.sorted_local_indices = np.argsort(self.global_overlap_indices)
        self.unsorted_local_indices = np.arange(
               self.global_overlap_indices.shape[0])[self.sorted_local_indices]

        # x stride
        self.my_x_overlap_stride = (
             np.argmax(my_x_overlap[self.sorted_local_indices])
             )


        self.my_x = my_x
        self.my_y = my_y

        # This is a mess, I'm not even attempting...
        #
        # Establish Boundary Condition
        balanced_nodes = zip(self.my_x,self.my_y)
        hgs = 0.5 * self.grid_spacing
        gs = self.grid_spacing
        l = self.length
        w = self.width
        num_elements = balanced_map.NumMyElements()


        #Right BC with one horizon thickness
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

        #Left BC with one horizon thickness
        x_min_left= np.where(self.my_x >= -hgs)[0]
        x_max_left= np.where(self.my_x <= (3.0*gs+hgs))[0]
        BC_Left_Edge = np.intersect1d(x_min_left,x_max_left)
        BC_Left_Index = np.sort( BC_Left_Edge )

        BC_Left_fill = np.zeros(len(BC_Left_Edge), dtype=np.int32)
        BC_Left_fill_ux = np.zeros(len(BC_Left_Edge), dtype=np.int32)
        BC_Left_fill_uy = np.zeros(len(BC_Left_Edge), dtype=np.int32)
        for item in range(len( BC_Left_Index ) ):
            BC_Left_fill[item] = BC_Left_Index[item]
            BC_Left_fill_ux[item] = 2*BC_Left_Index[item]
            BC_Left_fill_uy[item] = 2*BC_Left_Index[item]+1


        self.BC_Left_fill_uy = BC_Left_fill_uy
        self.BC_Left_fill_ux = BC_Left_fill_uy




        #Bottom BC with one horizon thickness"""
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

        #center  bc with two horizon radius
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


    ###########################################################################
    ####################### NOX Required Functions ############################
    ###########################################################################

    def computeF(self, x, F, flag):
        """
           Implements the residual calculation as required by NOX.
        """
        try:

            #Import off processor data
            self.my_field_overlap.Import(x, self.get_field_overlap_importer(),
                                        Epetra.Insert)

            # Theses are the sorted and reshaped overlap vectors
            my_ux = (self.my_field_overlap[:-1:2][self.sorted_local_indices]
                                            .reshape(-1, self.my_x_overlap_stride))
	    my_ux_n = self.velocity_x_n[self.sorted_local_indices].reshape(-1, self.my_x_overlap_stride)
            my_uy = (self.my_field_overlap[1::2][self.sorted_local_indices]
                                            .reshape(-1, self.my_x_overlap_stride))
	    my_uy_n = self.velocity_y_n[self.sorted_local_indices].reshape(-1, self.my_x_overlap_stride)
	    #let's calculate pressure based on penalty method for incompressible flow
	    current_pressure = self.pressure_const * ((my_ux[1:-1, :-2] - my_ux[1:-1, 2:]) / 2.0 / self.delta_x +  (my_uy[:-2,1:-1] - my_uy[2:, 1:-1]) / 2.0 / self.delta_y)
            # Now we'll compute the residual
            residual = (x - self.my_field)  / self.delta_t

            # u · ∇u term, with central difference approximation of gradient
            term1_x = my_ux[1:-1, 1:-1] * (my_ux[1:-1, :-2] - my_ux[1:-1, 2:]) / 2.0 / self.delta_x
            term1_y = my_uy[1:-1, 1:-1] * (my_uy[:-2, 1:-1] - my_uy[2:, 1:-1]) / 2.0 / self.delta_y
            # Add these terms into the residual
            residual[:-1:2] += term1_x.flatten()[self.unsorted_local_indices]
            residual[1::2]  += term1_y.flatten()[self.unsorted_local_indices]

	    #second term, viscous term ,calculation
	    term2_x = visc * ((my_ux[1:-1, 2:] - 2*my_ux[1:-1,1:-1]+my_ux[1:-1,0:-2])/(self.delta_x**2)+ ((my_ux[2:,1:-1]-2*my_ux[1:-1,1:-1]+my_ux[:-2,1,-1])/seld.delta_y**2.0))
	    term2_y = visc * ((my_uy[1:-1, 2:] - 2*my_uy[1:-1,1:-1]+my_uy[1:-1,0:-2])/(self.delta_x**2)+ ((my_uy[2:,1:-1]-2*my_uy[1:-1,1:-1]+my_uy[:-2,1,-1])/seld.delta_y**2.0))
	    residual[:-1:2] -= term2_x.flatten()[self.unsorted_local_indices]
	    residual[1::2] -= term2_y.flatten()[self.unsorted_local_indices]



	    #third term, Pressure term, calculation
	    term3_x = (-1.0/rho) * ( current_pressure[1:-1,:-2]-current_pressure[1:-1,2:])/2.0/self.delta_x
	    term3_y = (-1.0/rho) * ( current_pressure[:-2,1:-1]-current_pressure[2:,1:-1])/2.0/self.delta_y
            # Do the rest of the terms
	    residual[:-1:2] -= term3_x.flatten()[self.unsorted_local_indices]
	    residual[1::2] -= term3_y.flatten()[self.unsorted_local_indices]

	    #fourth term, time derivative calculation
	    term4_x = (my_ux[1:-1,1:-1] - my_ux_n[1:-1,1:-1]) / self.time_stepping
	    term4_y = (my_uy[1:-1,1:-1] - my_uy_n[1:-1,1:-1]) / self.time_stepping
	    residual[:-1:2] += term4_x.flatten()[self.unsorted_local_indices]
	    residual[1::2] += term4_y.flatten()[self.unsorted_local_indices]


            # Put residual into my_field
            num_owned = self.get_neighborhood_graph().NumMyRows()
            self.F_fill_overlap[:] = residual

            # Export off-processor contributions to the residual
            self.F_fill_overlap.Export(self.F_fill,
                                 self.get_field_overlap_importer,
                                 Epetra.Add)

            ### velocity_x BOUNDARY CONDITION & RESIDUAL APPLICATION ###
            self.F_fill_overlap[self.ux_overlap_indices]= residual[:-1:2]
            self.F_fill_overlap[self.uy_overlap_indices]= residual[1::2]
            #Export F fill from [ghost+owned] to [owned]
            # Epetra.Add adds off processor contributions to local nodes
            self.F_fill.Export(self.F_fill_overlap, vel_overlap_importer, Epetra.Add)
            ##update residual F with F_fill
            F[:] = self.F_fill[:]

            F[self.BC_Right_fill_ux] = x[self.BC_Right_fill_ux] - 0.157
            F[self.BC_Right_fill_uy] = x[self.BC_Right_fill_uy] - 0.0
            F[self.BC_Left_fill_ux] = x[self.BC_Left_fill_ux] -  0.157
            F[self.BC_Left_fill_uy] = x[self.BC_Left_fill_uy] -  0.0
            F[self.BC_Top_fill_ux] = x[self.BC_Top_fill_ux] -  0.157
            F[self.BC_Top_fill_uy] = x[self.BC_Top_fill_uy] -  0.0
            F[self.BC_Bottom_fill_ux] = x[self.BC_Bottom_fill_ux] -  0.157
            F[self.BC_Bottom_fill_uy] = x[self.BC_Bottom_fill_uy] -  0.0
            F[self.center_fill_ux] = x[self.center_fill_ux] -0.0
            F[self.center_fill_uy] = x[self.center_fill_uy] -0.0



            self.i = self.i + 1

        except (Exception, e):
            print("Exception in PD.computeF method")
            print(e)

            return False

        return True


    # Compute Jacobian as required by NOX
    def computeJacobian(self, x, Jac):
        try:
            #print " Jacobian called "
            pass

        except (Exception, e):
            print("Exception in PD.computeJacobian method")
            print(e)
            return False

        return True


    #Getter functions
    def get_balanced_map(self):
        return self.balanced_map

    def get_balanced_field_map(self):
        return self.balanced_field_map

    def get_overlap_map(self):
        return self.balanced_neighborhood_graph.ColMap()

    def get_field_overlap_map(self):
        return self.balanced_field_graph.ColMap()

    def get_overlap_importer(self):
        return self.balanced_neighborhood_graph.Importer()
        #return self.overlap_importer

    def get_field_overlap_importer(self):
        return self.balanced_field_graph.Importer()
        #return self.field_overlap_importer

    def get_overlap_exporter(self):
        return self.balanced_neighborhood_graph.Exporter()
        #return self.overlap_exporter

    def get_field_overlap_exporter(self):
        return self.balanced_neighborhood_graph.Exporter()
        #return self.vel_overlap_exporter

    def get_balanced_neighborhood_graph(self):
        return self.balanced_neighborhood_graph

    def get_balanced_field_graph(self):
        return self.balanced_field_graph

    def get_neighborhood_graph(self):
        return self.neighborhood_graph

    def get_jacobian(self):
        return self.__jac
    def get_solution_velocity_x(self):
        return self.my_velocity_x

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
        nodes = 100
        problem = PD(nodes, 10.0)
        pressure_const = problem.pressure_const
        comm = problem.comm
        #Define the initial guess
        init_vel_guess = np.zeros([2*nodes*nodes,])
        init_vel_guess[:] = problem.UX
        ps_graph = problem.get_balanced_field_graph()
        ux_local_indices = problem.ux_local_indices
        uy_local_indices = problem.uy_local_indices
        time_stepping = problem.time_stepping
        uy_overlap_indices = problem.uy_overlap_indices
        ux_overlap_indices = problem.ux_overlap_indices
        #problem.velocity_y_n = problem.vel_overlap[uy_overlap_indices]
        neighborhood_graph = problem.get_balanced_neighborhood_graph()
        num_owned = neighborhood_graph.NumMyRows()
        #neighbors = problem.my_neighbors
        #node_number = neighbors.shape[0]
        #neighb_number = neighbors.shape[1]
        #size_upscaler = (node_number , neighb_number)
        #problem.up_scaler = np.ones(size_upscaler)
        horizon = problem.horizon
        #volumes = problem.my_volumes
        #size = ref_mag_state.shape
        #one = np.ones(size)





        #omega =one
        #problem.omega = omega
        #linear = 0

        vel_overlap_importer = problem.get_field_overlap_importer()
        field_overlap_map = problem.get_field_overlap_map()
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

        end_range=5000000
        for i in range(end_range):
            print(i)
            problem.jac_comp = True
            fdc_velocity = NOX.Epetra.FiniteDifferenceColoring(
                   nl_params, problem, init_vel_guess,
                    ps_graph, False, False)
            fdc_velocity.computeJacobian(init_vel_guess)
            jacobian = fdc_velocity_x.getUnderlyingMatrix()
            jacobian.FillComplete()
            problem.jac_comp = False
            #Create NOX solver object, solve for velocity_x and velocity_y
            if i<1:
                solver = NOX.Epetra.defaultSolver(init_vel_guess, problem,
                        problem, jacobian,nlParams = nl_params, maxIters=1,
                    wAbsTol=None, wRelTol=None, updateTol=None, absTol = 8.0e-7, relTol = None)
            else:
                solver = NOX.Epetra.defaultSolver(init_vel_guess, problem,
                    problem, jacobian,nlParams = nl_params, maxIters=100,
                    wAbsTol=None, wRelTol=None, updateTol=None, absTol = 1.0e-6, relTol = None)
            solveStatus = solver.solve()
            finalGroup = solver.getSolutionGroup()
            solution = finalGroup.getX()
            #resetting the initial conditions
            init_vel_guess[ux_local_indices]=solution[ux_local_indices]
            #start from the initial guess of zero
            init_vel_guess[uy_local_indices]= solution[uy_local_indices]
            #velocity_y_n = solution[uy_local_indices]
            problem.my_field_overlap.Import( solution, vel_overlap_importer, Epetra.Insert )
            problem.velocity_y_n = my_field_overlap[:-1:2]
            problem.velocity_x_n = my_field_overlap[1::2]
            #plotting the results
            sol_velocity_x = solution[ux_local_indices]
            sol_velocity_y = solution[uy_local_indices]
            velocity_x = problem.velocity_x_n
            velocity_y = problem.velocity_y_n



            p_out = comm.GatherAll(problem.int_pressure).flatten()
            x_out = comm.GatherAll(problem.my_x).flatten()
            y_out = comm.GatherAll(problem.my_y).flatten()
            v_x = comm.GatherAll(sol_velocity_x).flatten()
            v_y = comm.GatherAll(sol_velocity_y).flatten()

            velocity_total = (sol_velocity_y **2 + sol_velocity_x**2 )** 0.5
            v_out = comm.GatherAll(velocity_total).flatten()

            if i%100==0:
                if problem.rank==0:
                    plt.scatter(x_out,y_out,c=v_out)
                    file_name= "pressure"+"_"+str(i)
                    plt.savefig(file_name+".png")

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
    main()
