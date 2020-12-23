from __future__ import print_function
import os
import socket


class Ensight:

    def __init__(self, filename='output', vector_var_names=None,
                 scalar_var_names=None, comm=None, viz_path=None):
        """
           This API can be used to output Ensight CASE files from a scientific
           simulation in Python.  It can be used in serial, or parallel, using
           the PyTrilinos syntax.

           Inputs:
             + filename -> name of output file, will have .case file extension
             + vector_var_names -> a Python list of the names of variables of
                                   vector type
             + scalar_var_names -> a Python list of the names of variables of
                                   scalar type
             + comm -> the parallel communicator
             + viz_path -> the file path to your vizualization tool, e.g.
                           ParaView

           Outputs:
             + Ensight case, geo, and variable files
        """

        if comm is not None and comm.NumProc() != 1:

            rank = comm.MyPID()

            #Create a folder to house output files
            directory = './ensight_files/'
            if not os.path.exists(directory):
                os.makedirs(directory)

            #Open geometry file
            self.__geo_file = open(directory+filename+'.'+str(rank)+'.geo',
                                   'w')

            self.__vv_names = vector_var_names
            self.__sv_names = scalar_var_names
            self.__fname = filename
            self.times = []

            #Open vector files
            if self.__vv_names is not None:
                self.__vector_var_files = [open(directory + afilename + '.' +
                                           str(rank) + '.vec', 'w')
                                           for afilename in self.__vv_names]
            #Open scalar files
            if self.__sv_names is not None:
                self.__scalar_var_files = [open(directory + afilename + '.' +
                                           str(rank) + '.scl', 'w')
                                           for afilename in self.__sv_names]

            #Write the SOS file
            self.__write_sos_file(comm, viz_path)

        else:

            #Create a folder to house output files
            directory = './ensight_files/'
            if not os.path.exists(directory):
                os.makedirs(directory)

            #Open geometry file
            self.__geo_file = open(directory+filename+'.geo','w')

            self.__vv_names = vector_var_names
            self.__sv_names = scalar_var_names
            self.__fname = filename
            self.times = []

            if self.__vv_names != None:
                self.__vector_var_files = [ open(directory+afilename+'.vec','w')
                        for afilename in self.__vv_names ]

            if self.__sv_names != None:
                self.__scalar_var_files = [ open(directory+afilename+'.scl','w')
                        for afilename in self.__sv_names ]


        return


    def write_case_file(self,comm=None):
        """Initialize Ensight case file"""

        if comm != None and comm.NumProc() != 1:

            rank = comm.MyPID()
            size = comm.NumProc()


            directory = './ensight_files/'
            self.__case_file = open(directory+self.__fname+'.'+str(rank)+'.case','w')
            print("hi")
            print('FORMAT', file=self.__case_file)
            print('type: ensight gold', file=self.__case_file)
            print('GEOMETRY', file=self.__case_file)
            print('model: 1 1 ' + self.__fname+'.'+str(rank)+'.geo', file=self.__case_file)
            print('VARIABLE', file=self.__case_file)

            if self.__vv_names != None:
                for item in self.__vv_names:
                    print(('vector per node: 1 1 ' +
                            item + ' ' + item +'.'+str(rank)+'.vec'), file=self.__case_file)

            if self.__sv_names != None:
                for item in self.__sv_names:
                    print(('scalar per node: 1 1 ' +
                            item + ' ' + item +'.'+str(rank)+'.scl'), file=self.__case_file)

            print('TIME', file=self.__case_file)
            print('time set: 1', file=self.__case_file)
            print('number of steps: ' + str(len(self.times)), file=self.__case_file)
            print('time values: ', file=self.__case_file)
            for item in self.times:
                print(item, file=self.__case_file)
            print('FILE', file=self.__case_file)
            print('file set: 1', file=self.__case_file)
            print('number of steps: ' + str(len(self.times)), file=self.__case_file)

            self.__case_file.close()

        else:

            directory = './ensight_files/'
            self.__case_file = open(directory+self.__fname+'.case','w')

            print('FORMAT', file=self.__case_file)
            print('type: ensight gold', file=self.__case_file)
            print('GEOMETRY', file=self.__case_file)
            print('model: 1 1 ' + self.__fname + '.geo', file=self.__case_file)
            print('VARIABLE', file=self.__case_file)

            if self.__vv_names != None:
                for item in self.__vv_names:
                    print(('vector per node: 1 1 ' +
                            item + ' ' + item +'.vec'), file=self.__case_file)

            if self.__sv_names != None:
                for item in self.__sv_names:
                    print(('scalar per node: 1 1 ' +
                            item + ' ' + item +'.scl'), file=self.__case_file)

            print('TIME', file=self.__case_file)
            print('time set: 1', file=self.__case_file)
            print('number of steps: ' + str(len(self.times)), file=self.__case_file)
            print('time values: ', file=self.__case_file)
            for item in self.times:
                print(item, file=self.__case_file)
            print('FILE', file=self.__case_file)
            print('file set: 1', file=self.__case_file)
            print('number of steps: ' + str(len(self.times)), file=self.__case_file)

            self.__case_file.close()

        return

    #Create Ensight Format geometry file
    def write_geometry_file_time_step(self, x, y):
        """ Initialize Ensight geometry file"""

        print('BEGIN TIME STEP', file=self.__geo_file)
        print('Ensight Gold geometry file\n', file=self.__geo_file)
        print('node id off', file=self.__geo_file)
        print('element id off', file=self.__geo_file)
        print('part', file=self.__geo_file)
        print('1', file=self.__geo_file)
        print('grid', file=self.__geo_file)
        print('coordinates', file=self.__geo_file)
        print(len(x), file=self.__geo_file)
        for item in x:
            print(item, file=self.__geo_file)
        for item in y:
            print(item, file=self.__geo_file)
        for item in range(len(x)):
            print(0.0, file=self.__geo_file)
        print('point', file=self.__geo_file)
        print(len(x), file=self.__geo_file)
        for item in range(len(x)):
            print(item + 1, file=self.__geo_file)
        print('END TIME STEP', file=self.__geo_file)

        return

    def write_vector_variable_time_step(self, variable_name, variable, time):

        write_index = None
        for index,aname in enumerate(self.__vv_names):
            if variable_name == aname:
                write_index = index
                break

        print('BEGIN TIME STEP', file=self.__vector_var_files[write_index])
        print('time = ', time, file=self.__vector_var_files[write_index])
        print('part', file=self.__vector_var_files[write_index])
        print('1', file=self.__vector_var_files[write_index])
        print('coordinates', file=self.__vector_var_files[write_index])
        for xyz in variable:
            for item in xyz:
                print(item, file=self.__vector_var_files[write_index])

        print('END TIME STEP', file=self.__vector_var_files[write_index])

        return


    def write_scalar_variable_time_step(self, variable_name, variable, time):

        write_index = None
        for index,aname in enumerate(self.__sv_names):
            if variable_name == aname:
                write_index = index
                break

        print('BEGIN TIME STEP', file=self.__scalar_var_files[write_index])
        print('time = ', time, file=self.__scalar_var_files[write_index])
        print('part', file=self.__scalar_var_files[write_index])
        print('1', file=self.__scalar_var_files[write_index])
        print('coordinates', file=self.__scalar_var_files[write_index])
        for item in variable:
            print(item, file=self.__scalar_var_files[write_index])

        print('END TIME STEP', file=self.__scalar_var_files[write_index])

        return


    def append_time_step(self,time):

        self.times.append(time)

        return


    def finalize(self):

        self.__geo_file.close()

        if self.__vv_names != None:
            for item in self.__vector_var_files:
                item.close()

        if self.__sv_names != None:
            for item in self.__scalar_var_files:
                item.close()

        return

    def __write_sos_file(self,comm=None,viz_path=None):

        if comm != None:

            rank = comm.MyPID()
            size = comm.NumProc()

            directory = './ensight_files/'

            if rank == 0:
                with open(directory+self.__fname+'.sos','w') as ff:

                    print("FORMAT", file=ff)
                    print("type: master_server gold", file=ff)
                    print("SERVERS", file=ff)
                    print("number of servers: " + str(size), file=ff)

                    for server_number in range(size):

                        print("#Server " + str(server_number), file=ff)
                        print("machine id: " + socket.gethostname(), file=ff)

                        if viz_path != None:
                            print("execuatable: " + viz_path, file=ff)
                        else:
                            print("execuatable: paraview", file=ff)
                        print(("casefile: " + self.__fname + '.'
                                + str(server_number) + '.case'), file=ff)
