#!/usr/bin/env python
# coding: utf-8

# NC Join Adapted for Python .ipynb Breakdown

# ----- EDIT ----- #

# Location of .nc files
data_dir = '/home/user/ucla-roms/Examples/.../'

# Set the name of the output file where all netcdfs will be submitted 
odir     = './'
ofname   = '(...).nc'

import numpy as np 
from netCDF4 import Dataset, num2date
import glob as glob
import os as os
import matplotlib.pyplot as plt
import time
import multiprocessing 
from multiprocessing import Process, Pool
start_run = time.time()

# Using provided locations make a list of all netcdf files 
files = glob.glob(data_dir + '*.nc')

# Index out last file 
fname = files[-1]
print(fname)

# Dataset from last netcdf file in provided directory
nc = Dataset(fname)


# nvars is used to make an empty array for logicals, with a T/F for each variable
# in the netcdf when filled.
nvars         = len(nc.variables)
partitionable = np.zeros(nvars, dtype=bool)

output_file = odir+ofname
nco = Dataset(output_file, "w", format="NETCDF4")

# Below is a loop that runs through each variable and adds it to our output file, with dimensions of all subdomains summed

for ivar, (vname, vinfo) in enumerate(nc.variables.items()):
    vname      = vinfo.name                   # Name of the variable of focus
    # NOTE: The way datasets load in information leaves spatial data last, that is x and y are the last and 
    #       second to last, respecitively. Therefore, when we need to grab information indexing the first 
    #       spatial term would be -1, making things less intuitive. So the dimensions and their names are
    #       momentarily inverted such that x and y are now the first and second indices, making the code 
    #       more intuitive to the reader
    vdim_names = vinfo.dimensions[::-1]       # Name of the dimensions
    vdims      = np.array(vinfo.shape)[::-1]  # Size of each dimension
    # check for dimensions, i.e. is it partitionable? Only multidimensional variables may be split into subdomains
    if len(vdim_names) >= 2: 
        # Check if it is spatial data - xi and eta coordinates (x & y)
        # the in operand simply confirms whether a string contains the part we specify
        if 'xi' in vdim_names[0] and 'eta' in vdim_names[1]:
            partitionable[ivar] = 1 # since partitionable is set to bool, 1 corresponds to true! 
            # We simply save that and move on to making empty space in a new netcdf 
            # The first step is grabbing the partition of the final file. fname, and adding our variables length 
            partition = nc.partition
            # We add our dimensions to the final file, subtracting one, to get new dimensions that are the 
            # size of the desired joined output file
            vdims[0] = vdims[0]+partition[2]-1
            vdims[1] = vdims[1]+partition[3]-1
            # ROMS uses rho and u/v/w points, which do not align. Therefore we account for that below. by adjusting our 
            # gridding if the dimension names specify they are u/v/w, not rho!
            if "xi_u" in vdim_names[0]:
                vdims[0] = vdims[0] - 1
            if "eta_v" in vdim_names[1]:
                vdims[1] = vdims[1] - 1
        # Now allocate dimenions
        for l in range(len(vdims)):
            try:
                nco.createDimension(vdim_names[l], vdims[l])
            except:
                pass
        # And add them to a variable
        if len(vdims) == 2:
            nco.createVariable(vname, vinfo.datatype, (vdim_names[1], vdim_names[0])) 
            print(f'Added Variable "{vname}" with dimensions {vdims[1], vdims[0]} as {vdim_names[1], vdim_names[0]}')
        elif len(vdims) == 3:
            nco.createVariable(vname, vinfo.datatype, (vdim_names[2], vdim_names[1], vdim_names[0]), \
                               compression='zlib') #,significant_digits=4)
            print((f'Added Variable "{vname}" with dimensions {vdims[2], vdims[1], vdims[0]}')+ \
                  (f' as {vdim_names[2], vdim_names[1], vdim_names[0]}'))
        else:
            nco.createVariable(vname, vinfo.datatype, (vdim_names[3], vdim_names[2], vdim_names[1], vdim_names[0]), \
                               compression='zlib') #, significant_digits=4)
            print((f'Added Variable "{vname}" with dimensions {vdims[3], vdims[2], vdims[1], vdims[0]}')+ \
                 (f' as {vdim_names[3], vdim_names[2], vdim_names[1], vdim_names[0]}'))
    # For one dimensional products
    else:
        nco.createDimension(vname, None)      
        nco.createVariable(vname, vinfo.datatype) 
        print(f'Added Variable "{vname}"')
            
start = time.time()

for ivar, (vname, vinfo) in enumerate(nco.variables.items()):
    vname = vinfo.name
    vdim_names = vinfo.dimensions
    vdims = np.array(vinfo.shape)

    if partitionable[ivar]:
        fvar = np.zeros(vdims)
        
        for f in range(len(files)):
            nc = Dataset(files[f])
            llc = nc.partition

            # Adjust llc as needed
            if "xi_u" in vdim_names[-1]:
                llc[2] = max(llc[2]-1, 1)
            if 'eta_v' in vdim_names[-2]:
                llc[3] = max(llc[3]-1, 1)

            # Grab data to put into subdomain 
            data = nc.variables[vname][:]
            dims = data.shape

            # With the parts of the output file selected for current subdomian, add respective data            
            if len(vdims) == 2:
                fvar[llc[3]-1:llc[3] + (dims[-2]-1), llc[2]-1:llc[2] + (dims[-1]-1)] = data
            elif len(vdims) == 3:
                fvar[:, llc[3]-1:llc[3] + (dims[-2]-1), llc[2]-1:llc[2] + (dims[-1]-1)] = data
            else:
                fvar[:,:,llc[3]-1:llc[3] + (dims[-2]-1), llc[2]-1:llc[2] + (dims[-1]-1)] = data

            print(f'{vname} - File Progress = {((f+1)/len(files))*100: .2f}%', end = '\r')
            
            # Close current netcdf file 
            nc.close()

        if len(vdims) == 2:
            nco.variables[vname][:,:] = fvar
        elif len(vdims) == 3:
            nco.variables[vname][:,:,:] = fvar
        else:
            nco.variables[(vname)][:,:,:,:] = fvar
        
    else:
        nc = Dataset(files[0])
        data = nc.variables[(vname)] # Read data once
        nco.variables[vname] = data
        nc.close()

    print(f'Variable {vname} added at: {time.time() - start:.2f} seconds \n')

print(f'Total Time: {time.time() - start:.2f}')

# Now close the file since we are done editing it
nco.close()    


# In[ ]:




