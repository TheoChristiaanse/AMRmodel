# -*- coding: utf-8 -*-
"""
This code sweeps points starting from small tspan to high
"""


import numpy as np
import pandas as pd
import pylab
import os

# Load all the data
thefile = "test_R1.txt"
df=pd.read_csv(thefile, header=None)

# Rename the headers
df.columns = ['Thot', 'Tcold','Qgr','Qnet','time','Pdrop','Qh']

# Calculate the Span
df['Tspan']=df['Thot']-df['Tcold']

# determine which hot side temperatures are considered
hotside_values = np.unique(df['Thot'])

# Which cooling powers do we want to find
Qset=[0,2.5]



def lin_interp(x_in,x0,x1,y0,y1):
    ## General linear interpolation function
    return y0+((y1-y0)/(x1-x0))*(x_in-x0)


# This saves a matrix of [nxm] content to a tab spaced out file
def FileSaveMatrix(filename, content):
    with open(filename, "a") as f:
        for line in content:
            f.write(" ".join("{}\t".format(x) for x in line))
            f.write("\n")

# this saves one line of content to a file
def FileSave(filename, content):
    with open(filename, "a") as myfile:
        myfile.write(content)

# This refreshes the files content and header.
# if the file does not excist it will make a file for you.
def refreshFile(filename,content,header):
    fN= filename
    try:
        os.remove(fN)
    except IOError:
        print("Making file")
        open(fN, 'a').close()

    FileSave(fN,header)
    FileSaveMatrix(fN,content)



for DIRECTION in ["UP","DOWN"]:
    for qset in Qset:
        ze_hot_values = []
        ze_values     = []
        for i in range(len(hotside_values)):
            # Select just the data from the hotside.
            part_df = df.loc[df['Thot']==hotside_values[i]]
            # Up Down
            if DIRECTION == "UP":
                val = 0
            if DIRECTION == "DOWN":
                val = 1
            # Make sure to have the lower values first
            part_df=part_df.sort_values(by=['Tspan'], ascending=val)
            # print(part_df.head(5))
            # Set the load value
            part_df['Qnet'] = part_df['Qnet']-qset
            # Loop through each temperature span to find where we find the intersection with 0
            shape_pd=part_df.shape
            switch_val = val
            for j in range(shape_pd[0]):
                if j>0:
                    if ((part_df.iloc[j,3]<=0) & (switch_val==1)):
                        # DOWN DIRECTION
                        if ((part_df.iloc[j-1,3]>0) & (part_df.iloc[j,3]<=0)):
                            cold_values = lin_interp(0,part_df.iloc[j-1,3],part_df.iloc[j,3],part_df.iloc[j-1,1],part_df.iloc[j,1])
                            ze_hot_values.append(hotside_values[i]-273.15)
                            ze_values.append(hotside_values[i]-cold_values)
                            switch_val = 0
                        else:
                            # if we can't find any values just set the value to NaN
                            ze_hot_values.append(hotside_values[i]-273.15)
                            ze_values.append("NaN")
                        if ((hotside_values[i]-cold_values)>20):
                            print(0,part_df.iloc[j-1,3],part_df.iloc[j,3],part_df.iloc[j-1,1],part_df.iloc[j,1],cold_values)
                        break
                    if ((part_df.iloc[j,3]>=0) & (switch_val==0)):
                        # DOWN DIRECTION
                        if ((part_df.iloc[j-1,3]<0) & (part_df.iloc[j,3]>=0)):
                            cold_values = lin_interp(0,part_df.iloc[j-1,3],part_df.iloc[j,3],part_df.iloc[j-1,1],part_df.iloc[j,1])
                            ze_hot_values.append(hotside_values[i]-273.15)
                            ze_values.append(hotside_values[i]-cold_values)
                            switch_val = 1
                        else:
                            # if we can't find any values just set the value to NaN
                            ze_hot_values.append(hotside_values[i]-273.15)
                            ze_values.append("NaN")                           
                        break
        pylab.plot(ze_hot_values, ze_values,'-', label='{:3.1f} W {:s}'.format(qset,DIRECTION))
        filename = "./OUTPUT/{}_{}_{}".format(DIRECTION,qset,thefile)
        header = "HotSide\tTempSpan\n"
        refreshFile(filename,np.column_stack((ze_hot_values,ze_values)),header)
        pylab.legend(loc='upper left')
        pylab.xlim((8,40))
        pylab.ylim((0,20))
        pylab.xlabel("Hot side Temperature") 
        pylab.ylabel("Temperature Span [K]") 

print(ze_hot_values)
print(ze_values)
