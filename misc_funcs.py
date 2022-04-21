#from pyutils import *
from ctypes import sizeof
from datetime import date, time, timedelta
from tkinter.messagebox import NO
import geopandas
from shapely import geometry
from sklearn import neighbors
from pyutils.voronoi import *
#import pyutils.pipeline as pipe

#from misc import *
from misc.globe import *

#from workspace import *
from workspace.utils import *

from numpy import *

import datetime

import math

import  csv

import powerlaw

from vincenty import vincenty

import scipy

import duckdb

import random

#import transfer_entropy_func
#import self_loop_entropy_func
#import avalanche_numbering

import collections
import itertools

import numpy_indexed
from pandarallel import pandarallel
import fastparquet as fpar


import data as data_loader


def conflict_position(conflict_type):
    '''
    This script is used to create and then save a geodataframe of all the geographic points where events took place using lattitude and longitude of 
    events from ACLED dataset and the dates on which those events occured
    '''
    data = data_loader.conflict_data_loader(conflict_type)

    temp_list = []
    for i in range(len(data)):
        temp_point = Point(data["longitude"][i] , data["latitude"][i])
        temp_list.append(temp_point)

    conflict_event_positions = geopandas.GeoDataFrame(temp_list)
    conflict_event_positions["date"] = data["event_date"]

    conflict_event_positions.set_geometry(0 , inplace=True)
    conflict_event_positions.rename_geometry('geometry' , inplace=True)

    conflict_event_positions.to_file(f"generated_data/{conflict_type}/conflict_positions/conflict_positions.shp")

    return None


def conflict_event_polygon_mapping(dx , conflict_type , cpu_cores):
    print("Finding event to polygon mapping!")

    gridix = 0

    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')

    conflict_positions = gpd.read_file(f"generated_data/{conflict_type}/conflict_positions/conflict_positions.shp")


    pandarallel.initialize(nb_workers=cpu_cores , progress_bar=True)

    def location(point):
        ix = np.where(polygons.contains(point))[0]
        return ix[0]


    event_pol_mapping = conflict_positions["geometry"].parallel_apply(location)
    event_pol_mapping.to_numpy()

    mapping = np.zeros((len(event_pol_mapping),2))
    mapping[:,0] = list(range(len(event_pol_mapping)))
    mapping[:,1] = event_pol_mapping

    #np.savetxt(f"data_{conflict_type}/time_series_{str(dx)}.csv" , mapping , delimiter=',')

    print("Done!")

    return mapping


def single_tile_events(dx , conflict_type):

    conflict_data = data_loader.conflict_data_loader(conflict_type)

    event_mappings = conflict_event_polygon_mapping(dx,conflict_type,12)
    event_mappings = event_mappings[:,1]
    event_mappings = pd.DataFrame({'polygon_number' : event_mappings})

    conflict_data.drop(conflict_data.columns.difference(["Unnamed: 0" , "Unnamed: 0.1"]), 1, inplace=True)

    conflict_data["polygon_number"] = event_mappings["polygon_number"]

    groups = conflict_data.groupby("polygon_number")

    if(os.path.exists(f"generated_data/{conflict_type}/{str(dx)}/") == False):
        os.makedirs(f"generated_data/{conflict_type}/{str(dx)}/")

    for i,v in groups.size().items():
        d = groups.get_group(i)
        #d.drop('Unnamed: 0.1', inplace=True, axis=1)
        d.reset_index(inplace=True , drop=True)
        d.rename({'Unnamed: 0': f'event_number_{conflict_type}'}, axis=1, inplace=True)
        d.rename({'Unnamed: 0.1': f'event_number_all'}, axis=1, inplace=True)

        fpar.write(f"generated_data/{conflict_type}/{str(dx)}/{str(int(i))}.parq" , d)

    return None