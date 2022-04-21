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


###Preparing the data====Start###


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

    np.savetxt(f"generated_data/{conflict_type}/event_mappings/event_mapping_{str(dx)}.csv" , mapping , delimiter=',')

    print("Done!")

    return None


def single_tile_events(dx , conflict_type):

    conflict_data = data_loader.conflict_data_loader(conflict_type)

    event_mappings = np.loadtxt(f"generated_data/{conflict_type}/event_mappings/event_mapping_{str(dx)}.csv" , delimiter=",")
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


def binning(time , dx , conflict_type):
    print("Creating time bins!")

    time_binning = np.loadtxt(f"generated_data/{conflict_type}/event_mappings/event_mapping_{str(dx)}.csv" , delimiter=",")
    time_binning = time_binning[:,1]

    time_binning = pd.DataFrame({'polygon_number' : time_binning})

    data = data_loader.conflict_data_loader(conflict_type)
    time_binning["date"] = data["event_date"]

    day = pd.to_datetime(data["event_date"] , dayfirst=True)  

    time_binning["days"] = (day-day.min()).apply(lambda x : x.days)

    bins = np.digitize(time_binning["days"] , bins=arange(0 , max(time_binning["days"]) + time , time))

    time_binning["bins"] = bins

    #time_binning.to_csv(f"data_{conflict_type}/time_bins_{str(time)}_{str(dx)}.csv")

    print("Done!")

    return time_binning


###Preparing the data====End###

def time_series_all_polygons(time,dx,conflict_type):    
    """Creates time series of all the valid polygons , Here time equals to the size of time bin you need in the time series""" 

    data = data_loader.conflict_data_loader(conflict_type)

    connection = duckdb.connect(database=':memory:', read_only=False)

    time_bin_data = pd.DataFrame()
    time_bin_data["event_date"] = data["event_date"]
    day = pd.to_datetime(time_bin_data["event_date"] , dayfirst=True)  
    time_bin_data["days"] = (day-day.min()).apply(lambda x : x.days)
    bins = np.digitize(time_bin_data["days"] , bins=arange(0 , max(time_bin_data["days"]) + time , time))
    time_bin_data["bins"] = bins
    time_series_len = max(time_bin_data["bins"])

    time_series = pd.DataFrame(index=range(1,time_series_len+1))       #range is from 1 because time bins were intially created such that the first time_bin value was 1

    def tile_time_series_creation(tile_number):    
        time_series[f"{tile_number}"] = 0
        file_directory = f"generated_data/{conflict_type}/{str(dx)}/{str(tile_number)}.parq"
        task = f"SELECT event_number_{conflict_type} FROM parquet_scan('{file_directory}');"
        tile_info = connection.execute(task).fetchdf()

        def update_time_series_dataframe(event_id,tile_number):
            #global time_series
            bin_on = time_bin_data.iloc[event_id]["bins"]
            time_series.loc[bin_on][f"{tile_number}"] = 1
            return None

        tile_info[f"event_number_{conflict_type}"].apply(update_time_series_dataframe , args=(tile_number,))
        return None

    path = f"generated_data/{conflict_type}/{str(dx)}"
    files = os.listdir(path)

    valid_polygons = []
    for f in files:
        valid_polygons.append(int(f.split(".")[0]))
    valid_polygons.sort()

    for i in valid_polygons:
        tile_time_series_creation(i)

    time_series.columns = time_series.columns.map(int)

    return time_series


def avalanche_creation_fast_st(time_series , time , dx  ,conflict_type):
#Here enter the time series(invert the time series if you want to calculate peace avalanche).

    gridix = 0
    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')

    data_bin = binning(time,dx,conflict_type)
    def polygon_neigbors(n):
        n = int(n)
        neighbors_local = polygons["neighbors"].iloc[n]
        neighbors_local = neighbors_local.split(",")
        neighbors_local = list(map(int , neighbors_local))
        return neighbors_local

    data_bin["neighbors"] = (data_bin["polygon_number"]).apply(polygon_neigbors)


    time_series_arr = time_series.to_numpy()
    valid_polygons = time_series.columns.to_numpy()
    neighbors = polygon_neigbors_arr(polygons , valid_polygons , "st")

    avalanche_list = avalanche_st(time_series_arr,neighbors,valid_polygons)
    avalanche_list = convert_back_to_regular_form(avalanche_list,valid_polygons)

    return avalanche_list , data_bin


def polygon_neigbors_arr(polygons , valid_polygons , type):
    """Returns an array of arrays which contains the valid naeighbors of valid polygons
    valid_polygons = time_series.columns.to_numpy()
    type == "te" if you want to enter polygons_TE and create neighbors_arr in te case. For normal polygons, type == anything else.
    """
    neighbors_list = []
    for valid in valid_polygons:
        
        neighbors_local = polygons["neighbors"].iloc[valid]
        
        if(type == "te"):
            pass
        else:
            neighbors_local = neighbors_local.split(",")
            neighbors_local = list(map(int , neighbors_local))
        
        neighbors_local = np.intersect1d(neighbors_local , valid_polygons)
        neighbors_list.append(neighbors_local)
    neighbors_arr = np.array(neighbors_list)
    return neighbors_arr


def avalanche_st(time_series_arr , neighbors , valid_polygons):
    """Input preparation code-
    time_series_arr = time_series.to_numpy()
    valid_polygons = time_series.columns.to_numpy()
    neighbors = misc_funcs.polygon_neigbors_arr(polygons , valid_polygons)
    """
    #Mapping polygon numbers to sequence of numbers from 0 to total number of valid polygons
    for neighbor_arr_index in range(len(neighbors)):
        for pol_num in range(len(neighbors[neighbor_arr_index])):
            neighbors[neighbor_arr_index][pol_num] = np.where(valid_polygons == neighbors[neighbor_arr_index][pol_num])[0][0]
    ##
    
    avalanche_list = []
    for time_step in range(len(time_series_arr)):
        initial_boxes = [(pol_index,time_step) for pol_index in np.where(time_series_arr[time_step,:] == 1)[0]]
        #print(initial_boxes , "\n")
        
        
        
        for box in initial_boxes:
            if(time_series_arr[box[1],box[0]] == 1):
                secondary_boxes = []
                avalanche_temp = []
                
                avalanche_temp.append(box)
                time_series_arr[box[1],box[0]] = 0
                
                active_neighbors = np.intersect1d(np.where(time_series_arr[box[1],:] == 1) , neighbors[box[0]])
                avalanche_temp.extend([(i,box[1]) for i in active_neighbors])
                secondary_boxes.extend([(i,box[1]) for i in active_neighbors])
                time_series_arr[box[1],active_neighbors] = 0
                
                
                if(box[1]+1 < len(time_series_arr)):
                    
                    active_neighbors = np.intersect1d(np.where(time_series_arr[box[1]+1,:] == 1) , neighbors[box[0]])
                    avalanche_temp.extend([(i,box[1]+1) for i in active_neighbors])
                    secondary_boxes.extend([(i,box[1]+1) for i in active_neighbors])
                    time_series_arr[box[1]+1,active_neighbors] = 0
                    
                    if(time_series_arr[box[1]+1,box[0]] == 1):
                        avalanche_temp.append((box[0],box[1]+1))
                        secondary_boxes.append((box[0],box[1]+1))
                        time_series_arr[box[1]+1,box[0]] = 0
                        
                
                
                for secondary_box in secondary_boxes:
                    active_neighbors = np.intersect1d(np.where(time_series_arr[secondary_box[1],:] == 1) , neighbors[secondary_box[0]])
                    avalanche_temp.extend([(i,secondary_box[1]) for i in active_neighbors])
                    secondary_boxes.extend([(i,secondary_box[1]) for i in active_neighbors])
                    time_series_arr[secondary_box[1],active_neighbors] = 0
                    
                    if(secondary_box[1]+1 < len(time_series_arr)):
                        
                        active_neighbors = np.intersect1d(np.where(time_series_arr[secondary_box[1]+1,:] == 1) , neighbors[secondary_box[0]])
                        avalanche_temp.extend([(i,secondary_box[1]+1) for i in active_neighbors])
                        secondary_boxes.extend([(i,secondary_box[1]+1) for i in active_neighbors])
                        time_series_arr[secondary_box[1]+1,active_neighbors] = 0
                        
                        if(time_series_arr[secondary_box[1]+1,secondary_box[0]] == 1):
                            avalanche_temp.append((secondary_box[0],secondary_box[1]+1))
                            secondary_boxes.append((secondary_box[0],secondary_box[1]+1))
                            time_series_arr[secondary_box[1]+1,secondary_box[0]] = 0
                            
            
                avalanche_list.append(avalanche_temp)
                
    return avalanche_list


def convert_back_to_regular_form(a , valid_polygons):
    """Here "a" is the avalanche box list that was generated from the numpy algorithm"""
    for ava_index in range(len(a)):
        ava = np.array(a[ava_index])
        ava[:,0] = [valid_polygons[i] for i in ava[:,0]]
        ava[:,1] = [i+1 for i in ava[:,1]]
        
        a[ava_index] = list(map(tuple , ava))
        
    return a
