#from pyutils import *
from ctypes import sizeof
from datetime import date, time, timedelta
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

import data as data_loader


def time_series_all_polygons(time,dx,conflict_type):    
    """Creates time series of all the valid polygons , Here time equals to the size of time bin you need in the time series""" 

    data = pd.read_csv(f"data_{conflict_type}/data_{conflict_type}.csv" , encoding='ISO-8859-1' , dtype={"ADMIN3": "string"}) 

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
        file_directory = f"data_{conflict_type}/{str(dx)}/{str(tile_number)}.parq"
        task = f"SELECT event_number_{conflict_type} FROM parquet_scan('{file_directory}');"
        tile_info = connection.execute(task).fetchdf()

        def update_time_series_dataframe(event_id,tile_number):
            #global time_series
            bin_on = time_bin_data.iloc[event_id]["bins"]
            time_series.loc[bin_on][f"{tile_number}"] = 1
            return None

        tile_info[f"event_number_{conflict_type}"].apply(update_time_series_dataframe , args=(tile_number,))
        return None

    path = f"data_{conflict_type}/{str(dx)}"
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

    data_bin = pd.read_csv(f"data_{conflict_type}/time_bins_{str(time)}_{str(dx)}.csv")
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



def discrete_power_law_plot(dt , xlabel):
    #For discrete quantities

    dt1 = bincount(dt)      #For getting frequency distribution
    dt1 = dt1/dt1.sum()             #For Normalization
    dt1[dt1 == 0] = np.nan
    dt1 = pd.DataFrame(dt1)
    dt1 = dt1.cumsum(skipna=True)           #To get commulaative distribution
    dt1 = (1-dt1)                    #To get complimentary commulative distribution
    dt1 = dt1[0]

    plt.scatter(np.arange(1 , dt1.size-1) , dt1[1:-1] , marker='.')
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([10**-4 , 10**0])
    ax.set_xlim([10**0 , 10**4])

    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)

    plt.xlabel(xlabel , fontsize=20)
    plt.ylabel("1-CDF" , fontsize=20)

    #plt.title(f"{str(time)},{str(dx)}")

    #plt.savefig(f"{conflict_type}_{parameter_of_interest}_{str(dx)}_{str(time)}.png")

    return None


def sites_for_box_avas(avalanches):
    """Input avalanches in box form"""
    sites = []
    for ava in avalanches:
        sites.append(len(unique(list(zip(*ava))[0])))
    return sites , "Sites"    #returns dt and xlabel

def duration_for_box_avas_2(avalanches,time):
    """Input avalanches in box form.
    
    This calculates duration in the multiples of bin size."""
    duration = []
    for ava in avalanches:
        duration.append((len(unique(list(zip(*ava))[1])) * time))
    return duration , "Duration"

def duration_for_box_avas(avalanches):
    """Input avalanches in box form.
    
    This calculates duration in bin units or the coarse grained duration (temporal analogous to sites)"""
    duration = []
    for ava in avalanches:
        duration.append(len(unique(list(zip(*ava))[1])))
    return duration , "Duration"



def avalanche_creation_fast_te(time , dx  , conflict_type , type_of_events):

    if(type_of_events == "null"):
        time_series , time_series_FG = null_model_time_series_generator(time,640,dx,conflict_type)
    elif(type_of_events == "data"):
        time_series = time_series_all_polygons(time,dx,conflict_type)
        time_series_FG = pd.read_csv(f"data_{conflict_type}/time_series_all_polygons/time_series_1_{str(dx)}.csv")

    polygons_TE , neighbor_info_dataframe , list_of_tuples_tile = neighbor_finder_TE(time_series , time ,dx ,conflict_type)
    valid_polygons = time_series.columns.to_numpy()

    neighbors_arr = polygon_neigbors_arr(polygons_TE,valid_polygons,"te")

    tiles_with_self_loop_list = self_loop_finder_TE(time_series , time , dx , conflict_type)
    time_series_arr = time_series.to_numpy()

    #Only needed to return data_bin so that I can save avalanches in event form too
    data = pd.read_csv(f"data_{conflict_type}/data_{conflict_type}.csv")
    time_bin_data = pd.DataFrame()
    time_bin_data["event_date"] = data["event_date"]
    day = pd.to_datetime(time_bin_data["event_date"] , dayfirst=True)  
    time_bin_data["days"] = (day-day.min()).apply(lambda x : x.days)
    bins = np.digitize(time_bin_data["days"] , bins=arange(0 , max(time_bin_data["days"]) + time , time))
    time_bin_data["bins"] = bins
    pol_num = np.loadtxt(f"data_{conflict_type}/time_series_{str(dx)}.csv" , delimiter=',')
    pol_num = pol_num[:,1]
    pol_num = pd.DataFrame({'polygon_TE_number' : pol_num})
    avalanche_data = time_bin_data
    avalanche_data["polygon_TE_number"] = pol_num
    avalanche_data["fatalities"] = data["fatalities"]
    avalanche_data["event_number"] = [i for i in range(len(avalanche_data))]
    data_bin = avalanche_data
    ###

    data_bin_CG = data_bin_extracter(time_series_FG,time)


    avalanche_list = avalanche_te(time_series_arr , neighbors_arr , valid_polygons , tiles_with_self_loop_list)
    avalanche_list = convert_back_to_regular_form(avalanche_list,valid_polygons)


    return avalanche_list , data_bin , data_bin_CG



def CG_time_series_fatalities(time,time_series):     #Function to geenrate a coarse grained version of a fine grained system time series
    #time = 32 #Time bin size, will be used to transform simulation snapshots to a new time coarse grained version
    #time_series = simulation_snapshots_smaller
    
    duration = len(time_series)
    bins = np.digitize(time_series.index , bins=arange(1 , duration + time , time))
    
    time_series["bins"] = bins
    
    #Creating a coarse grained time series
    time_series_CG = pd.DataFrame(index = range(1,max(time_series["bins"])+1))
    for col_num in time_series.columns[:-1]:
        #temp_pd = pd.DataFrame([0 for i in range(1,max(time_series["bins"])+1)])
        #temp_pd.rename({0:col_num} , axis="columns" , inplace=True)
        #time_series_CG = pd.concat((time_series_CG,temp_pd) , axis=1)
        time_series_CG[col_num] = 0
        
    for bin_num in pd.unique(time_series["bins"]):
        a = time_series.groupby("bins").get_group(bin_num)
        for col_num in time_series.columns[:-1]:
            time_series_CG[col_num].loc[bin_num] = time_series.groupby("bins").get_group(bin_num)[col_num].sum()
            
    #Now we can generate avalaches for random model using time_series_CG
    
    return time_series_CG






def actor_coeff(time,dx,conflict_type,algo_type):
    
    definition_type = ""
    file_type = "conflict"
    
    #New algo
    
    ava_list_path = f"data_{conflict_type}/box_algo_avas/{file_type}/{algo_type}/{algo_type}_avalanche_events_{str(time)}_{str(dx)}.csv"
    
    data = pd.read_csv(f"data_{conflict_type}/data_{conflict_type}.csv" , encoding='ISO-8859-1' , dtype={"ADMIN3": "string"}) 
    avalanche_data = avalanche_numbering.numbering(time , dx , conflict_type , ava_list_path)
    avalanche_data["fatalities"] = data["fatalities"]
    
    
    actor_coeff_list = []
    
    ava_group_size = avalanche_data.groupby("avalanche_number").size()
    filtered_index =  ava_group_size[ava_group_size.loc[ava_group_size != 1].index].index
    
    #for ava_num in range(max(avalanche_data["avalanche_number"])+1):   #For all avas
    #for ava_num in avalanche_data.groupby("avalanche_number").size().sort_values(ascending=False).head(20).index: #For top 20 avas
    for ava_num in filtered_index:
        
        actors_frame = data.iloc[avalanche_data.groupby("avalanche_number").get_group(ava_num).index][["actor1" , "actor2"]]
        actors = actors_frame.groupby("actor1").size().sort_values().index.to_list()
        actor2 = actors_frame.groupby("actor2").size().sort_values().index.to_list()
        actors.extend(actor2)
        actors = unique(actors).tolist()
        
        possible_pairs = list(itertools.combinations(actors , 2))
        counts = collections.Counter(possible_pairs)
        for i in counts:
            counts[i] = 0
        
        for event_num in range(len(actors_frame)):
            event_actors = [actors_frame.iloc[event_num]["actor1"] , actors_frame.iloc[event_num]["actor2"]]
            event_actors = tuple(np.sort(event_actors).tolist())
            counts[event_actors] += 1 
            
        sum_of_counts = np.sum(np.array(list(counts.values())))
        num_of_nonzero_counts = np.count_nonzero(np.array(list(counts.values())))
        
        actor_coeff_list.append(sum_of_counts / (num_of_nonzero_counts * len(actors_frame)))
        
    return np.mean(actor_coeff_list) , actor_coeff_list



@njit
def CG_time_series_fast(time_series_FG , col_nums , time):
    """time_series_FG in array form, col_nums = len(time_series_FG.columns)
    
    Run these lines to convert back to pandas dataframe->
    time_series_CG = time_series_CG.astype(int)
    time_series_CG = pd.DataFrame(time_series_CG , index=range(1,len(time_series_CG)+1))
    time_series_CG.columns = time_series_FG.columns
    """
    duration = len(time_series_FG)
    row_change_arr = np.arange(0 , duration + time , time)
    time_series_CG = np.zeros((len(row_change_arr)-1,col_nums))
    
    current_row = 0
    CG_row = 0
    while(current_row < duration):
        #print(current_row)
        valid_col_arr = np.asarray(np.arange(col_nums))
        
        for row_addition in np.arange(time):
            if(current_row+row_addition < duration):
                for col in valid_col_arr:
                    if(time_series_FG[current_row+row_addition,col] == 1):
                        time_series_CG[CG_row,col] = 1
                        
                        valid_col_arr = valid_col_arr[valid_col_arr != col]
                    else:
                        pass
                    
                if(len(valid_col_arr) == 0):
                    CG_row += 1
                    current_row = row_change_arr[CG_row]
                    break
                else:
                    pass 
                
            else:
                pass
            
        CG_row += 1
        current_row = row_change_arr[CG_row]
    
    return time_series_CG


def significant_links(TE_array , shuffled_TE_arr):
    return TE_array > np.percentile(shuffled_TE_arr , 95 , axis=0)

def box_str_to_tuple(box_list_path):
    """Extracts data from ava list file in box form and then outputs a list of lists which contains boxes in tuple form"""
    box_list = []
    with open(box_list_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            box_list_temp = []
            for box in row:
                a = box.replace("(","")
                a = a.replace(")","")
                a = tuple(map(int, a.split(', ')))
                
                box_list_temp.append(a)
                
            box_list.append(box_list_temp)
            
    return box_list


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


def convert_back_to_regular_form(a , valid_polygons):
    """Here "a" is the avalanche box list that was generated from the numpy algorithm"""
    for ava_index in range(len(a)):
        ava = np.array(a[ava_index])
        ava[:,0] = [valid_polygons[i] for i in ava[:,0]]
        ava[:,1] = [i+1 for i in ava[:,1]]
        
        a[ava_index] = list(map(tuple , ava))
        
    return a


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




def neighbor_finder_TE(time_series , time , dx  , conflict_type):    
    gridix = 0
    definition_type = ""
    valid_polygons = time_series.columns.to_list()
    
    TE_type = "transfer_entropy"
    number_of_shuffles = 50
    gridix = 0
    
    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    def neighbors_to_list(neighbor_list):
        return list(map(int , neighbor_list.replace(" ", "").split(",")))
    neighbor_info_dataframe = polygons.drop("geometry" , axis=1)
    neighbor_info_dataframe["neighbors_temp"] = neighbor_info_dataframe["neighbors"].apply(neighbors_to_list)
    neighbor_info_dataframe.drop("neighbors" , inplace=True , axis=1)
    neighbor_info_dataframe.rename({"neighbors_temp": "neighbors"}, axis=1 , inplace=True)
    
    #tiles_transfer_entropy = pd.read_pickle(f"data_{str(conflict_type)}/{TE_type}/tiles_{TE_type}_{definition_type}{str(time)}_{str(dx)}_{str(number_of_shuffles)}")
    
    args = (time , dx , conflict_type , number_of_shuffles , "n" , time_series)
    tiles_transfer_entropy = transfer_entropy_func.TE_tiles(*args)
    
    #For shuffled data, Creating tuple list with all neighbor pairs
    neighbor_TE_details = pd.DataFrame()
    for shuffle_number in range(number_of_shuffles+1):
        list_of_tuples_tile = []
        for index in range(len(tiles_transfer_entropy)):
            if(tiles_transfer_entropy[f"{TE_type}_{str(shuffle_number)}"].iloc[index] == ["NA"]):
                pass
            else:
                counter = 0
                for distribution_list in tiles_transfer_entropy[f"{TE_type}_{str(shuffle_number)}"].iloc[index]:
                    if(distribution_list == "NA"):
                        counter += 1
                    else:
                        tile_tuple = (index , neighbor_info_dataframe["neighbors"].iloc[index][counter] , tiles_transfer_entropy[f"{TE_type}_{str(shuffle_number)}"].iloc[index][counter])
                        list_of_tuples_tile.append(tile_tuple)
                        counter += 1
        neighbor_TE_details = pd.concat([neighbor_TE_details,pd.DataFrame(list_of_tuples_tile)], ignore_index=True, axis=1)
    
    TE_array = neighbor_TE_details[2].to_numpy()
    
    #Calculating effective TE, the standard deviation in the measurements of TEs and the mean TE of shuffled timeseries for each pair of neighbor tiles
    col_num_list = [3*i - 1 for i in range(1,number_of_shuffles+2)]
    shuffled_TE_dataframe = pd.DataFrame()
    for col_num in col_num_list[1:]:
        shuffled_TE_dataframe = pd.concat([shuffled_TE_dataframe,neighbor_TE_details[col_num]] , ignore_index=True , axis=1)
    
    shuffled_TE_arr = shuffled_TE_dataframe.transpose().to_numpy()
    
    a = significant_links(TE_array , shuffled_TE_arr)
    #a = (a*1).astype(float)
    a = (a*1)*TE_array
    a[a==0] = np.nan
    
    list_of_tuples_tile = list(zip(neighbor_TE_details[0].to_list() , neighbor_TE_details[1].to_list() , list(a)))
    
    TE_dataframe = pd.DataFrame(list_of_tuples_tile , columns=["pol_1" , "pol_2" , "TE"])
    polygons_TE = pd.DataFrame(columns=["index" , "neighbors"])
    neighbor_list = []
    pol_number_list = []
    for primary_tile_number in neighbor_info_dataframe["index"]:
        if(os.path.exists(f"data_{conflict_type}/{str(dx)}/{str(primary_tile_number)}.parq") == True):
            if(primary_tile_number in TE_dataframe["pol_1"].values):
                current_neighbours = TE_dataframe.groupby("pol_1").get_group(primary_tile_number)
                current_neighbours = current_neighbours[np.isnan(current_neighbours["TE"]) == False]
                neighbor_list.append(current_neighbours["pol_2"].to_list())
                pol_number_list.append(primary_tile_number)
    polygons_TE["index"] = pol_number_list
    polygons_TE["neighbors"] = neighbor_list
    isolated_polygon_list = []
    for primary_tile_number in neighbor_info_dataframe["index"]:
        if(os.path.exists(f"data_{conflict_type}/{str(dx)}/{str(primary_tile_number)}.parq") == True):
            if((primary_tile_number in pol_number_list) == False):
                isolated_polygon_list.append(primary_tile_number)
            else:
                pass
    #polygons_TE = polygons_TE.append(pd.DataFrame({"index":isolated_polygon_list,"neighbors":[[] for i in isolated_polygon_list]}))
    polygons_TE = pd.concat([polygons_TE,pd.DataFrame({"index":isolated_polygon_list,"neighbors":[[] for i in isolated_polygon_list]})])
    polygons_TE.sort_values("index" , inplace=True)
    polygons_TE.reset_index(inplace=True , drop=True)
    for pol in polygons["index"].to_list():
        if((pol in polygons_TE["index"].to_list()) == False):
            #polygons_TE = polygons_TE.append(pd.DataFrame({"index":pol,"neighbors":[[]]}))
            polygons_TE = pd.concat([polygons_TE,pd.DataFrame({"index":pol,"neighbors":[[]]})])
    polygons_TE.sort_values("index" , inplace=True)
    polygons_TE.reset_index(inplace=True , drop=True)
    
    return polygons_TE , neighbor_info_dataframe , list_of_tuples_tile



def self_loop_finder_TE(time_series , time , dx  , conflict_type):
    TE_type = "self_loop_entropy"
    
    number_of_shuffles = 50
    
    #tiles_transfer_entropy = pd.read_pickle(f"data_{str(conflict_type)}/{TE_type}/tiles_{TE_type}_{definition_type}{str(time)}_{str(dx)}_{str(number_of_shuffles)}")
    args = (time , dx , conflict_type , number_of_shuffles , time_series)
    tiles_transfer_entropy = self_loop_entropy_func.self_loop_entropy_calculator(*args)
    
    neighbor_TE_details = pd.DataFrame()
    for shuffle_number in range(number_of_shuffles+1):
        for index in range(len(tiles_transfer_entropy)):
            if(tiles_transfer_entropy[f"{TE_type}_{str(shuffle_number)}"].iloc[index] == ["NA"]):
                tiles_transfer_entropy[f"{TE_type}_{str(shuffle_number)}"].iloc[index] = np.nan
            else:
                tiles_transfer_entropy[f"{TE_type}_{str(shuffle_number)}"].iloc[index] = tiles_transfer_entropy[f"{TE_type}_{str(shuffle_number)}"].iloc[index][0]
    tiles_transfer_entropy = tiles_transfer_entropy.loc[tiles_transfer_entropy["self_loop_entropy_0"].isna() == False]
    tiles_transfer_entropy.reset_index(inplace=True)
    
    TE_array = tiles_transfer_entropy["self_loop_entropy_0"].to_numpy()
    pol_arr = tiles_transfer_entropy["index"].to_numpy()
    shuffle_TE_arr = tiles_transfer_entropy.iloc[:,2:].transpose().to_numpy()
    
    b = significant_links(TE_array,shuffle_TE_arr)
    tiles_with_self_loop_list = (pol_arr[b]).tolist()
    
    return tiles_with_self_loop_list
            


def avalanche_te(time_series_arr , neighbors , valid_polygons , tiles_with_self_loop_list):
    """Input preparation code-
    time_series_arr = time_series.to_numpy()
    valid_polygons = time_series.columns.to_numpy()
    neighbors = misc_funcs.polygon_neigbors_arr(polygons , valid_polygons)
    """
    #Mapping polygon numbers to sequence of numbers from 0 to total number of valid polygons
    for neighbor_arr_index in range(len(neighbors)):
        for pol_num in range(len(neighbors[neighbor_arr_index])):
            neighbors[neighbor_arr_index][pol_num] = np.where(valid_polygons == neighbors[neighbor_arr_index][pol_num])[0][0]
    
    tiles_with_self_loop = np.zeros(len(tiles_with_self_loop_list) , dtype=int)
    for tile_index in range(len(tiles_with_self_loop_list)):
        tiles_with_self_loop[tile_index] = np.where(valid_polygons == tiles_with_self_loop_list[tile_index])[0][0]
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
                
                active_neighbors = np.intersect1d(np.where(time_series_arr[box[1],:] == 1) , neighbors[box[0]]).astype(int)
                avalanche_temp.extend([(i,box[1]) for i in active_neighbors])
                secondary_boxes.extend([(i,box[1]) for i in active_neighbors])
                time_series_arr[box[1],active_neighbors] = 0
                
                
                if(box[1]+1 < len(time_series_arr)):
                    
                    active_neighbors = np.intersect1d(np.where(time_series_arr[box[1]+1,:] == 1) , neighbors[box[0]]).astype(int)
                    avalanche_temp.extend([(i,box[1]+1) for i in active_neighbors])
                    secondary_boxes.extend([(i,box[1]+1) for i in active_neighbors])
                    time_series_arr[box[1]+1,active_neighbors] = 0
                    
                    
                    if((time_series_arr[box[1]+1,box[0]] == 1) and (box[0] in tiles_with_self_loop)):
                        avalanche_temp.append((box[0],box[1]+1))
                        secondary_boxes.append((box[0],box[1]+1))
                        time_series_arr[box[1]+1,box[0]] = 0
                        
                
                
                for secondary_box in secondary_boxes:
                    active_neighbors = np.intersect1d(np.where(time_series_arr[secondary_box[1],:] == 1) , neighbors[secondary_box[0]]).astype(int)
                    avalanche_temp.extend([(i,secondary_box[1]) for i in active_neighbors])
                    secondary_boxes.extend([(i,secondary_box[1]) for i in active_neighbors])
                    time_series_arr[secondary_box[1],active_neighbors] = 0
                    
                    if(secondary_box[1]+1 < len(time_series_arr)):
                        
                        active_neighbors = np.intersect1d(np.where(time_series_arr[secondary_box[1]+1,:] == 1) , neighbors[secondary_box[0]]).astype(int)
                        avalanche_temp.extend([(i,secondary_box[1]+1) for i in active_neighbors])
                        secondary_boxes.extend([(i,secondary_box[1]+1) for i in active_neighbors])
                        time_series_arr[secondary_box[1]+1,active_neighbors] = 0
                        
                        
                        if((time_series_arr[secondary_box[1]+1,secondary_box[0]] == 1) and (secondary_box[0] in tiles_with_self_loop)):
                            avalanche_temp.append((secondary_box[0],secondary_box[1]+1))
                            secondary_boxes.append((secondary_box[0],secondary_box[1]+1))
                            time_series_arr[secondary_box[1]+1,secondary_box[0]] = 0
                            
            
                avalanche_list.append(avalanche_temp)
                
    return avalanche_list



def significant_links_TE_calculator(time,dx,conflict_type):
    gridix = 0
    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    time_series = time_series_all_polygons(time,dx,conflict_type)
    valid_polygons = time_series.columns.to_numpy()
    
    neighbors_arr = polygon_neigbors_arr(polygons,valid_polygons,"st")
    total_possible_links = np.sum([len(i) for i in neighbors_arr])
    
    polygons_TE , neighbor_info_dataframe , list_of_tuples_tile = neighbor_finder_TE(time_series,time,dx,conflict_type)

    neighbors_arr = polygon_neigbors_arr(polygons_TE,valid_polygons,"te")
    total_significant_links = np.sum([len(i) for i in neighbors_arr])
    
    return total_significant_links , total_possible_links , (total_significant_links/total_possible_links)


def sl_tiles_TE_calculator(time,dx,conflict_type):
    time_series = time_series_all_polygons(time,dx,conflict_type)
    valid_polygons = time_series.columns.to_numpy()
    
    sl_tiles_list = self_loop_finder_TE(time_series,time,dx,conflict_type)
    total_sl_tiles = len(sl_tiles_list)
    
    total_tiles = len(valid_polygons)
    
    return total_sl_tiles , total_tiles , (total_sl_tiles/total_tiles)


def random_links_te_avalanche_generator(time,dx,conflict_type,number_of_links,number_of_sl_tiles):
    """number_of_links = significant_links_TE_calculator(time,dx,conflict_type)
    number_of_sl_tiles = sl_tiles_TE_calculator(time,dx,conflict_type)
    
    This function will return a list of ava in box format along with data_bin for event ava creation.
    It also returns the randomly selected tiles which have a link between then, in case you wanna to visualize the links
    """
    gridix = 0

    ts = time_series_all_polygons(time,dx,conflict_type)
    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    
    #Generating random links
    
    valid_polygons = ts.columns.to_numpy()
    neighbors = polygon_neigbors_arr(polygons,valid_polygons,"st")
    
    a = pd.DataFrame(neighbors)
    a.set_index(valid_polygons , inplace=True)
    
    all_neighbor_pairs = [(i,k) for i in valid_polygons for j in a.loc[i] for k in j]
    all_neighbor_pairs = np.array(all_neighbor_pairs)
    
    random_link_index = np.random.choice(np.arange(len(all_neighbor_pairs)) , size=number_of_links , replace=False)
    random_link_index = sort(random_link_index)
    
    random_links = all_neighbor_pairs[random_link_index]
    
    #Creating random neighbors_arr
    polygons_random = unique(random_links[:,0])

    random_links = pd.DataFrame(random_links)
    
    neighbors_random = []
    for tile in valid_polygons:
        if(tile in polygons_random):
            neighbors_random.append(random_links.groupby(0).get_group(tile)[1].to_numpy())
        else:
            neighbors_random.append(np.array([]))
        
    neighbors_random = np.array(neighbors_random)   
    
    
    #Creating random self loop tiles

    tiles_with_self_loop_random = np.random.choice(valid_polygons , size=number_of_sl_tiles , replace=False)
    
    #Generating avalanches
    time_series = ts.to_numpy()
    ava = avalanche_te(time_series,neighbors_random,valid_polygons,tiles_with_self_loop_random)
    ava = convert_back_to_regular_form(ava,valid_polygons)


    ##To return data_bin in order to convert back to event form(optimize this part as a lot of useless information is also provided here)
    data = pd.read_csv(f"data_{conflict_type}/data_{conflict_type}.csv")
    time_bin_data = pd.DataFrame()
    time_bin_data["event_date"] = data["event_date"]
    day = pd.to_datetime(time_bin_data["event_date"] , dayfirst=True)  
    time_bin_data["days"] = (day-day.min()).apply(lambda x : x.days)
    bins = np.digitize(time_bin_data["days"] , bins=arange(0 , max(time_bin_data["days"]) + time , time))
    time_bin_data["bins"] = bins
    pol_num = np.loadtxt(f"data_{conflict_type}/time_series_{str(dx)}.csv" , delimiter=',')
    pol_num = pol_num[:,1]
    pol_num = pd.DataFrame({'polygon_TE_number' : pol_num})
    avalanche_data = time_bin_data
    avalanche_data["polygon_TE_number"] = pol_num
    avalanche_data["fatalities"] = data["fatalities"]
    avalanche_data["event_number"] = [i for i in range(len(avalanche_data))]
    data_bin = avalanche_data
    ##
    
    return ava , data_bin , neighbors_random


def boxAva_to_eventAva(avalanches_box,data_bin):
    avalanches_event = []
    for index in range(len(avalanches_box)):
        ava = []
        for box in avalanches_box[index]:
            ava.extend(data_bin.groupby(data_bin.columns.values[2]).get_group(box[0]).groupby("bins").get_group(box[1]).index.to_list())
        avalanches_event.append(ava)
        
    return avalanches_event


def ava_numbering(ava_events,time,dx,conflict_type):
    """Input avalanche list in event form and get a data table with avalanche
    number mapped to each conflict event.
    """
    full_data = pd.read_csv(f"data_{conflict_type}/data_{conflict_type}.csv" , encoding='ISO-8859-1' , dtype={"ADMIN3": "string"})
    data = pd.read_csv(f"data_{conflict_type}/time_bins_{str(time)}_{str(dx)}.csv")
    data["avalanche_number"] = 0
    
    def avalanche_numbering(ava_num , i):
        data["avalanche_number"].iloc[ava_num] = i
        return None 
    
    i = 0
    for ava in ava_events:
        ava = pd.DataFrame(ava)
        ava.apply(avalanche_numbering , args=(i,))
        
        i += 1

    data["fatalities"] = full_data["fatalities"]
        
    return data


def null_model(dx_primary,dx_interest,conflict_type,prob_type,cpu_cores):
    """Returns simulation time series of primary dx and the dx you are interested in(in dataframe form).
    prob_type = "reassign" for reassigned probabilities.
    
    dx_primary > dx_interest."""
    
    gridix = 0
    
    #time_series = misc_funcs.time_series_all_polygons(1,dx_smaller,conflict_type) #If you want to generate time series right now
    #time_series.to_csv(f"data_{conflict_type}/time_series_all_polygons/time_series_1_{str(dx_smaller)}.csv" , index=False) #So save the generated time series
    
    
    time_series_FG = pd.read_csv(f"data_{conflict_type}/time_series_all_polygons/time_series_1_{str(dx_primary)}.csv")
    time_series_FG_arr = time_series_FG.to_numpy()
    
    polygons_primary = gpd.read_file(f'voronoi_grids/{dx_primary}/borders{str(gridix).zfill(2)}.shp')
    polygons_interest = gpd.read_file(f'voronoi_grids/{dx_interest}/borders{str(gridix).zfill(2)}.shp')

    valid_polygons_primary = time_series_FG.columns.to_numpy().astype(int)
    polygons_primary_valid = polygons_primary.iloc[valid_polygons_primary]
    
    prob_arr = np.zeros((len(valid_polygons_primary),2))
    prob_arr[:,0] = valid_polygons_primary
    
    for i in range(len(valid_polygons_primary)):
        on_off_groups = numpy_indexed.group_by(time_series_FG_arr[:,i]).split_array_as_list(time_series_FG_arr[:,i])
        
        prob_arr[i,1] = len(on_off_groups[1]) / len(time_series_FG_arr)
        
    if(prob_type == "reassign"):
        np.random.shuffle(prob_arr[:,1])
        
        
    def dx_mapping(centroid_point):
        return np.where(polygons_interest.contains(centroid_point))[0][0]
    
    pandarallel.initialize(nb_workers=cpu_cores , progress_bar=False)
    
    mapping = (polygons_primary_valid["geometry"].centroid).parallel_apply(dx_mapping)
    
    valid_polygons_interest = np.sort(np.unique(mapping.to_numpy()))
    
    mapping_arr = np.zeros((len(mapping),2)).astype(int)
    mapping_arr[:,0] = mapping.index
    mapping_arr[:,1] = mapping.to_numpy()
    
    
    simulation_time_series_FG_primary = np.zeros(time_series_FG_arr.shape)
    simulation_time_series_FG_interest = np.zeros((len(time_series_FG_arr),len(valid_polygons_interest)))
    
    for i in range(len(prob_arr)):
        prob = prob_arr[i,1]
        on_locations = np.where(np.array([random.choices([0,1] , [1-prob,prob])[0] for j in range(len(simulation_time_series_FG_primary))]) == 1)[0]
        
        if(len(on_locations) != 0):
            simulation_time_series_FG_primary[on_locations,i] = 1
            simulation_time_series_FG_interest[on_locations,np.where(valid_polygons_interest == mapping_arr[i,1])[0][0]] = 1
          
        
    simulation_time_series_FG_primary = pd.DataFrame(simulation_time_series_FG_primary,columns=valid_polygons_primary).astype(int)
    simulation_time_series_FG_primary.index = range(1,len(simulation_time_series_FG_primary)+1)
    
    simulation_time_series_FG_interest = pd.DataFrame(simulation_time_series_FG_interest,columns=valid_polygons_interest).astype(int)
    simulation_time_series_FG_interest.index = range(1,len(simulation_time_series_FG_interest)+1)
    
    
    return simulation_time_series_FG_primary , simulation_time_series_FG_interest



def null_model_time_series_generator(time,dx_primary,dx_interest,conflict_type):
    #cpu_cores = int(input("Enter # of cpu_cores to use: "))
    cpu_cores = 3
    x,time_series_FG_interest = null_model(dx_primary,dx_interest,conflict_type,"reassign",cpu_cores)
    time_series_FG = time_series_FG_interest
    
    time_series_FG_arr = time_series_FG.to_numpy()
    col_nums = len(time_series_FG.columns)
    
    time_series_CG = CG_time_series_fast(time_series_FG_arr,col_nums,time)
    
    time_series_CG = time_series_CG.astype(int)
    time_series_CG = pd.DataFrame(time_series_CG , index=range(1,len(time_series_CG)+1))
    time_series_CG.columns = time_series_FG.columns
    
    return time_series_CG , time_series_FG


def spreading_pathways_list(neighbor_info_dataframe,list_of_tuples_tiles , time , dx , conflict_type):
    """Returns a list of pathways via which a conflict can spread in a particular causal network"""

    
    #Use unsplit(original) list_of_tuples_tiles 
    
    TE_dataframe = pd.DataFrame(list_of_tuples_tiles , columns=["pol_1" , "pol_2" , "TE"])
    
    cluster_list = []
    
    for primary_tile_number in neighbor_info_dataframe["index"]:
        if(os.path.exists(f"data_{conflict_type}/{str(dx)}/{str(primary_tile_number)}.parq") == True):
            if(primary_tile_number in TE_dataframe["pol_1"].values):
                temp_neighbour_list = []
                temp_cluster_list = []
                
                current_neighbours = TE_dataframe.groupby("pol_1").get_group(primary_tile_number)
                current_neighbours = current_neighbours[np.isnan(current_neighbours["TE"]) == False]
                temp_neighbour_list.extend(current_neighbours["pol_2"].to_list())
                
                
                if(len(temp_neighbour_list) != 0):
                    temp_cluster_list.append(primary_tile_number)
                    
                    for neighbour_tile in temp_neighbour_list:
                        temp_cluster_list.append(neighbour_tile)
                        if(neighbour_tile in TE_dataframe["pol_1"].values):
                            current_neighbours = TE_dataframe.groupby("pol_1").get_group(neighbour_tile)
                            current_neighbours = current_neighbours[np.isnan(current_neighbours["TE"]) == False]
                            
                            temp_list = current_neighbours["pol_2"].to_list()
                            temp_list = [i for i in temp_list if((i in temp_cluster_list) == False)]
                            temp_list = [i for i in temp_list if((i in temp_neighbour_list) == False)]
    
                            
                            temp_neighbour_list.extend(temp_list)
                            
                            
                        else:
                            pass
                        
                    cluster_list.append(temp_cluster_list)
                else:
                    pass
                
            else:
                pass
                
        else:
            pass
        
    
    polygon_list = [pol for cluster in cluster_list for pol in cluster]
    #polygon_list_unique = np.unique(np.array(polygon_list)).tolist()
    
    isolated_polygon_list = []
    for primary_tile_number in neighbor_info_dataframe["index"]:
        if(os.path.exists(f"data_{conflict_type}/{str(dx)}/{str(primary_tile_number)}.parq") == True):
            #if((primary_tile_number in polygon_list_unique) == False):
            if((primary_tile_number in polygon_list) == False):
                isolated_polygon_list.append(primary_tile_number)
            else:
                pass
    
    for isolated_polygon in isolated_polygon_list:
        cluster_list.append([isolated_polygon])
        
    return cluster_list


def undirected_clusters(cluster_list):
    """Input cluster_list from the spreading_pathways function.

    This function returns a list of clusters of nodes as if the links are undirected. These are link based clusters and so there are no links between different clusters."""

    cluster_list.sort(key=len)

    clusters_to_remove = []
    for index in range(len(cluster_list)):
        for checker_index in range(index+1,len(cluster_list)):
            if(set(cluster_list[index]).issubset(set(cluster_list[checker_index])) == True):
                clusters_to_remove.append(index)
                break

    undirected_cluster_list = np.delete(np.array(cluster_list),np.array(clusters_to_remove)).tolist()

    return undirected_cluster_list


def data_bin_extracter(time_series_FG,time):
    """This function extracts data_bin or the list of events arranged in chronological order
    from the fine grained time series of conflict.
    It returns data_bin with bins according to the time bin size we need. The data_bin returned is a CG version of data."""
    
    for day in range(len(time_series_FG)):
        time_series_day = np.zeros((time_series_FG.shape[1],2))
        time_series_day[:,0] = time_series_FG.columns.astype(int).tolist()
        time_series_day[:,1] = time_series_FG.iloc[day]
        
        pols_with_events = numpy_indexed.group_by(time_series_day[:,1]).split_array_as_list(time_series_day)
        
        if(len(pols_with_events) == 2):
            pols_with_events = pols_with_events[1][:,0]
            
            if(day == 0):
                polygon_number = pols_with_events
                days_array = np.array([day for i in range(len(pols_with_events))])
            else:
                polygon_number = np.concatenate((polygon_number,pols_with_events))
                days_array = np.concatenate((days_array,np.array([day for i in range(len(pols_with_events))])))
        else:
            pass
        
    data_bin_array = np.zeros((len(polygon_number),3))
    data_bin_array[:,0] = days_array
    data_bin_array[:,2] = polygon_number
    
    bins = np.digitize(data_bin_array[:,0] , bins=arange(0 , max(data_bin_array[:,0]) + time , time))
    data_bin_array[:,1] = bins
    
    data_bin = pd.DataFrame(data_bin_array , columns=["days","bins","polygon_number"])
    
    return data_bin


def ava_temporal_profile(time,dx,conflict_type,algo_type,spatial_cutoff,temporal_cutoff,plot,boundary):
    """spatial and temporal cutoff are the minimum number of spatial and temporal bins that are required for an "box" cluster to be called an avalanche.
    boundary = y/n. y if you want to remove avalanches that start or end at the edge of dataset"""
    file_type = "conflict"
    
    box_list_path = f"data_{conflict_type}/box_algo_avas/{file_type}/{algo_type}/{algo_type}_avalanche_box_{str(time)}_{str(dx)}.csv"
    avalanches_box = box_str_to_tuple(box_list_path)
    
    temporal_upper_limit = max(list(zip(*avalanches_box[-1]))[1])
    temporal_lower_limit = min(list(zip(*avalanches_box[0]))[1])
    
    x_interp = np.arange(0,1,0.001)
    
    interp_points = np.zeros(len(x_interp))
    
    for ava in avalanches_box:
        spatial_tester = numpy_indexed.group_by(np.array(ava)[:,0]).split_array_as_list(np.array(ava))
        temporal_tester = numpy_indexed.group_by(np.array(ava)[:,1]).split_array_as_list(np.array(ava))
        
        if(len(spatial_tester) <= spatial_cutoff or len(temporal_tester) <= temporal_cutoff):
            pass
        else:
            if(boundary == "y"):
                if(max(list(zip(*ava))[1]) != temporal_upper_limit and min(list(zip(*ava))[1]) != temporal_lower_limit):
                    y_data = np.array([len(time_component) for time_component in temporal_tester])
                    x_data = np.array(range(len(y_data)))
        
                    if(max(y_data-min(y_data)) != 0):
                        y_data = (y_data-min(y_data)) / max(y_data-min(y_data))  #Normalizing y axis
                    else:
                        pass
                    x_data = (x_data-min(x_data)) / max(x_data-min(x_data))  #Normalizing x axis
                    
                    y_interp = np.interp(x_interp,x_data,y_data)
                    
                    interp_points = np.vstack([interp_points,y_interp])
                    
                else:
                    pass
            else:
                y_data = np.array([len(time_component) for time_component in temporal_tester])
                x_data = np.array(range(len(y_data)))
    
                if(max(y_data-min(y_data)) != 0):
                    y_data = (y_data-min(y_data)) / max(y_data-min(y_data))  #Normalizing y axis
                else:
                    pass
                x_data = (x_data-min(x_data)) / max(x_data-min(x_data))  #Normalizing x axis
                
                y_interp = np.interp(x_interp,x_data,y_data)
                
                interp_points = np.vstack([interp_points,y_interp])
                
                
                
    
    interp_points = interp_points[1:,:]
    y_interp_mean = np.mean(interp_points,axis=0)
    
    if(plot == "y"):
        plt.plot(x_interp,y_interp_mean)
        
        plt.xlabel("Time (t/T)" , fontsize=14)
        plt.ylabel("Average # of sites" , fontsize=14)
        
        plt.title(f"{str(time)},{str(dx)},{algo_type}"  ,fontsize=14)

    return interp_points.shape


def conflict_event_polygon_mapping(dx , conflict_type , cpu_cores):
    print("Finding event to polygon mapping!")

    gridix = 0

    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')

    conflict_positions = gpd.read_file(f"data_{conflict_type}/conflict_positions/conflict_positions.shp")


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