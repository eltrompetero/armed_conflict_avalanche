#from pyutils import *
from ctypes import sizeof
from datetime import date, time, timedelta
import geopandas
from shapely import geometry
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

import misc_funcs

from numba import jit,njit

def TE_tiles(*args):
    """args = time , dx , conflict_type , number_of_shuffles , type_of_TE , time_series_all_pol """

    time , dx , conflict_type , number_of_shuffles , type_of_TE , time_series_all_pol = args


    if(type_of_TE == "n"):
        print("Calculating transfer entropy between tiles!")
        important_indexes = [(0,0,0,0),(1,0,1,0),(2,1,2,1),(3,1,3,1),(4,2,0,0),(5,2,1,0),(6,3,2,1),(7,3,3,1)]
        TE_type = "transfer_entropy"
        definition_type = ""

    elif(type_of_TE == "ep"):
        print("Calculating excitatory transfer entropy between tiles(paper's definition)!")
        important_indexes = [(0,0,0,0),(2,1,2,1),(5,2,1,0),(7,3,3,1)]
        TE_type = "exci"
        definition_type = "paper_"

    elif(type_of_TE == "eo"):
        print("Calculating excitatory transfer entropy between tiles(our definition)!")
        important_indexes = [(1,0,1,0),(3,1,3,1),(5,2,1,0),(7,3,3,1)]
        TE_type = "exci"
        definition_type = "our_"

    elif(type_of_TE == "ip"):
        print("Calculating inhibitory transfer entropy between tiles(paper's definition)!")
        important_indexes = [(1,0,1,0),(3,1,3,1),(4,2,0,0),(6,3,2,1)]
        TE_type = "inhi"
        definition_type = "paper_"

    elif(type_of_TE == "io"):
        print("Calculating inhibitory transfer entropy between tiles(our definition)!")
        important_indexes = [(0,0,0,0),(2,1,2,1),(4,2,0,0),(6,3,2,1)]
        TE_type = "inhi"
        definition_type = "our_"


    valid_polygons = time_series_all_pol.columns.to_list()

    gridix = 0
    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')

    def neighbors_to_list(neighbor_list):
        return list(map(int , neighbor_list.replace(" ", "").split(",")))

    neighbor_info_dataframe = polygons.drop("geometry" , axis=1)
    neighbor_info_dataframe["neighbors_temp"] = neighbor_info_dataframe["neighbors"].apply(neighbors_to_list)
    neighbor_info_dataframe.drop("neighbors" , inplace=True , axis=1)
    neighbor_info_dataframe.rename({"neighbors_temp": "neighbors"}, axis=1 , inplace=True)

    tiles_transfer_entropy = pd.DataFrame()


    @njit()
    def distribution_y_new(time_series_two_pols):
        counts_array = np.zeros(2)   #0->0 , 1->1

        time_series_y = time_series_two_pols[:,1]

        #time_series_y = time_series_y[1:]
        time_series_y = time_series_y[:-1]

        time_series_length = len(time_series_y)

        for i in range(len(time_series_y)):
            if(time_series_y[i] == 0):
                counts_array[0] = counts_array[0] + 1
            elif(time_series_y[i] == 1):
                counts_array[1] = counts_array[1] + 1

        distribution_array = np.zeros(2)

        distribution_array[0] = counts_array[0] / time_series_length
        distribution_array[1] = counts_array[1] / time_series_length

        return distribution_array


    @njit()
    def distribution_x_new(time_series_two_pols):
        counts_array = np.zeros(2)   #0->0 , 1->1

        time_series_y = time_series_two_pols[:,0]

        #time_series_y = time_series_y[1:]
        time_series_y = time_series_y[:-1]

        time_series_length = len(time_series_y)

        for i in range(len(time_series_y)):
            if(time_series_y[i] == 0):
                counts_array[0] = counts_array[0] + 1
            elif(time_series_y[i] == 1):
                counts_array[1] = counts_array[1] + 1

        distribution_array = np.zeros(2)

        distribution_array[0] = counts_array[0] / time_series_length
        distribution_array[1] = counts_array[1] / time_series_length

        return distribution_array

    @njit()
    def joint_distribution_yt_xt_new(time_series_two_pols):
        counts_array = np.zeros(4)   #(yt,xt)  0->00 , 1->01 , 2->10 , 3-> 11

        #time_series_yt_xt = time_series_two_pols[1:]
        time_series_yt_xt = time_series_two_pols[:-1]

        time_series_length = len(time_series_yt_xt)


        for i in range(time_series_length):           
            if(time_series_yt_xt[i][1] == 0 and time_series_yt_xt[i][0] == 0):
                counts_array[0] = counts_array[0] + 1
            elif(time_series_yt_xt[i][1] == 0 and time_series_yt_xt[i][0] == 1):
                counts_array[1] = counts_array[1] + 1
            elif(time_series_yt_xt[i][1] == 1 and time_series_yt_xt[i][0] == 0):
                counts_array[2] = counts_array[2] + 1
            elif(time_series_yt_xt[i][1] == 1 and time_series_yt_xt[i][0] == 1):
                counts_array[3] = counts_array[3] + 1

        distribution_array = np.zeros(4)

        for i in range(len(distribution_array)):
            distribution_array[i] = counts_array[i] / time_series_length

        return distribution_array

    @njit()
    def joint_distribution_yT_yt_new(time_series_two_pols):
        counts_array = np.zeros(4)   #(yT,yt)  0->00 , 1->01 , 2->10 , 3-> 11

        time_series_yT_yt = time_series_two_pols
        #time_series_yT_yt = time_series_yT_yt[:-1]

        time_series_length = len(time_series_yT_yt)


        for i in range(time_series_length-1):           
            if(time_series_yT_yt[i+1][1] == 0 and time_series_yT_yt[i][1] == 0):
                counts_array[0] = counts_array[0] + 1
            elif(time_series_yT_yt[i+1][1] == 0 and time_series_yT_yt[i][1] == 1):
                counts_array[1] = counts_array[1] + 1
            elif(time_series_yT_yt[i+1][1] == 1 and time_series_yT_yt[i][1] == 0):
                counts_array[2] = counts_array[2] + 1
            elif(time_series_yT_yt[i+1][1] == 1 and time_series_yT_yt[i][1] == 1):
                counts_array[3] = counts_array[3] + 1

        distribution_array = np.zeros(4)

        for i in range(len(distribution_array)):
            distribution_array[i] = counts_array[i] / (time_series_length-1)

        return distribution_array

    @njit()
    def joint_distribution_yT_yt_xt_new(time_series_two_pols):
        counts_array = np.zeros(8)   #(yT,yt,xt)  0->000,1->001,2->010,3->011,4->100,5->101,6->110,7->111

        time_series_yT_yt_xt = time_series_two_pols
        #time_series_yT_yt_xt = time_series_yT_yt_xt[:-1]

        time_series_length = len(time_series_yT_yt_xt)


        for i in range(time_series_length-1):           
            if(time_series_yT_yt_xt[i+1][1] == 0 and time_series_yT_yt_xt[i][1] == 0 and time_series_yT_yt_xt[i][0] == 0):
                counts_array[0] = counts_array[0] + 1
            elif(time_series_yT_yt_xt[i+1][1] == 0 and time_series_yT_yt_xt[i][1] == 0 and time_series_yT_yt_xt[i][0] == 1):
                counts_array[1] = counts_array[1] + 1
            elif(time_series_yT_yt_xt[i+1][1] == 0 and time_series_yT_yt_xt[i][1] == 1 and time_series_yT_yt_xt[i][0] == 0):
                counts_array[2] = counts_array[2] + 1
            elif(time_series_yT_yt_xt[i+1][1] == 0 and time_series_yT_yt_xt[i][1] == 1 and time_series_yT_yt_xt[i][0] == 1):
                counts_array[3] = counts_array[3] + 1
            elif(time_series_yT_yt_xt[i+1][1] == 1 and time_series_yT_yt_xt[i][1] == 0 and time_series_yT_yt_xt[i][0] == 0):
                counts_array[4] = counts_array[4] + 1
            elif(time_series_yT_yt_xt[i+1][1] == 1 and time_series_yT_yt_xt[i][1] == 0 and time_series_yT_yt_xt[i][0] == 1):
                counts_array[5] = counts_array[5] + 1
            elif(time_series_yT_yt_xt[i+1][1] == 1 and time_series_yT_yt_xt[i][1] == 1 and time_series_yT_yt_xt[i][0] == 0):
                counts_array[6] = counts_array[6] + 1              
            elif(time_series_yT_yt_xt[i+1][1] == 1 and time_series_yT_yt_xt[i][1] == 1 and time_series_yT_yt_xt[i][0] == 1):
                counts_array[7] = counts_array[7] + 1            



        distribution_array = np.zeros(8)

        for i in range(len(distribution_array)):
            distribution_array[i] = counts_array[i] / (time_series_length-1)

        return distribution_array




    def transfer_entropy_calculator(distribution_data):
        yT_yt_xt , yT_yt , yt_xt , y = distribution_data
        return (yT_yt_xt * np.log2((yT_yt_xt * y) / (yT_yt * yt_xt)))


    def transfer_entropy_tiles(tile_data , shuffle_number):
        transfer_entropy_list = []
        primary_tile_number , neighbor_list = tile_data
        if(primary_tile_number in valid_polygons):
            for neighbor_tile_number in neighbor_list:
                if(neighbor_tile_number in valid_polygons): 

                    time_series_two_pols = time_series_all_pol[[primary_tile_number,neighbor_tile_number]]
                    time_series_two_pols.columns = ["x" , "y"]

                    if(shuffle_number != 0):
                        np.random.shuffle(time_series_two_pols["x"].to_numpy())
                        time_series_two_pols = time_series_two_pols.to_numpy()
                    else:
                        time_series_two_pols = time_series_two_pols.to_numpy()          
                    
                    
                    joint_distribution_yT_yt_xt_list = joint_distribution_yT_yt_xt_new(time_series_two_pols)
                    joint_distribution_yT_yt_list = joint_distribution_yT_yt_new(time_series_two_pols)
                    joint_distribution_yt_xt_list = joint_distribution_yt_xt_new(time_series_two_pols)
                    distribution_y_list = distribution_y_new(time_series_two_pols)


                    transfer_entropy_individual_terms = []

                    for i,j,k,l in important_indexes:
                        if(joint_distribution_yT_yt_xt_list[i] == 0 or joint_distribution_yT_yt_list[j] == 0 or joint_distribution_yt_xt_list[k] == 0 or distribution_y_list[l] == 0):
                            pass
                        else:
                            transfer_entropy_individual_terms.append(transfer_entropy_calculator((joint_distribution_yT_yt_xt_list[i],joint_distribution_yT_yt_list[j],joint_distribution_yt_xt_list[k],distribution_y_list[l])))

                    transfer_entropy_value = sum(transfer_entropy_individual_terms)


                    transfer_entropy_list.append(transfer_entropy_value)
                else:
                    transfer_entropy_list.append("NA")
        else:
            transfer_entropy_list.append("NA")            
        return transfer_entropy_list

    for shuffle_number in range(number_of_shuffles+1):
        if(shuffle_number % 10 == 0):
            print("Shuffle Number:", shuffle_number)
        
        tiles_transfer_entropy[f"{TE_type}_{str(shuffle_number)}"] = neighbor_info_dataframe.apply(transfer_entropy_tiles, args=(shuffle_number,) , axis=1)
    #tiles_transfer_entropy.to_pickle(f"data_{str(conflict_type)}/{TE_type}/tiles_{TE_type}_{definition_type}{str(time)}_{str(dx)}_{str(number_of_shuffles)}")

    print("Done!")
    return tiles_transfer_entropy