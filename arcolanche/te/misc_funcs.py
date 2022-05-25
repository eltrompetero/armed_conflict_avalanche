from .utils import *

from . import transfer_entropy_func 
from . import self_loop_entropy_func

from . import data as data_loader

from . import network as net
#import avalanche_numbering


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


def conflict_event_polygon_mapping(dx , gridix , conflict_type , cpu_cores , progress_bar="y"):
    print("Finding event to polygon mapping!")

    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')

    conflict_positions = gpd.read_file(f"generated_data/{conflict_type}/conflict_positions/conflict_positions.shp")

    if(progress_bar == "y"):
        pandarallel.initialize(nb_workers=cpu_cores , progress_bar=True)
    else:
        pandarallel.initialize(nb_workers=cpu_cores , progress_bar=False)

    def location(point):
        ix = np.where(polygons.contains(point))[0]
        return ix[0]


    event_pol_mapping = conflict_positions["geometry"].parallel_apply(location)
    #event_pol_mapping = conflict_positions["geometry"].apply(location)

    event_pol_mapping.to_numpy()

    mapping = np.zeros((len(event_pol_mapping),2))
    mapping[:,0] = list(range(len(event_pol_mapping)))
    mapping[:,1] = event_pol_mapping

    np.savetxt(f"generated_data/{conflict_type}/gridix_{gridix}/event_mappings/event_mapping_{str(dx)}.csv" , mapping , delimiter=',')

    print("Done!")

    return None


def single_tile_events(dx , gridix , conflict_type):

    conflict_data = data_loader.conflict_data_loader(conflict_type)

    event_mappings = np.loadtxt(f"generated_data/{conflict_type}/gridix_{gridix}/event_mappings/event_mapping_{str(dx)}.csv" , delimiter=",")
    event_mappings = event_mappings[:,1]
    event_mappings = pd.DataFrame({'polygon_number' : event_mappings})

    conflict_data.drop(conflict_data.columns.difference(["Unnamed: 0" , "Unnamed: 0.1"]), 1, inplace=True)

    conflict_data["polygon_number"] = event_mappings["polygon_number"]

    groups = conflict_data.groupby("polygon_number")

    #if(os.path.exists(f"generated_data/{conflict_type}/{str(dx)}/") == False):
    #    os.makedirs(f"generated_data/{conflict_type}/{str(dx)}/")

    for i,v in groups.size().items():
        d = groups.get_group(i)
        #d.drop('Unnamed: 0.1', inplace=True, axis=1)
        d.reset_index(inplace=True , drop=True)
        d.rename({'Unnamed: 0': f'event_number_{conflict_type}'}, axis=1, inplace=True)
        d.rename({'Unnamed: 0.1': f'event_number_all'}, axis=1, inplace=True)

        fpar.write(f"generated_data/{conflict_type}/gridix_{gridix}/{str(dx)}/{str(int(i))}.parq" , d)

    return None


def binning(time , dx , gridix , conflict_type):
    print("Creating time bins!")

    time_binning = np.loadtxt(f"generated_data/{conflict_type}/gridix_{gridix}/event_mappings/event_mapping_{str(dx)}.csv" , delimiter=",") #this var is named time_binning because later it will become time_binning. Right now it is event_mappings.
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


###ST Avalanches and required functions===Start###

def avalanche_creation_fast_st(time_series , time , dx  , gridix , conflict_type):
#Here enter the time series(invert the time series if you want to calculate peace avalanche).

    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')

    data_bin = binning(time,dx,gridix,conflict_type)
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
    """Returns an array of arrays which contains the valid neighbors of valid polygons
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


###ST Avalanches and required functions===End###


###TE Avalanches and required functions===Start###
def avalanche_creation_fast_te(time , dx  , gridix , conflict_type , type_of_events):
    #Needs overall restructure!!!! 
    dtdx = (time, dx)

    polygons = gpd.read_file(f'voronoi_grids/{dtdx[1]}/borders{str(gridix).zfill(2)}.shp')
    def neighbors_to_list(neighbor_list):
        return list(map(int , neighbor_list.replace(' ', '').split(',')))
    neighbor_info_df = polygons.drop('geometry' , axis=1)
    neighbor_info_df['neighbors'] = neighbor_info_df['neighbors'].apply(neighbors_to_list)

    if(type_of_events == "null_reassign"):
        time_series , time_series_FG = null_model_time_series_generator(time,640,dx,gridix,conflict_type)
    elif(type_of_events == "data"):
        time_series_FG = pd.read_csv(f'generated_data/{conflict_type}/gridix_{gridix}/FG_time_series/time_series_1_{dtdx[1]}.csv')
        time_series = CG_time_series_fast(time_series_FG.values, dtdx[0])
        time_series = pd.DataFrame(time_series, columns=time_series_FG.columns.astype(int) , index=range(1,len(time_series)+1))


    # Calculate transfer entropies and shuffles for pairs and self
    self_poly_te = net.self_links(time_series)
    pair_poly_te = net.links(time_series, neighbor_info_df)

    G = net.CausalGraph()
    G.setup(self_poly_te,pair_poly_te)

    # To create neighbors_arr that avalanches_te requires i.e
    # a array of array where every array contains successive neighbors of valid polygons
    # such that the total length of neighbors_arr is equal to the length of time_series.columns 
    # Line 2 to 4 adds empty lists for isolated nodes that are not in the causal network.
    #neighbors = G.causal_neighbors()
    #for poly in time_series.columns:
    #    if(poly not in neighbors.keys()):
    #        neighbors[poly] = []
    #neighbors_list = []
    #for i in neighbors:
    #    neighbors_list.append(np.array(neighbors[i]))
    #neighbors_arr = np.array(neighbors_list , dtype=object)

    neighbors_basix = []
    for node in time_series.columns:
        neighbors_basix.append([])
        if(node in G.nodes):
            for n in G.neighbors(node):
                neighbors_basix[-1].append(np.where(time_series.columns == n)[0][0])


    valid_polygons = time_series.columns.to_numpy()


    tiles_with_self_loop_list = G.self_loop_list()
    time_series_arr = time_series.to_numpy()

    avalanche_list = avalanche_te(time_series_arr , neighbors_basix)
    avalanche_list = convert_back_to_regular_form(avalanche_list,valid_polygons)


    return avalanche_list , time_series_arr , neighbors_basix


def null_model_time_series_generator(time,dx_primary,dx_interest,gridix,conflict_type,cpu_cores=1):
    #cpu_cores = int(input("Enter # of cpu_cores to use: "))
    x,time_series_FG_interest = null_model(dx_primary,dx_interest,gridix,conflict_type,"reassign",cpu_cores)
    time_series_FG = time_series_FG_interest
    
    time_series_FG_arr = time_series_FG.to_numpy()
    col_nums = len(time_series_FG.columns)
    
    time_series_CG = CG_time_series_fast(time_series_FG_arr,time)
    
    time_series_CG = time_series_CG.astype(int)
    time_series_CG = pd.DataFrame(time_series_CG , index=range(1,len(time_series_CG)+1))
    time_series_CG.columns = time_series_FG.columns
    
    return time_series_CG , time_series_FG


def null_model(dx_primary,dx_interest,gridix,conflict_type,prob_type,cpu_cores):
    """Returns simulation time series of primary dx and the dx you are interested in(in dataframe form).
    prob_type = "reassign" for reassigned probabilities.
    
    dx_primary > dx_interest."""
     
    
    time_series_FG = pd.read_csv(f"generated_data/{conflict_type}/gridix_{str(gridix)}/FG_time_series/time_series_1_{str(dx_primary)}.csv")
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


@njit
def CG_time_series_fast(time_series_FG, time):
    """time_series_FG in array form, col_nums = len(time_series_FG.columns)
    
    Run these lines to convert back to pandas dataframe->
    time_series_CG = time_series_CG.astype(int)
    time_series_CG = pd.DataFrame(time_series_CG , index=range(1,len(time_series_CG)+1))
    time_series_CG.columns = time_series_FG.columns
    """

    duration = time_series_FG.shape[0]
    row_change_arr = np.arange(0 , duration + time , time)
    time_series_CG = np.zeros((len(row_change_arr)-1, time_series_FG.shape[1])).astype(np.int64)
    
    current_row = 0
    CG_row = 0
    while(current_row < duration):
        #print(current_row)
        valid_col_arr = np.asarray(np.arange(time_series_FG.shape[1]))
        
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

def te_causal_network(time_series, neighbor_info_dataframe,
                      number_of_shuffles=50):
    """Calculates transfer entropy and identifies significant links between Voronoi
    neighbors assuming a 95% confidence interval.

    Parameters
    ----------
    time_series : pd.DataFrame
    neighbor_info_dataframe : pd.DataFrame
    number_of_shuffles : int, 50
    
    Returns
    -------
    pd.DataFrame
    pd.DataFrame
    list of tuples
        (cell index, cell index, TE)
        TE is nan when non-significant
    """

    # calculate transfer entropy between pairs of tiles
    def polygon_pair_gen():
        """Pairs of legitimate neighboring polygons."""
        for i, row in neighbor_info_dataframe.iterrows():
            for n in row['neighbors']:
                # only consider pairs of polygons that appear in the time series
                if row['index'] in time_series.columns and n in time_series.columns:
                    yield (row['index'], n)
    
    pair_poly_te = transfer_entropy_func.iter_polygon_pair(polygon_pair_gen(),
                                                           number_of_shuffles, 
                                                           time_series)
    # process output into convenient packaging
    clean_pair_poly_te = []
    filtered_neighbors = {}
    for key, val in pair_poly_te.items():
        # check if polygon already in dict
        if not key[0] in filtered_neighbors.keys():
            filtered_neighbors[key[0]] = []

        # add sig neighbors
        if (val[0]>val[1]).mean()>.95:
            filtered_neighbors[key[0]].append(key[1])
            clean_pair_poly_te.append((key[0], key[1], val[0]))
        else:
            clean_pair_poly_te.append((key[0], key[1], np.nan))

    return pair_poly_te, filtered_neighbors, clean_pair_poly_te

def significant_links(TE_array , shuffled_TE_arr):
    return TE_array > np.percentile(shuffled_TE_arr , 95 , axis=0)


def self_loop_finder_TE(time_series , time , dx  , gridix , conflict_type):
    TE_type = "self_loop_entropy"
    
    number_of_shuffles = 50
    
    #tiles_transfer_entropy = pd.read_pickle(f"data_{str(conflict_type)}/{TE_type}/tiles_{TE_type}_{definition_type}{str(time)}_{str(dx)}_{str(number_of_shuffles)}")
    args = (time , dx , gridix , conflict_type , number_of_shuffles , time_series)
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


def avalanche_te(time_series_arr , neighbors):
    """This function geenrates avalanches using time series and information about
    neighbors of each polygon. The neighbor array contains self loop details and
    also information about isloated nodes. Isolated nodes have empty neighbor arrays.
    
    Input preparation code-
    time_series_arr = time_series.to_numpy()
    valid_polygons = time_series.columns.to_numpy()
    neighbors = misc_funcs.polygon_neigbors_arr(polygons , valid_polygons)
    """
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
                    
                    
                    #if((time_series_arr[box[1]+1,box[0]] == 1) and (box[0] in tiles_with_self_loop)):
                    #    avalanche_temp.append((box[0],box[1]+1))
                    #    secondary_boxes.append((box[0],box[1]+1))
                    #    time_series_arr[box[1]+1,box[0]] = 0
                    
                
                
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
                        
                        
                        #if((time_series_arr[secondary_box[1]+1,secondary_box[0]] == 1) and (secondary_box[0] in tiles_with_self_loop)):
                        #    avalanche_temp.append((secondary_box[0],secondary_box[1]+1))
                        #    secondary_boxes.append((secondary_box[0],secondary_box[1]+1))
                        #    time_series_arr[secondary_box[1]+1,secondary_box[0]] = 0
                            
            
                avalanche_list.append(avalanche_temp)
                
    return avalanche_list
###TE Avalanches and required functions===End###




###Misc###
def ava_numbering(time , dx , gridix , conflict_type , avaEvent_list_path):
    print("Creating final data table!")

    data = binning(time,dx,gridix,conflict_type)
    data["avalanche_number"] = 0

    def avalanche_numbering(ava_num , i):
        data["avalanche_number"].iloc[ava_num] = i
        return None    

    i = 0
    with open(avaEvent_list_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            row = list(map(int , row))
            ava_temp = pd.DataFrame(row)

            ava_temp.apply(avalanche_numbering , args=(i,))

            i += 1

    #data.to_csv(f"data_{conflict_type}/st_avalanche/st_avalanche_{str(time)}_{str(dx)}.csv")

    ACLED_data = data_loader.conflict_data_loader(conflict_type)
    data["fatalities"] = ACLED_data["fatalities"]

    print("Done!")

    return data



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



def sites_for_box_avas(avalanches):
    """Input avalanches in box form"""
    sites = []
    for ava in avalanches:
        sites.append(len(unique(list(zip(*ava))[0])))
    return sites , "Sites"    #returns dt and xlabel

def duration_for_box_avas(avalanches):
    """Input avalanches in box form.
    
    This calculates duration in bin units or the coarse grained duration (temporal analogous to sites)"""
    duration = []
    for ava in avalanches:
        duration.append(len(unique(list(zip(*ava))[1])))
    return duration , "Duration"


def time_series_generator(time, dx, gridix, conflict_type):
    polygons = gpd.read_file(f'voronoi_grids/{str(dx)}/borders{str(gridix).zfill(2)}.shp')

    def neighbors_to_list(neighbor_list):
        return list(map(int , neighbor_list.replace(' ', '').split(',')))
    neighbor_info_df = polygons.drop('geometry' , axis=1)
    neighbor_info_df['neighbors'] = neighbor_info_df['neighbors'].apply(neighbors_to_list)

    time_series_FG = pd.read_csv(f'generated_data/{conflict_type}/gridix_{gridix}/FG_time_series/time_series_1_{str(dx)}.csv')
    time_series = CG_time_series_fast(time_series_FG.values, time)
    time_series = pd.DataFrame(time_series, columns=time_series_FG.columns.astype(int) , index=range(1,len(time_series)+1))

    return time_series , neighbor_info_df

def FG_time_series(time,dx,gridix,conflict_type,randomize_polygons=False):
    """Generates fine grained time series of conflicts.
    
    Parameters
    ----------
    time : int
    dx : int
    gridix : int
    conflict_type : str
    randomize_polygons : bool , False
    
    Returns
    -------
    numpy array
        Fine grained time series.
    numpy array
        An array containing the polygon numbers in ascending order
        which can be used as column names for dataframe of fine grained
        time series.
    """
    data_frame = binning(time,dx,gridix,conflict_type)
    
    if(randomize_polygons == False):
        data_array = np.array(data_frame[["polygon_number","days"]] , dtype=int)
    elif(randomize_polygons == True):
        data_array = np.array(data_frame[["polygon_number","days"]] , dtype=int)
        data_array[:,0] = np.random.permutation(data_array[:,0])   #Randomly changing the polygon where a conflict event occurs
        
    polygon_groups = numpy_indexed.group_by(data_array[:,0]).split_array_as_list(data_array)
    
    col_index = np.unique(data_array[:,0])
    time_series_FG = np.zeros((max(data_array[:,1])+1,len(col_index)))
    
    for event_day_index,index in zip(polygon_groups,range(len(col_index))):
        time_series_FG[event_day_index[:,1],index] = 1
        
    return time_series_FG , col_index