from .utils import *

from .. import transfer_entropy_func
from .. import self_loop_entropy_func

from .. import network as net
#import avalanche_numbering

from arcolanche.data import *
from voronoi_globe import load_voronoi

from shapely import geometry
import geopandas

###Preparing the data====Start###


def conflict_position(conflict_type):
    '''
    This script is used to create and then save a geodataframe of all the geographic points where events took place using lattitude and longitude of
    events from ACLED dataset and the dates on which those events occured
    '''
    data = ACLED2020.battles_df()

    temp_list = []
    for i in range(len(data)):
        temp_point = geometry.Point(data["LONGITUDE"].iloc[i] , data["LATITUDE"].iloc[i])
        temp_list.append(temp_point)

    conflict_event_positions = geopandas.GeoDataFrame(temp_list)
    conflict_event_positions["date"] = data["EVENT_DATE"].astype(str)

    conflict_event_positions.set_geometry(0 , inplace=True)
    conflict_event_positions.rename_geometry('geometry' , inplace=True)

    conflict_event_positions.to_file(f"generated_data/{conflict_type}/conflict_positions/conflict_positions.shp")

    return None


def conflict_event_polygon_mapping(dx , gridix , conflict_type , cpu_cores , progress_bar="y"):
    print("Finding event to polygon mapping!")

    polygons = load_voronoi(dx,gridix)
    #polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')

    conflict_positions = gpd.read_file(f"generated_data/{conflict_type}/conflict_positions/conflict_positions.shp")

    if(progress_bar == "y"):
        pandarallel.initialize(nb_workers=cpu_cores , progress_bar=True)
    else:
        pandarallel.initialize(nb_workers=cpu_cores , progress_bar=False)

    def location(point):
        ix = np.where(polygons.contains(point))[0]
        return polygons.index[ix[0]]


    event_pol_mapping = conflict_positions["geometry"].parallel_apply(location)
    #event_pol_mapping = conflict_positions["geometry"].apply(location)

    event_pol_mapping.to_numpy()

    mapping = np.zeros((len(event_pol_mapping),2))
    mapping[:,0] = list(range(len(event_pol_mapping)))
    mapping[:,1] = event_pol_mapping
    mapping = mapping.astype(int)

    np.savetxt(f"generated_data/{conflict_type}/gridix_{gridix}/event_mappings/event_mapping_{str(dx)}.csv" , mapping , delimiter=',')

    print("Done!")

    return None


def single_tile_events(dx , gridix , conflict_type):

    conflict_data = ACLED2020.battles_df()

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
    #print("Creating time bins!")

    time_binning = np.loadtxt(f"generated_data/{conflict_type}/gridix_{gridix}/event_mappings/event_mapping_{str(dx)}.csv" , delimiter=",") #this var is named time_binning because later it will become time_binning. Right now it is event_mappings.
    time_binning = time_binning.astype(int)
    time_binning = time_binning[:,1]

    time_binning = pd.DataFrame({'polygon_number' : time_binning})

    if(conflict_type == "battles"):
        data = ACLED2020.battles_df()
    elif(conflict_type == "VAC"):
        data = ACLED2020.vac_df()
    elif(conflict_type == "RP"):
        data = ACLED2020.riots_and_protests_df()

    time_binning["date"] = data["EVENT_DATE"].reset_index()["EVENT_DATE"]

    day = pd.to_datetime(data["EVENT_DATE"].reset_index()["EVENT_DATE"] , dayfirst=True)

    time_binning["days"] = (day-day.min()).apply(lambda x : x.days)

    bins = np.digitize(time_binning["days"] , bins=np.arange(0 , max(time_binning["days"]) + time , time))

    time_binning["bins"] = bins

    time_binning["event_number"] = data.index
    time_binning.set_index(data.index , inplace=True)

    #time_binning.to_csv(f"data_{conflict_type}/time_bins_{str(time)}_{str(dx)}.csv")

    #print("Done!")

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
def avalanche_creation_fast_te(time , dx  , gridix , conflict_type , type_of_events , number_of_shuffles):

    dtdx = (time, dx)

    polygons = load_voronoi(dx,gridix)

    #polygons = gpd.read_file(f'voronoi_grids/{dtdx[1]}/borders{str(gridix).zfill(2)}.shp')
    #def neighbors_to_list(neighbor_list):
    #    return list(map(int , neighbor_list.replace(' ', '').split(',')))

    neighbor_info_df = polygons.drop('geometry' , axis=1)

    #neighbor_info_df['neighbors'] = neighbor_info_df['neighbors'].apply(neighbors_to_list)

    if(type_of_events == "null_reassign"):
        time_series , time_series_FG = null_model_time_series_generator(time,640,dx,gridix,conflict_type)
        data_bin_array = None
    elif(type_of_events == "data"):
        time_series_FG = pd.read_csv(f'generated_data/{conflict_type}/gridix_{gridix}/FG_time_series/time_series_1_{dtdx[1]}.csv')
        time_series = CG_time_series_fast(time_series_FG.values, dtdx[0])
        time_series = pd.DataFrame(time_series, columns=time_series_FG.columns.astype(int) , index=range(1,len(time_series)+1))
        #data_bin_array = None

        data_bin = binning(time,dx,gridix,conflict_type)
        data_bin = data_bin[["polygon_number","days","bins","event_number"]]
        data_bin_array = np.array(data_bin)


    elif(type_of_events == "randomize_polygons"):
        time_series_FG,col_label,data_bin_array = FG_time_series(dx,gridix,conflict_type,randomize_polygons=True)
        time_series = CG_time_series_fast(time_series_FG,time)
        time_series = pd.DataFrame(time_series, columns=col_label , index=range(1,len(time_series)+1))

    elif(type_of_events == "shuffle_ts"):
        time_series_events,col_label = CG_time_series_events(time,dx,gridix,conflict_type)

        for col_index in range(time_series_events.shape[1]):
            time_series_events[:,col_index] = np.random.permutation(time_series_events[:,col_index])

        data_bin_array = CG_event_ts_to_data_bin(time_series_events,col_label)
        data_bin = pd.DataFrame(data_bin_array)
        data_bin["event_number"] = data_bin.index
        data_bin_array = np.array(data_bin)

        time_series = CG_events_to_CG_binary(time_series_events)
        time_series = pd.DataFrame(time_series, columns=col_label , index=range(1,len(time_series)+1))


    # Calculate transfer entropies and shuffles for pairs and self
    self_poly_te = net.self_links(time_series, number_of_shuffles)
    pair_poly_te = net.links(time_series, neighbor_info_df, number_of_shuffles)

    G = net.CausalGraph()
    G.setup(self_poly_te,pair_poly_te,sig_threshold=95)

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

    return avalanche_list , time_series_arr , neighbors_basix , data_bin_array


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
    also information about isolated nodes. Isolated nodes have empty neighbor arrays.

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
def ava_numbering(time,dx,gridix,conflict_type,ava_events):
    """Creates an avalanche data table which contains polygon_number, bins, days, avalanche_number and fatalities corresponding
    to each conflict event.
    Parameters
    ----------
    time : int
    dx : int
    gridix : int
    conflict_type : str
    ava_events : list of lists
    Returns
    -------
    pd.Dataframe
        Dataframe contains the following information corresponding to each conflict event:-
        ['polygon_number', 'date', 'days', 'bins', 'avalanche_number','fatalities']
    """

    avalanche_data = binning(time,dx,gridix,conflict_type)
    avalanche_data["avalanche_number"] = 0
    avalanche_number_dict = dict.fromkeys(avalanche_data.index , 0)

    for ava,index in zip(ava_events,range(len(ava_events))):
        for event in ava:
            avalanche_number_dict[event] = index

    avalanche_data["avalanche_number"] = avalanche_number_dict.values()

    if(conflict_type == "battles"):
        ACLED_data = ACLED2020.battles_df()
    elif(conflict_type == "VAC"):
        ACLED_data = ACLED2020.vac_df()
    elif(conflict_type == "RP"):
        ACLED_data = ACLED2020.riots_and_protests_df()

    avalanche_data["fatalities"] = ACLED_data["FATALITIES"]

    return avalanche_data



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


def event_str_to_tuple(event_list_path):
    """Extracts data from ava list file in event form and then outputs a list of lists which contains events with datatype int"""
    event_list = []
    with open(event_list_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            a = list(map(int , row))

            event_list.append(a)

    return event_list


def sites_for_box_avas(avalanches):
    """Input avalanches in box form"""
    sites = []
    for ava in avalanches:
        sites.append(len(np.unique(list(zip(*ava))[0])))
    return sites , "Sites"    #returns dt and xlabel

def duration_for_box_avas(avalanches):
    """Input avalanches in box form.

    This calculates duration in bin units or the coarse grained duration (temporal analogous to sites)"""
    duration = []
    for ava in avalanches:
        duration.append(len(np.unique(list(zip(*ava))[1])))
    return duration , "Duration"


def time_series_generator(time, dx, gridix, conflict_type):

    polygons = load_voronoi(dx,gridix)
    #polygons = gpd.read_file(f'voronoi_grids/{str(dx)}/borders{str(gridix).zfill(2)}.shp')

    neighbor_info_df = polygons.drop('geometry' , axis=1)

    time_series_FG = pd.read_csv(f'generated_data/{conflict_type}/gridix_{gridix}/FG_time_series/time_series_1_{str(dx)}.csv')
    time_series = CG_time_series_fast(time_series_FG.values, time)
    time_series = pd.DataFrame(time_series, columns=time_series_FG.columns.astype(int) , index=range(1,len(time_series)+1))

    return time_series , neighbor_info_df

def FG_time_series(dx,gridix,conflict_type,randomize_polygons=False):
    """Generates fine grained time series of conflicts.

    Parameters
    ----------
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
    numpy array
        Array containing two columns where column one has polygon numbers
        and column two has the day index of conflict event. The length
        of this array is equal to the total number of events in the
        dataset.
    """

    data_frame = binning(1,dx,gridix,conflict_type)

    if(randomize_polygons == False):
        data_array = np.array(data_frame[["polygon_number","days","bins","event_number"]] , dtype=int)
    elif(randomize_polygons == True):
        data_array = np.array(data_frame[["polygon_number","days","bins","event_number"]] , dtype=int)
        data_array[:,0] = np.random.permutation(data_array[:,0])   #Randomly changing the polygon where a conflict event occurs

    polygon_groups = numpy_indexed.group_by(data_array[:,0]).split_array_as_list(data_array)

    col_label = np.unique(data_array[:,0])
    time_series_FG = np.zeros((max(data_array[:,1])+1,len(col_label)))

    for event_day_index,index in zip(polygon_groups,range(len(col_label))):
        time_series_FG[event_day_index[:,1],index] = 1

    return time_series_FG.astype(int) , col_label , data_array


def boxAva_to_eventAva(time , dx , gridix , conflict_type , algo_type , box_ava=None , data_array=None):
    """Converts avalanches from box form to event form.
    Parameters
    ----------
    time : int
    dx : int
    gridix : int
    conflict_type : str
    algo_type : str
    box_ava : list of lists , None
        List of avalanches in box form inside a list.
    data_array : numpy array , None
        Array containing polygon_number,days and bins corresponding to all events.
    Returns
    -------
    list of lists
        List of avalanches in event form inside a list.
    """

    if box_ava is None:
        box_path = f"avalanches/{conflict_type}/gridix_{gridix}/{algo_type}/{algo_type}_ava_box_{str(time)}_{str(dx)}.csv"
        box_ava = box_str_to_tuple(box_path)

    if data_array is None:
        data_bin = binning(time,dx,gridix,conflict_type)
        data_bin = data_bin[["polygon_number","days","bins"]]
        data_bin["event_number"] = data_bin.index
        data_bin = np.array(data_bin)
    else:
        data_bin = pd.DataFrame(data_array,columns=["polygon_number","days","bins"])
        data_bin["event_number"] = data_bin.index
        data_bin = np.array(data_bin)


    pol_groups = numpy_indexed.group_by(data_bin[:,0]).split_array_as_list(data_bin)
    pol_labels = np.sort(np.unique(data_bin[:,0]))

    pol_bin_groups = []
    for pol_group in pol_groups:
        pol_bin_groups.extend(numpy_indexed.group_by(pol_group[:,2]).split_array_as_list(pol_group))
    pol_bin_groups = np.concatenate(tuple(pol_bin_groups))


    ava_event = []
    for ava in box_ava:
        ava_event_temp = []
        for box in ava:
            ava_event_temp.extend(pol_bin_groups[:,3][np.where((pol_bin_groups[:,0] == box[0]) & (pol_bin_groups[:,2] == box[1]))[0]].astype(int).tolist())

        ava_event.append(ava_event_temp)

    return ava_event


def CG_time_series_events(time,dx,gridix,conflict_type):
    """Generates a CG time series where instead of 1's we have a list of
    events corresponding to that box.

    Parameter
    ---------
    time : int
    dx : int
    gridix : int
    conflict_type : str

    Returns
    -------
    ndarray
        CG time series with events.
    ndarray
        Array containing column numbers which correponds to
        polygon numbers.
    """

    data_bin = binning(time,dx,gridix,conflict_type)

    data_bin_arr = data_bin[["event_number","polygon_number","bins"]].values.astype(int)

    event_groups = numpy_indexed.group_by(data_bin_arr[:,[1,2]]).split_array_as_list(data_bin_arr)

    time_series_FG,col_label,data_bin_array = FG_time_series(dx,gridix,conflict_type)
    time_series_arr = CG_time_series_fast(time_series_FG,time)

    time_series_events = np.zeros(time_series_arr.shape , dtype=object)

    for group in event_groups:
        col_num = group[0,1]
        time_bin_num = group[0,2]-1

        col_index = np.where(col_label==col_num)[0][0]

        time_series_events[time_bin_num,col_index] = group[:,0]

    return time_series_events , col_label


def CG_events_to_CG_binary(time_series_events):
    """Convert CG time series containing event information to
    CG time series in standard binary form.

    Parameters
    ----------
    time_series_events : ndarray
        CG time series with events.

    Returns
    -------
    ndarray
        Standard binary CG time series.
    """

    time_series = np.zeros(time_series_events.shape , dtype=int)

    for event_group,box in zip(np.nditer(time_series_events , flags=["refs_ok"]),np.nditer(time_series , op_flags=["readwrite"])):
        if(event_group != 0):
            box[...] = 1

    return time_series


def CG_event_ts_to_data_bin(time_series_events,col_label):
    """Extract or generate data_bin_array from CG time series containing
    event info.
    Parameters
    ----------
    time_series_events : ndarray
        CG time series with events.
    col_label : ndarray
        Array containing column numbers which correponds to
        polygon numbers.
    Returns
    -------
    ndarray
        data_bin_array with 3 columns: polygon_number,days=0,bins
    """

    number_of_events = 0
    for i in np.nditer(time_series_events , flags=["refs_ok"] , order="F"):
        if(i != 0):
            max_temp = np.amax(i.tolist())
            if(max_temp > number_of_events):
                number_of_events = max_temp

    data_bin_array = np.zeros((number_of_events+1,3) , dtype=int)

    overall_counter = 1
    time_bin_counter = 1
    col_switch_indicator = time_series_events.shape[0]
    col_index = 0
    for event_group in np.nditer(time_series_events , flags=["refs_ok"] , order="F"):
        if(event_group != 0):
            data_bin_array[event_group.tolist(),0] = col_label[col_index]
            data_bin_array[event_group.tolist(),2] = time_bin_counter

        if(overall_counter != 0):
            if(overall_counter % col_switch_indicator == 0):
                col_index += 1
                time_bin_counter = 0

        overall_counter += 1
        time_bin_counter += 1

    return data_bin_array


def conflict_zone_generator(time,dx,gridix,conflict_type,type_of_algo,threshold):
    """Generates conflict zones across Africa using aggregation of conflict
    avalanches.
    Parameters
    ----------
    time: int
    dx : int
    gridix : int
    conflict_type : str
    type_of_algo : str
    threshold : int/float
        Determines the lower bound for the avalanche size above which avalanches
        are considered during aggregation step.
    Returns
    -------
    list of arrays
        the arrays contain polygon number/id of polygons in each conflict
        zone
    """

    box_path = f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/{type_of_algo}_ava_box_{str(time)}_{str(dx)}.csv"
    ava_box = box_str_to_tuple(box_path)

    size_arr = np.array([len(unique(list(zip(*i))[0])) for i in ava_box])

    if(type(threshold) == int):
        threshold_size = threshold
    elif(type(threshold) == float):
        threshold_size = max(size_arr) * threshold

    ava_box_threshold = np.array(ava_box , dtype=object)[np.where(size_arr > threshold_size)[0]].tolist()

    zones = []
    for ava in ava_box_threshold:
        valid = True
        indexes_to_delete = []
        ava_unique_arr = np.unique(np.array(list(zip(*ava))[0]))
        if(len(zones) == 0):
            zones.append(ava_unique_arr)
        else:
            for index,zone in enumerate(zones):
                if(len(set(zone).intersection(set(ava_unique_arr))) != 0):
                    valid = False
                    zones[index] = np.unique(np.concatenate((zone,ava_unique_arr)))
                    ava_unique_arr = zones[index]
                    indexes_to_delete.append(index)

                elif((index == (len(zones)-1)) & (valid == True)):
                    zones.append(ava_unique_arr)

            index_correction = 0
            for index in indexes_to_delete[:-1]:
                del zones[index-index_correction]
                index_correction += 1

    return zones


def actor_dict_generator(acled_data):
    """Generates a dictionary containing all unique actors(actor1 and actor2 combined)
        and their corresponding keys.

    Parameters
    ----------
    acled_data : pd.Dataframe
        Dataframe of the data downloaded from ACLED, filtered according to event_type.

    Returns
    -------
    dict
        Dictionary containing all unique actors(actor1 and actor2 combined)
        and their corresponding keys.
    """

    #acled_data = data_loader.conflict_data_loader(conflict_type)
    acled_data_actors = acled_data[["ACTOR1","ACTOR2"]]

    actor1_arr = (acled_data_actors["ACTOR1"]).to_numpy()
    actor2_arr = (acled_data_actors["ACTOR2"]).to_numpy()

    actors_arr = np.concatenate((actor1_arr,actor2_arr))
    actors_arr = np.unique(actors_arr)

    actors_dict = {}
    for index,actor in zip(range(len(actors_arr)),actors_arr):
        actors_dict[index] = (actor)

    return actors_dict


def event_actor_counter(event_nums , actors_dict , acled_data):
    """Finds the actor composition in the list of entered event numbers.
    Here actor1 and actor2 are treated the same.

    Parameters
    ----------
    event_nums : list
        list of event numbers
    actors_dict : dict
        Dictionary containing all unique actors(actor1 and actor2 combined)
        and their corresponding keys.
    acled_data : pd.Dataframe
        Dataframe of the data downloaded from ACLED, filtered according to event_type.

    Returns
    -------
    list of tuples
        First entry of tuple corresponds to the key of actor in the
        actor_dict. Second entry of tuple correponds to the total number of
        occurances of this actor in the entered event list.
    """

    def actor_dict_lookup(actor):
        for key,value in actors_dict.items():
            if(actor == value):
                key_to_return = key
                break

        return key_to_return

    #acled_data = data_loader.conflict_data_loader(conflict_type)
    acled_data_actors = acled_data[["ACTOR1","ACTOR2"]]

    actors_event = acled_data_actors.loc[event_nums].to_numpy()
    actors_event = actors_event.reshape(actors_event.shape[0]*actors_event.shape[1])

    actor_count = []
    for actor_group in numpy_indexed.group_by(actors_event).split_array_as_list(actors_event):
        actor_count.append((actor_dict_lookup(actor_group[0]),len(actor_group)))

    return actor_count


def events_in_zone(time,dx,gridix,conflict_type,type_of_algo,zone):
    """Find all the vents that are in a particular zone.

    Parameters
    ----------
    time : int
    dx : int
    gridix : int
    conflict_type : str
    type_of_algo : str
    zone : list
        List containing the polygon indexes of polygons that are in the selected zone.

    Returns
    -------
    list
        A list of all the events that are in a particular zone.
    """

    box_path = f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/{type_of_algo}_ava_box_{str(time)}_{str(dx)}.csv"
    ava_box = box_str_to_tuple(box_path)

    zone_set = set(zone)
    in_ava_indexes = []
    for index,ava in enumerate(ava_box):
        ava_pol_set = set(list(zip(*ava))[0])
        if(len(ava_pol_set.intersection(zone_set)) != 0):
            in_ava_indexes.append(index)

    event_path = f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/{type_of_algo}_ava_event_{str(time)}_{str(dx)}.csv"
    ava_event = event_str_to_tuple(event_path)
    ava_event = np.array(ava_event , dtype=object)

    in_zone_events = ava_event[in_ava_indexes]
    in_zone_events = [x for l in in_zone_events for x in l]

    return in_zone_events


def zone_actor_counter(time,dx,gridix,conflict_type,type_of_algo,zone,acled_data):
    """Find the actor composition in a given zone for a particular scale and gridix.
    Parameters
    ----------
    time : int
    dx : int
    gridix : int
    conflict_type : str
    type_of_algo : str
    zone : list
        List containing the polygon indexes of polygons that are in the selected zone.
    acled_data : pd.Dataframe
        Dataframe of the data downloaded from ACLED, filtered according to event_type.
    Returns
    -------
    list of tuples
        First entry of tuple corresponds to the key of actor in the
        actor_dict. Second entry of tuple correponds to the total number of
        occurances of this actor in the avalanches present in entered zone.
    """

    actor_dict = actor_dict_generator(acled_data)
    in_zone_events = events_in_zone(time,dx,gridix,conflict_type,type_of_algo,zone)
    actor_count = event_actor_counter(in_zone_events,actor_dict,acled_data)

    return actor_count


def common_actors_coeff_calculator(time,dx,gridix,conflict_type,type_of_algo,threshold,weighted=False):
    """Calculates the summation of ratio of common actors and sum of number of actors in
    each pair of conflict zones.

    Parameters
    ----------
    time : int
    dx : int
    gridix : int
    conflict_type : str
    type_of_algo : str
    threshold : int/float
        Determines the lower bound for the avalanche size above which avalanches
        are considered during aggregation step.
    weighted : bool , False

    Returns
    -------
    float
    """

    acled_data = conflict_data_loader("battles")
    #acled_data = data_loader.conflict_data_loader(conflict_type)

    zones = conflict_zone_generator(time,dx,gridix,conflict_type,type_of_algo,threshold)

    sorted_zones = sorted(zones , key=len)
    sorted_zones.reverse()

    actor_sets = []
    actor_dicts_list = []
    for index,zone in enumerate(sorted_zones):
        actor_count = zone_actor_counter(time,dx,gridix,conflict_type,type_of_algo,zone,acled_data)

        actor_sets.append(set(list(zip(*actor_count))[0]))
        actor_dicts_list.append(dict(zip(list(zip(*actor_count))[0],list(zip(*actor_count))[1])))

    if(weighted == False):
        common_actors_coeff = 0
        count = 0
        for index in range(len(actor_sets)):
            for jndex in range(index,len(actor_sets)):
                if(index == jndex):
                    common_actors_term = 1
                    common_actors_coeff += common_actors_term
                    count += 1
                else:
                    common_actors_term = (2*len(actor_sets[index].intersection(actor_sets[jndex]))) / (len(actor_sets[index]) + len(actor_sets[jndex]))
                    common_actors_coeff += common_actors_term * 2
                    count += 2
    else:
        common_actors_coeff = 0
        count = 0
        for index in range(len(actor_sets)):
            primary_zones = actor_dicts_list[index]
            primary_zone_actors = primary_zones.keys()
            primary_zone_counts = sum(list(primary_zones.values()))

            for jndex in range(index,len(actor_sets)):
                if(index == jndex):
                    common_actors_term = 1
                    common_actors_coeff += common_actors_term
                    count += 1
                else:
                    secondary_zones = actor_dicts_list[jndex]
                    secondary_zone_actors = secondary_zones.keys()
                    secondary_zone_counts = sum(list(secondary_zones.values()))

                    weights_term = 0
                    for actor in primary_zone_actors:
                        if(actor in secondary_zone_actors):
                            weights_term += (primary_zones[actor]/primary_zone_counts) * \
                                                 (secondary_zones[actor]/secondary_zone_counts)

                    common_actors_term = (2 * weights_term) / (len(primary_zone_actors)+len(secondary_zone_actors))
                    common_actors_coeff += common_actors_term
                    count += 2



    if(count == 0):
        common_actors_coeff = 0
    else:
        common_actors_coeff = common_actors_coeff/count

    return common_actors_coeff


def discrete_power_law_plot(dt , xlabel):
    #For discrete quantities

    dt1 = bincount(dt)      #For getting frequency distribution
    dt1 = dt1/dt1.sum()             #For Normalization
    dt1[dt1 == 0] = np.nan
    dt1 = pd.DataFrame(dt1)
    dt1 = dt1.cumsum(skipna=True)           #To get cumulative distribution
    dt1 = (1-dt1)                    #To get complimentary cumulative distribution
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



def common_actors_coeff_calculator_events(time,dx,gridix,conflict_type,type_of_algo,weighted=False):
    """Calculates the summation of ratio of common actors and sum of number of actors in
    each pair of conflict avalacnhes.

    Parameters
    ----------
    time : int
    dx : int
    gridix : int
    conflict_type : str
    type_of_algo : str
    weighted : bool , False

    Returns
    -------
    float
    """

    acled_data = data_loader.conflict_data_loader(conflict_type)
    actor_dict = actor_dict_generator(acled_data)

    event_path = f"avalanches/battles/gridix_{gridix}/{type_of_algo}/{type_of_algo}_ava_event_{str(time)}_{str(dx)}.csv"
    ava_event = event_str_to_tuple(event_path)

    actor_sets = []
    actor_dicts_list = []
    for index,ava in enumerate(ava_event):
        actor_count = event_actor_counter(ava,actor_dict,acled_data)

        actor_sets.append(set(list(zip(*actor_count))[0]))
        actor_dicts_list.append(dict(zip(list(zip(*actor_count))[0],list(zip(*actor_count))[1])))

    if(weighted == False):
        common_actors_coeff = 0
        count = 0
        for index in range(len(actor_sets)):
            for jndex in range(index,len(actor_sets)):
                if(index == jndex):
                    common_actors_term = 1
                    common_actors_coeff += common_actors_term
                    count += 1
                else:
                    common_actors_term = (2*len(actor_sets[index].intersection(actor_sets[jndex]))) / (len(actor_sets[index]) + len(actor_sets[jndex]))
                    common_actors_coeff += common_actors_term * 2
                    count += 2
    else:
        common_actors_coeff = 0
        count = 0
        for index in range(len(actor_sets)):
            primary_zones = actor_dicts_list[index]
            primary_zone_actors = primary_zones.keys()
            primary_zone_counts = sum(list(primary_zones.values()))

            for jndex in range(index,len(actor_sets)):
                if(index == jndex):
                    common_actors_term = 1
                    common_actors_coeff += common_actors_term
                    count += 1
                else:
                    secondary_zones = actor_dicts_list[jndex]
                    secondary_zone_actors = secondary_zones.keys()
                    secondary_zone_counts = sum(list(secondary_zones.values()))

                    weights_term = 0
                    for actor in primary_zone_actors:
                        if(actor in secondary_zone_actors):
                            weights_term += (primary_zones[actor]/primary_zone_counts) * \
                                                 (secondary_zones[actor]/secondary_zone_counts)

                    common_actors_term = (2 * weights_term) / (len(primary_zone_actors)+len(secondary_zone_actors))
                    common_actors_coeff += common_actors_term
                    count += 2



    if(count == 0):
        common_actors_coeff = 0
    else:
        common_actors_coeff = common_actors_coeff/count

    return common_actors_coeff


def actors_Jaccard_index(time,dx,gridix,conflict_type,type_of_algo,threshold):
    """Calculates the summation of Jaccard index of conflict zones. Here A and B are the
    number of actors in zone 1 and zone 2 which are being compared.
    Parameters
    ----------
    time : int
    dx : int
    gridix : int
    conflict_type : str
    type_of_algo : str

    Returns
    -------
    float
    """

    acled_data = data_loader.conflict_data_loader(conflict_type)

    zones = conflict_zone_generator(time,dx,gridix,conflict_type,type_of_algo,threshold)

    sorted_zones = sorted(zones , key=len)
    sorted_zones.reverse()

    actor_sets = []
    actor_dicts_list = []
    for index,zone in enumerate(sorted_zones):
        actor_count = zone_actor_counter(time,dx,gridix,conflict_type,type_of_algo,zone,acled_data)

        actor_sets.append(set(list(zip(*actor_count))[0]))
        actor_dicts_list.append(dict(zip(list(zip(*actor_count))[0],list(zip(*actor_count))[1])))


    common_actors_coeff = 0
    count = 0
    for index in range(len(actor_sets)):
        for jndex in range(index,len(actor_sets)):
            if(index == jndex):
                common_actors_term = 1
                common_actors_coeff += common_actors_term
                count += 1
            else:
                common_actors_term = len(actor_sets[index].intersection(actor_sets[jndex])) / (len(actor_sets[index]) + len(actor_sets[jndex]) \
                                                                                             - len(actor_sets[index].intersection(actor_sets[jndex])))

                common_actors_coeff += common_actors_term * 2
                count += 2

    if(count == 0):
        common_actors_coeff = 0
    else:
        common_actors_coeff = common_actors_coeff/count

    return common_actors_coeff
