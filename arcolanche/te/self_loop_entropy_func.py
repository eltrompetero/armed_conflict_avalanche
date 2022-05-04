from .utils import *


def self_loop_entropy_calculator(*args):
    print("Calculating self loop entropies!")

    time , dx , gridix , conflict_type , number_of_shuffles , time_series_all_pol = args

    valid_polygons = time_series_all_pol.columns.to_list()
    
    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    
    important_indexes = [(0,0,0),(1,1,0),(2,0,1),(3,1,1)]
    
    TE_type = "self_loop_entropy"
    definition_type = ""
    
        
    tile_indexes = polygons.drop(["geometry","neighbors"] , axis=1)
    tiles_transfer_entropy = pd.DataFrame()
    
    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')



    @njit()
    def joint_distribution_xT_xt_new(time_series_two_pols):
        counts_array = np.zeros(4)   #(xT,xt)  0->00 , 1->01 , 2->10 , 3-> 11

        time_series_xT_xt = time_series_two_pols
        #time_series_xT_xt = time_series_xT_xt[:-1]

        time_series_length = len(time_series_xT_xt)


        for i in range(time_series_length-1):           
            if(time_series_xT_xt[i+1][0] == 0 and time_series_xT_xt[i][0] == 0):
                counts_array[0] = counts_array[0] + 1
            elif(time_series_xT_xt[i+1][0] == 0 and time_series_xT_xt[i][0] == 1):
                counts_array[1] = counts_array[1] + 1
            elif(time_series_xT_xt[i+1][0] == 1 and time_series_xT_xt[i][0] == 0):
                counts_array[2] = counts_array[2] + 1
            elif(time_series_xT_xt[i+1][0] == 1 and time_series_xT_xt[i][0] == 1):
                counts_array[3] = counts_array[3] + 1

        distribution_array = np.zeros(4)

        for i in range(len(distribution_array)):
            distribution_array[i] = counts_array[i] / (time_series_length-1)

        return distribution_array


    @njit()
    def distribution_xt_new(time_series_two_pols):
        counts_array = np.zeros(2)   #0->0 , 1->1

        time_series_x = time_series_two_pols[:,0]

        #time_series_x = time_series_x[1:]
        time_series_x = time_series_x[:-1]

        time_series_length = len(time_series_x)

        for i in range(len(time_series_x)):
            if(time_series_x[i] == 0):
                counts_array[0] = counts_array[0] + 1
            elif(time_series_x[i] == 1):
                counts_array[1] = counts_array[1] + 1

        distribution_array = np.zeros(2)

        distribution_array[0] = counts_array[0] / time_series_length
        distribution_array[1] = counts_array[1] / time_series_length

        return distribution_array


    @njit()
    def distribution_xT_new(time_series_two_pols):
        counts_array = np.zeros(2)   #0->0 , 1->1

        time_series_x = time_series_two_pols[:,0]

        #time_series_x = time_series_x[1:]
        time_series_x = time_series_x[1:]

        time_series_length = len(time_series_x)

        for i in range(len(time_series_x)):
            if(time_series_x[i] == 0):
                counts_array[0] = counts_array[0] + 1
            elif(time_series_x[i] == 1):
                counts_array[1] = counts_array[1] + 1

        distribution_array = np.zeros(2)

        distribution_array[0] = counts_array[0] / time_series_length
        distribution_array[1] = counts_array[1] / time_series_length

        return distribution_array


    def self_loop_entropy_calculator(distribution_data):
        xT_xt , xt , xT = distribution_data
        return (xT_xt * np.log2((xT_xt) / (xt*xT)))


    def transfer_entropy_tiles(tile_data , shuffle_number):
        primary_tile_number = tile_data["index"]
        transfer_entropy_list = []
        if(primary_tile_number in valid_polygons):

            #tile_time_series_creation(primary_tile_number)
            time_series = time_series_all_pol[[primary_tile_number]]
            time_series.columns = ["x"]
        
            if(shuffle_number != 0):
                np.random.shuffle(time_series["x"].to_numpy())
                time_series = time_series.to_numpy()
            else:
                time_series = time_series.to_numpy()        
            
            
            joint_distribution_xT_xt_list = joint_distribution_xT_xt_new(time_series)
            distribution_xt_list = distribution_xt_new(time_series)
            distribution_xT_list = distribution_xT_new(time_series)
            transfer_entropy_individual_terms = []
            
            for i,j,k in important_indexes:
                if(joint_distribution_xT_xt_list[i] == 0 or distribution_xt_list[j] == 0 or distribution_xT_list[k] == 0):
                    pass
                else:
                    transfer_entropy_individual_terms.append(self_loop_entropy_calculator((joint_distribution_xT_xt_list[i],distribution_xt_list[j],distribution_xT_list[k])))
            transfer_entropy_value = sum(transfer_entropy_individual_terms)
            transfer_entropy_list.append(transfer_entropy_value)
        else:
            transfer_entropy_list.append("NA")
    
        return transfer_entropy_list
    
    for shuffle_number in range(number_of_shuffles+1):
        if(shuffle_number % 10 == 0):
            print("Shuffle Number:", shuffle_number)

        tiles_transfer_entropy[f"{TE_type}_{str(shuffle_number)}"] = tile_indexes.apply(transfer_entropy_tiles, args=(shuffle_number,) , axis=1)
        
    #tiles_transfer_entropy.to_pickle(f"data_{str(conflict_type)}/{TE_type}/tiles_{TE_type}_{definition_type}{str(time)}_{str(dx)}_{str(number_of_shuffles)}")
    
    return tiles_transfer_entropy
