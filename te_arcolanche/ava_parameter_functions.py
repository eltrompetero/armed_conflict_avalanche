from .utils import *


def fatalities(avalanche_data):
    dt = avalanche_data.groupby("avalanche_number").sum()["fatalities"]          

    return dt

def reports(avalanche_data):
    dt = avalanche_data.groupby("avalanche_number").size()

    return dt

def duration(avalanche_data):
    duration = avalanche_data.groupby("avalanche_number").agg({"days" : ["min" , "max"]})
    duration["duration"] = duration["days"]["max"] - duration["days"]["min"]
    dt = duration["duration"]

    return dt

def sites(avalanche_data):
    t = avalanche_data.groupby("avalanche_number").groups
    sites = []
    for i in range(len(t)):
        tl = t.get(i).to_list()
        tl = avalanche_data["polygon_number"].loc[tl]
        tl = len(unique(tl))
        sites.append(tl)
    dt = pd.DataFrame(sites)
    #dt = sites

    return dt[0]

def diameter_centers(avalanche_data , centers):
    groups = avalanche_data.groupby("avalanche_number")
    
    dt = []
    for group_number in range(len(groups.size())):
    
        pol_data = groups.get_group(group_number)
        pol_data = pd.DataFrame(pd.unique(pol_data["polygon_number"]) , columns=["polygons"])
        if(len(pol_data) == 1):
            dt.append(0)
        else:
            pol_data["center"] = ""
    
            for poly,index in zip(pol_data["polygons"] , range(size(pol_data))):
                pol_data.at[index,"center"] = centers.iloc[int(poly)]
                
            pol_data["center_tuple"] = [(x_cor , y_cor) for x_cor,y_cor in zip(geopandas.GeoSeries(pol_data["center"]).x , geopandas.GeoSeries(pol_data["center"]).y)]
            
            def pol_distance(pol_list , point):
                distance = vincenty(point , pol_list)
                return distance
            
            distance_list = []
            for index in range(len(pol_data)):
                distance_list = distance_list + pol_data["center_tuple"][index:].apply(pol_distance , args=(pol_data["center_tuple"].iloc[index],)).to_list()
                
            dt.append(max(distance_list))
    dt = np.array(dt)
    
    #dt,d = np.histogram(dt , bins=100)

    return dt


def diameter_events(avalanche_data , centers , data):
    def distance_events(event_pair):
        event_1 , event_2 = event_pair

        point_1 = (data.iloc[event_1]["longitude"] , data.iloc[event_1]["latitude"])
        point_2 = (data.iloc[event_2]["longitude"] , data.iloc[event_2]["latitude"])

        distance = vincenty(point_1 , point_2)

        return distance

    def distance_pol(pol_pair):
        pol_1 , pol_2 = pol_pair

        point_1 = pol_data2.loc[pol_data2["polygons"] == pol_1]["center_tuple"].to_list()[0]
        point_2 = pol_data2.loc[pol_data2["polygons"] == pol_2]["center_tuple"].to_list()[0]

        distance = vincenty(point_1 , point_2)

        return distance

    groups_avalanchenumber = avalanche_data.groupby("avalanche_number")

    dt = []
    for group_number in range(len(groups_avalanchenumber.size())):

        pol_data1 = groups_avalanchenumber.get_group(group_number)
        pol_data2 = pd.DataFrame(pd.unique(pol_data1["polygon_number"]) , columns=["polygons"])
        if(len(pol_data1) == 1):
            dt.append(0)
        else:
            if(len(pol_data2) == 1):
                event_pairs = []
                events_list = groups_avalanchenumber.get_group(group_number)["Unnamed: 0"].to_list()
                for pair in itertools.combinations(events_list, r=2):
                    event_pairs.append(pair)
                event_pairs_df = pd.DataFrame(event_pairs)

                dt.append(max(event_pairs_df.apply(distance_events , axis=1)))
            else:
                pol_data2["center"] = ""
                for poly,index in zip(pol_data2["polygons"] , range(size(pol_data2))):
                    pol_data2.at[index,"center"] = centers.iloc[int(poly)]

                pol_data2["center_tuple"] = [(x_cor , y_cor) for x_cor,y_cor in zip(geopandas.GeoSeries(pol_data2["center"]).x , geopandas.GeoSeries(pol_data2["center"]).y)]

                pol_pairs = []
                pol_list = pol_data2["polygons"].to_list()
                for pair in itertools.combinations(pol_list, r=2):
                    pol_pairs.append(pair)
                pol_pairs_df = pd.DataFrame(pol_pairs)

                pol_pairs_df["distance"] = pol_pairs_df.apply(distance_pol , axis=1)

                max_dist_pol_pair = (pol_pairs_df.iloc[pol_pairs_df["distance"].idxmax()][0] , pol_pairs_df.iloc[pol_pairs_df["distance"].idxmax()][1])

                events_list_1 = groups_avalanchenumber.get_group(group_number).loc[groups_avalanchenumber.get_group(group_number)["polygon_number"] == max_dist_pol_pair[0]]["Unnamed: 0"].to_list()
                events_list_2 = groups_avalanchenumber.get_group(group_number).loc[groups_avalanchenumber.get_group(group_number)["polygon_number"] == max_dist_pol_pair[1]]["Unnamed: 0"].to_list()

                event_pairs_df = pd.DataFrame(itertools.product(events_list_1, events_list_2))

                dt.append(max(event_pairs_df.apply(distance_events , axis=1)))

    dt = np.array(dt)

    #dt,d = np.histogram(dt , bins=1000)

    return dt