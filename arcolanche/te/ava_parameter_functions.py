from .utils import *


def fatalities(avalanche_data):
    dt = avalanche_data.groupby("avalanche_number").sum()["fatalities"]          

    return dt

def reports(avalanche_data):
    dt = avalanche_data.groupby("avalanche_number").size()

    return dt

def duration(avalanche_data):
    duration = avalanche_data.groupby("avalanche_number").agg({"day" : ["min" , "max"]})
    duration["duration"] = duration["day"]["max"] - duration["day"]["min"]
    dt = duration["duration"]

    return dt

def sites(avalanche_data):
    t = avalanche_data.groupby("avalanche_number").groups
    sites = []
    for i in range(len(t)):
        tl = t.get(i).to_list()
        tl = avalanche_data["polygon_number"].loc[tl]
        tl = len(np.unique(tl))
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


def diameter_events(avalanche_data,centers,location_data_arr,ava_box):
    
    avalanche_data["event_number"] = avalanche_data.index
    avalanche_data_arr = np.array(avalanche_data[["event_number","polygon_number","bins"]])

    def center_distance(args):
        pol1 , pol2 = args
        
        point1 = (centers.iloc[pol1].x,centers.iloc[pol1].y)
        point2 = (centers.iloc[pol2].x,centers.iloc[pol2].y)
        
        return vincenty(point1,point2)
    
    def events_finder(args):
        pol , time_bin = args
        
        return np.where((avalanche_data_arr[:,1] == pol) & (avalanche_data_arr[:,2] == time_bin))[0]
        
    def event_distance(args):
        event1 , event2 = args
        
        point1 = (location_data_arr[event1][0],location_data_arr[event1][1])
        point2 = (location_data_arr[event2][0],location_data_arr[event2][1])
        
        return vincenty(point1,point2)
    
    dt = []
    for ava,i in zip(ava_box,range(len(ava_box))):
        polys = np.unique(np.array(list(zip(*ava))[0]))
        
        if(len(polys) == 1):
            boxes_to_check = ava
            events = []
            for box in boxes_to_check:
                events.extend(events_finder(box).tolist())     
        
            ## Finding distance between events and saving the max distance as the distance of the avalanche
            event_pairs = list(combinations(events,2))
            if(len(event_pairs) == 0):
                dt.append(0)
            else:
                distances = []
                for pair in event_pairs:
                    distances.append(event_distance(pair))
                
                dt.append(max(distances))    
        
        else:
            ## Finding polygons with farthest centers
            pol_pairs = list((combinations(np.unique(np.array(list(zip(*ava))[0])),2)))
            
            distances = []
            for pair in pol_pairs:
                distances.append(center_distance(pair))
            
            distances = np.array(distances)
            max_pair = pol_pairs[np.where(distances == max(distances))[0][0]]
            
            
            ## Finding all events that are in the two farthest polygons in a given avalanche
            pols_arr = np.array(list((zip(*ava)))[0])
            
            boxes_to_check = np.array(ava)[np.where((pols_arr == max_pair[0]) | (pols_arr == max_pair[1]))[0]].tolist()
            boxes_to_check = [(i,j) for i,j in boxes_to_check]
            
            events1 = []
            events2 = []
            for box in boxes_to_check:
                if(box[0] == max_pair[0]):
                    events1.extend(events_finder(box).tolist())
                elif(box[0] == max_pair[1]):
                    events2.extend(events_finder(box).tolist())
            
            ## Finding distance between events and saving the max distance as the distance of the avalanche
            event_pairs = list(itertools.product(events1,events2))
            distances = []
            for pair in event_pairs:
                distances.append(event_distance(pair))
            
            dt.append(max(distances))
            
    return dt



def diameter_true(ava_event,location_data_arr):
    def event_distance(args):
        event1 , event2 = args
        
        point1 = (location_data_arr[event1][0],location_data_arr[event1][1])
        point2 = (location_data_arr[event2][0],location_data_arr[event2][1])
        
        return vincenty(point1,point2)
    
    dt = []
    for i,ava in enumerate(ava_event):
        event_pairs = list(combinations(ava,2))
        
        if(event_pairs):
            distances = []
            for pair in event_pairs:
                distances.append(event_distance(pair))
                
            dt.append(max(distances))
        else:
            dt.append(0)
            
    return dt