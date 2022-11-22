# ====================================================================================== #
# Module for analysing avalanches.
# Author: Niraj Kushwaha
# ====================================================================================== #
from .utils import *
from .data import ACLED2020
import numpy_indexed

class ConflictZones():
    def __init__(self, dt, dx, threshold,
                 gridix=0,
                 type_of_algo = "te",
                 conflict_type='battles',
                 iprint=False,
                 ):
        """
        Parameters
        ----------
        dt : int
            Time separation scale.
        dx : int
            Inverse distance separation scale.
        type_of_algo : str
            Specify the avalanche type (te/st/null/null_reassign)
        threshold : int
        gridix : int, 0
            Random Voronoi grid index.
        conflict_type : str, 'battles'
        iprint : bool, False
        """
        
        #### Add checks for the values below later ####
        self.dt = dt    
        self.dx = dx
        self.type_of_algo = type_of_algo
        self.threshold = threshold
        self.gridix = gridix
        self.conflict_type = conflict_type
        self.iprint = iprint

        self.box_path = (f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/" +
                            f"{type_of_algo}_ava_{str(dt)}_{str(dx)}.p")

        with open(self.box_path,"rb") as f:
            ava = pickle.load(f)
        self.ava_box = ava["ava_box"]
        self.ava_event = ava["ava_event"]

        self.acled_data = ACLED2020.battles_df()



    def common_actors_coeff_calculator(self, weighted=True):
        """Calculates the summation of ratio of common actors and sum of number of actors in
        each pair of conflict zones.

        Parameters
        ----------
        weighted : bool , True

        Returns
        -------
        float
        """

        zones = self.generator()

        sorted_zones = sorted(zones , key=len)
        sorted_zones.reverse()

        actor_sets = []
        actor_dicts_list = []
        for index,zone in enumerate(sorted_zones):
            actor_count = self.zone_actor_counter(zone)

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
                        common_actors_term = 1   #### If we want diagnoals to be one ####
                        common_actors_coeff += common_actors_term
                        count += 1

                        #### If we want diagonals to not be one ####
                        #weights_term = 0
                        #for actor in primary_zone_actors:
                        #    if(actor in primary_zone_actors):
                        #        weights_term += (primary_zones[actor]/primary_zone_counts) * \
                        #                             (primary_zones[actor]/primary_zone_counts)
#
                        #common_actors_term = (2 * weights_term) / (len(primary_zone_actors)+len(primary_zone_actors))
                        #common_actors_coeff += common_actors_term
                        #count += 1


                        #### Not considering diagonal terms at all
                        #pass
                    else:
                        secondary_zones = actor_dicts_list[jndex]
                        secondary_zone_actors = secondary_zones.keys()
                        secondary_zone_counts = sum(list(secondary_zones.values()))

                        #weights_term = 0
                        #for actor in primary_zone_actors:
                        #    if(actor in secondary_zone_actors):
                        #        weights_term += (primary_zones[actor]/primary_zone_counts) * \
                        #                             (secondary_zones[actor]/secondary_zone_counts)
#
                        #common_actors_term = (2 * weights_term) / (len(primary_zone_actors)+len(secondary_zone_actors))
                        #common_actors_coeff += common_actors_term
                        #count += 2


                        ##### Only similarity term (getting rid of the extra normalization)
                        weights_term = 0
                        for actor in primary_zone_actors:
                            if(actor in secondary_zone_actors):
                                weights_term += (primary_zones[actor]/primary_zone_counts) * \
                                                     (secondary_zones[actor]/secondary_zone_counts)

                        common_actors_term =  weights_term
                        common_actors_coeff += common_actors_term
                        count += 2                        



        if(count == 0):
            common_actors_coeff = 0
        else:
            common_actors_coeff = common_actors_coeff/count

        return common_actors_coeff


    def generator(self):
        """Generates conflict zones across Africa using aggregation of conflict
        avalanches.

        Returns
        -------
        list of arrays
            the arrays contain polygon number/id of polygons in each conflict
            zone
        """

        size_arr = np.array([len(np.unique(list(zip(*i))[0])) for i in self.ava_box])

        if(type(self.threshold) == int):
            threshold_size = self.threshold
        elif(type(self.threshold) == float):
            threshold_size = max(size_arr) * self.threshold

        ava_box_threshold = np.array(self.ava_box , dtype=object)[np.where(size_arr > threshold_size)[0]].tolist()

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


    def zone_actor_counter(self,zone):
        """Find the actor composition in a given zone for a particular scale and gridix.
 
        Returns
        -------
        list of tuples
            First entry of tuple corresponds to the key of actor in the
            actor_dict. Second entry of tuple correponds to the total number of
            occurances of this actor in the avalanches present in entered zone.
        """

        actor_dict = self.actor_dict_generator()
        in_zone_events = self.events_in_zone(zone)
        actor_count = self.event_actor_counter(in_zone_events,actor_dict)

        return actor_count


    def actor_dict_generator(self):
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

        acled_data_actors = self.acled_data[["ACTOR1","ACTOR2"]]

        actor1_arr = (acled_data_actors["ACTOR1"]).to_numpy()
        actor2_arr = (acled_data_actors["ACTOR2"]).to_numpy()

        actors_arr = np.concatenate((actor1_arr,actor2_arr))
        actors_arr = np.unique(actors_arr)

        actors_dict = {}
        for index,actor in zip(range(len(actors_arr)),actors_arr):
            actors_dict[index] = (actor)

        return actors_dict


    def events_in_zone(self,zone):
        """Find all the vents that are in a particular zone.

        Parameters
        ----------
        zone : list
            List containing the polygon indexes of polygons that are in the selected zone.
        Returns
        -------
        list
            A list of all the events that are in a particular zone.
        """

        zone_set = set(zone)
        in_ava_indexes = []
        for index,ava in enumerate(self.ava_box):
            ava_pol_set = set(list(zip(*ava))[0])
            if(len(ava_pol_set.intersection(zone_set)) != 0):
                in_ava_indexes.append(index)

        ava_event = np.array(self.ava_event , dtype=object)

        in_zone_events = ava_event[in_ava_indexes]
        in_zone_events = [x for l in in_zone_events for x in l]

        return in_zone_events



    def event_actor_counter(self , event_nums , actors_dict):
        """Finds the actor composition in the list of entered event numbers.
        Here actor1 and actor2 are treated the same.

        Parameters
        ----------
        event_nums : list
            list of event numbers
        actors_dict : dict
            Dictionary containing all unique actors(actor1 and actor2 combined)
            and their corresponding keys.

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

        acled_data_actors = self.acled_data[["ACTOR1","ACTOR2"]]

        actors_event = acled_data_actors.loc[event_nums].to_numpy()
        actors_event = actors_event.reshape(actors_event.shape[0]*actors_event.shape[1])

        actor_count = []
        for actor_group in numpy_indexed.group_by(actors_event).split_array_as_list(actors_event):
            actor_count.append((actor_dict_lookup(actor_group[0]),len(actor_group)))

        return actor_count


