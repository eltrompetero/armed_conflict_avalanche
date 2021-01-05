/*****************************************************************************************
 * CPP version of utils.py module.
 * Author : Eddie Lee, edlee@santafe.edu
*****************************************************************************************/
#define BOOST_TEST_DYN_LINK
#include <stdio.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <Python.h>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>

//for debugging and printing
#define Py_PRINT_STR 0

using namespace boost::python;
namespace np = boost::python::numpy;


list cluster_avalanche(np::ndarray &day,
                       np::ndarray &pixel,
                       int A,
                       dict cellneighbors,
                       double counter_mx) {
    /* Cluster events into avalanches by connecting all pixels that are neighbors in space
    that are active within the specified time interval A.
    
    Parameters
    ----------
    day : ndarray of int
        Integer days of event.
    pixel : ndarray of int
        Spatial pixels for each event.
    A : int
        Separation time scale
    cellneighbors : dict
        Key indicates which pixel and values indicate all neighboring pixels.
    counter_mx : double, np.inf
        If specified, only construct this number of avalanches.
       
    Returns
    -------
    list of list of ints
        Each list indicates the indices of points that belong to one avalanche or another.
    */
    
    list remaining = list();  // unclustered event ix
    list clustered = list();  // clustered event ix
    list avalanches = list();  // groups of conflict avalanches of event ix
    list thisCluster, toConsider, thisNeighbors;
    std::vector<bool> selectix (len(day), false);
    int thisEvent, thisPix, ix;
    int counter = 0;

    //initialize vars
    for (int i=0; i<day.shape(0); i++) {
        remaining.append(i);
    }

    while (len(remaining)>0 && counter<counter_mx) {
        thisCluster = list();
        toConsider = list();  // events whose neighbors remain to be explored

        //initialize a cluster
        toConsider.append(remaining[0]);

        while (len(toConsider)>0) {
            //add this event to the cluster
            thisEvent = extract<int>(toConsider.pop(0));
            remaining.pop(remaining.index(thisEvent));
            thisCluster.append(thisEvent);
            clustered.append(thisEvent);
            thisPix = extract<int64_t>(pixel[thisEvent]);
            thisNeighbors = extract<list>(cellneighbors[thisPix]);

            //find all the neighbors of this point amongst the remaining points
            for (int i=0; i<len(remaining); i++) {
                ix = extract<int>(remaining[i]);
                //first filter all other events not within time dt
                if (abs(int(extract<int64_t>(day[ix]) -
                            extract<int64_t>(day[thisEvent])))<=A) {
                    //for debugging
                    //PyObject_Print(object(thisNeighbors[0]).ptr(), stdout, Py_PRINT_STR);
                    //std::cout << std::endl;

                    //now filter by cell adjacency
                    if (thisNeighbors.count(int(extract<int64_t>(pixel[ix])))!=0 &&
                        thisCluster.count(ix)==0 &&
                        toConsider.count(ix)==0) {
                        toConsider.append(ix);
                    }
                }//end if
            }//end for
        }//end while
        avalanches.append(thisCluster);
        counter++;
    }//end while

    return avalanches;
};//end cluster_avalanche



//set up Python interface
BOOST_PYTHON_MODULE(utils_ext) {
    Py_Initialize();
    np::initialize();

    def("cluster_avalanche", cluster_avalanche);
}
