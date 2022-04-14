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


std::vector<int>::iterator index(std::vector<int> &v, int val) {
    /* Find the index of an element.
     */

    std::vector<int>::iterator it;

    for (it=v.begin(); it!=v.end(); ++it) {
        if (*it==val) {
            break;
        }
    }
    
    if (it!=v.end()) {
        return it;
    }
    throw;
}

int count(std::vector<int> &v, int val) {
    /* Count the number of times val appears in vector.
     */
    
    int counter = 0;

    for (std::vector<int>::iterator it=v.begin(); it!=v.end(); ++it) {
        if (*it==val) {
            counter++;
        }
    }
    return counter;    
}

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
    
    std::vector<int> remaining;  // unclustered event ix
    std::vector<int> toConsider;
    list avalanches = list();  // groups of conflict avalanches of event ix
    list thisCluster, thisNeighbors;
    std::vector<bool> selectix (len(day), false);
    int thisEvent, thisPix, ix;
    int counter = 0;
    std::vector<int>::iterator it;

    //initialize vars
    for (int i=0; i<day.shape(0); i++) {
        remaining.push_back(i);
    }

    while (remaining.size()>0 && counter<counter_mx) {
        thisCluster = list();
        toConsider.clear();  // events whose neighbors remain to be explored

        //initialize a cluster
        toConsider.push_back(remaining[0]);

        while (toConsider.size()) {
            //add this event to the cluster
            thisEvent = toConsider[0];
            toConsider.erase(toConsider.begin());
            it = index(remaining, thisEvent);
            remaining.erase(it);
            thisCluster.append(thisEvent);
            thisPix = extract<int64_t>(pixel[thisEvent]);
            thisNeighbors = extract<list>(cellneighbors[thisPix]);

            //find all the neighbors of this point amongst the remaining points
            for (int i=0; i<remaining.size(); i++) {
                ix = remaining[i];
                //first filter all other events not within time dt
                if (abs(int(extract<int64_t>(day[ix]) -
                            extract<int64_t>(day[thisEvent])))<=A) {
                    //for debugging
                    //PyObject_Print(object(thisNeighbors[0]).ptr(), stdout, Py_PRINT_STR);
                    //std::cout << std::endl;

                    //now filter by cell adjacency
                    if (thisNeighbors.count(int(extract<int64_t>(pixel[ix])))!=0 &&
                        thisCluster.count(ix)==0 &&
                        count(toConsider, ix)==0) {
                        toConsider.push_back(ix);
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
