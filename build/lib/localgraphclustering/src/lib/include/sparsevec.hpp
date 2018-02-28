/**
 * @file sparsevec.hpp
 * A set of functions to manipulate a fast hashtable for doubles
 */

#ifndef _SPARSEVEC_H_
#define _SPARSEVEC_H_

#include <vector>
#include <stdlib.h>
#include <limits>
#include "sparsehash/dense_hash_map.h"

struct sparsevec {
    typedef google::dense_hash_map<size_t,double> map_type;
    map_type map;
    sparsevec()  {
        map.set_empty_key((size_t)(-1));
    }
    /** Get an element and provide a default value when it doesn't exist
     * This command does not insert the element into the vector
     */
    double get(size_t index, double default_value=0.0) {
        map_type::iterator it = map.find(index);
        if (it == map.end()) {
            return default_value;
        } else {
            return it->second;
        }
    }
    
    /** Compute the sum of all the elements
     * Implements compensated summation
     */
    double sum() {
        double s=0.;
        for (map_type::iterator it=map.begin(),itend=map.end();it!=itend;++it) {
            s += it->second;
        }
        return s;
    }
    
    /** Compute the max of the element values
     * This operation returns the first element if the vector is empty.
     */
    size_t max_index() {
        size_t index=0;
        double maxval=std::numeric_limits<double>::min();
        for (map_type::iterator it=map.begin(),itend=map.end();it!=itend;++it) {
            if (it->second>maxval) { maxval = it->second; index = it->first; }
        }
        return index;
    }
};

#endif /*  _SPARSEVEC_H_  */
