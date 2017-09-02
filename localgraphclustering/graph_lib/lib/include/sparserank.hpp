/**
 * @file sparserank.hpp
 * A set of functions to store a set of elements in sorted order
 * for use when inserted elements are expected to stay close to the back
 */

#ifndef _SPARSERANK_H_
#define _SPARSERANK_H_


#include <vector>
#include <stdlib.h>
#include <limits>
#include "sparsehash/dense_hash_map.h"


template< typename index_type, typename value_type, class local_index_type = size_t > 
struct sparse_max_rank
{
    // setup the data structure
    // index_to_lindex takes the original, sparse indices, and maps them
    // to a local index in our array. 
    
    typedef google::dense_hash_map<index_type, local_index_type> local_map;
    local_map index_to_lindex;                 
    local_index_type nextind;
    std::vector< index_type > lindex_to_index; 
    
    // all of these vectors use our linear index
    std::vector< value_type > values;  // V[i] is value of entry with lindex i
    std::vector< local_index_type > T; // T[j] is lindex in L,V of entry with rank j
    std::vector< local_index_type > L; // L[i] is the position in T of entry with lindex i.
    size_t hsize; // always equal to size of heap = length of T,L,values
    
    // set a sentinal. 
    local_index_type lastval; // = std::numeric_limits<local_index_type>::max();  

    sparse_max_rank(size_t initial_size) 
    : nextind(0), lindex_to_index(initial_size), 
      values(initial_size), T(initial_size), L(initial_size), hsize(0)
    { 
    	lastval = std::numeric_limits<local_index_type>::max();
	    index_to_lindex.set_empty_key(std::numeric_limits<index_type>::max()); 
    }

    local_index_type index_to_rank(index_type j){
        if ( index_to_lindex.count(j) == 0 ){ return lastval; }
        index_type lind = index_to_lindex[j];
        if (lind >= hsize) { return lastval; }
        return L[lind];
    }

    index_type rank_to_index(local_index_type j){
        if (j >= hsize) { return lastval; }
        return lindex_to_index[T[j]];
    }

    /** 
     *  if an entry is inserted or altered, this moves the entry up the rank
     *  via swaps until the rank is re-sorted
     */
    local_index_type rank_up(local_index_type j){
        while (1) {
            if (j==0) { break; } 
            local_index_type listj = T[j];
            local_index_type j2 = j-1;
            local_index_type listj2 = T[j2];
            if (values[listj2] > values[listj]) { // entry is in correct ranking now
                break;
            } else { // child is bigger: swap it up
                T[j2] = listj; L[listj] = j2;
                T[j] = listj2; L[listj2] = j;
                j=j2;
            }
        }
        return j;
    }
    
    
    
    /** When altering an entry of map, this checks
     *  whether that entry is already in map, or if a new entry
     *  need be made in the list before adding 'value' to map[index].
     *
     *  function returns the place in the ranking where the updated entry ends up.
     *  `start_index` is set to the ranking where the entry begins (if the entry was
     *      not already in the array, then its start_index is set to the back.)
     */
    local_index_type update(index_type index, value_type value, local_index_type &start_index)
    {
        local_index_type lindex; 
        start_index = hsize; // set start_index to the current end of the array
        // check if we need to grow our vectors
        if ( index_to_lindex.count(index) == 0 ) {
            // we have not seen this entry before, and we need
            // to create a local index for it.
            if (nextind >= lindex_to_index.size()) {
                // this means that we need to grow all of the arrays
                size_t nextsize = 10*lindex_to_index.size();
                lindex_to_index.resize(nextsize);           
                values.resize(nextsize);
                T.resize(nextsize);
                L.resize(nextsize);
            }
            // create the entries of the map
            lindex = nextind;    
            index_to_lindex[index] = lindex;
            lindex_to_index[lindex] = index;
            L[lindex] = lastval;
            
            // set the nextval to be one larger
            nextind++ ;
        } else {
            lindex = index_to_lindex[index];
        }

        if ( L[lindex] == lastval ){
            // if 'index' is not in L:
            // insert new entry at the end
            values[lindex] = value;
            T[hsize] = lindex;
            L[lindex] = hsize;
            hsize++;
            return rank_up(hsize-1);
        }
        else{ // update old entry
            values[lindex] += value;
            start_index = L[lindex]; // entry already in: set start_index to rank j s.t. T[j] =  lindex
            return rank_up(start_index);
        }
    }
};

#endif  /*  _SPARSERANK_H_  */
