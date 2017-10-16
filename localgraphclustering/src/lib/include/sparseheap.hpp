/**
 * @file sparseheap.hpp
 * A set of functions to manipulate a max-heap as a set of 3 hashtables
 */

#ifndef _SPARSEHEAP_H_
#define _SPARSEHEAP_H_

#include <vector>
#include <stdlib.h>
#include <limits>
#include "sparsehash/dense_hash_map.h"

template< typename index_type, typename value_type, class local_index_type = size_t > 
struct sparse_max_heap 
{
    // setup the data structure
    // index_to_lindex takes the original, sparse indices, and maps them
    // to a local index in our array. 
    
    typedef google::dense_hash_map<index_type, local_index_type> local_map;
    local_map index_to_lindex;
    local_index_type nextind;
    std::vector< index_type > lindex_to_index;
    
    // all of these vectors use our linear index
    std::vector< value_type > values;
    std::vector< local_index_type > T;
    std::vector< local_index_type > L; // L[i] is the location of item i.
    size_t hsize;
    
    // set a sentinal. 
    local_index_type lastval; // = std::numeric_limits<local_index_type>::max();  

    double look_max(){
        return values[T[0]];    
    }
    

    index_type extractmax(value_type &val) {
        local_index_type lindmax = T[0];
        index_type rval = lindex_to_index[lindmax];
        val = values[lindmax];
        T[0] = T[hsize-1]; // move the last entry to the root
        L[T[0]] = 0;       // update the index to reflect this
        L[lindmax] = lastval; // set the location o the last index to be zero.
        hsize --;
        heap_down(0);
        return (rval);
    }
    
    
    void heap_down(local_index_type k){
        local_index_type heapk = T[k];
        while (1) {
            local_index_type i=2*(k+1)-1;
            if (i>=hsize) { break; } /* end of heap */
            if (i<hsize-1) {
                /* pick between children (unneeded if i==heap_size-1) */
                local_index_type left=T[i];
                local_index_type right=T[i+1];
                if (values[right] > values[left]) {
                    i=i+1; /* pick the larger child */
                }
            }
            if (values[heapk] > values[T[i]]) {
                /* k is larger than both children, so end */
                break;
            } else {
                T[k] = T[i]; 
                L[T[i]]=k;
                T[i] = heapk; 
                L[heapk] = i;
                k=i;
            }
        }
    }
        
    local_index_type heap_up(local_index_type j){
        while (1) {
            if (j==0) { break; } /* the element is at the top */
            local_index_type heapj = T[j];
            local_index_type j2 = (j-1)/2;  /* otherwise, compare with its parent */
            local_index_type heapj2 = T[j2];
            if (values[heapj2] > values[heapj]) {
                break; /* the parent is larger, so stop */
            } else {
                /* the parent is smaller, so swap */
                T[j2] = heapj; L[heapj] = j2;
                T[j] = heapj2; L[heapj2] = j;
                j = j2;
            }
        }
        return j;
    }
    
    /** When altering an entry of map, this checks
     *  whether that entry is already in map, or if a new entry
     *  need be made in the heap before adding 'value' to map[index].
     */
    void update(index_type index, value_type value) {
        local_index_type lindex; 
        
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
            // if 'index' was never in L, or 
            // if it was but has been removed: 
            // insert new entry at the end
            values[lindex] = value;
            T[hsize] = lindex;
            L[lindex] = hsize;
            hsize++;
            heap_up(hsize-1);
        }
        else{ // update old entry
            values[lindex] += value;
            heap_up(L[lindex]);
        }
    }
    
    sparse_max_heap(size_t initial_size) 
    : nextind(0), lindex_to_index(initial_size), 
      values(initial_size), T(initial_size), L(initial_size), hsize(0)
    { 
    	lastval = std::numeric_limits<local_index_type>::max();
	index_to_lindex.set_empty_key(std::numeric_limits<index_type>::max()); 
    }
};

#endif  /*  _SPARSEHEAP_H_  */
