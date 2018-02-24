#ifndef MAXFLOW_cpp
#define MAXFLOW_cpp

#include "routines.hpp"
#include <list>
#include <climits>
#include <utility>
#include <iostream>

/*
Based on http://www.geeksforgeeks.org/dinics-algorithm-maximum-flow/
*/

using namespace std;

// Finds if more flow can be sent from s to t.
// Also assigns levels to nodes.
template<typename vtype, typename itype>
bool graph<vtype,itype>::BFS(vtype s, vtype t, vtype V)
{
    for (vtype i = 0 ; i < V ; i++) {
        level[i] = -1;
    }
 
    level[s] = 0;  // Level of source vertex
    //cout << "start level" << level[t] << endl;
 
    // Create a queue, enqueue source vertex
    // and mark source vertex as visited here
    // level[] array works as visited array also.
    list< vtype > q;
    q.push_back(s);
 
    typename vector<Edge<vtype,itype>>::iterator i;
    while (!q.empty()) {
        vtype u = q.front();
        q.pop_front();
        for (i = adj[u].begin(); i != adj[u].end(); i++) {
            Edge<vtype,itype> &e = *i;
            if (level[e.v] < 0  && e.flow < e.C) {
                // Level of current vertex is,
                // level of parent + 1
                level[e.v] = level[u] + 1;
 
                q.push_back(e.v);
            }
        }
    }
 
    // IF we can not reach to the sink we
    // return false else true
    //cout << "level" << level[t] << endl;
    return level[t] < 0 ? false : true ;
}
 
// A DFS based function to send flow after BFS has
// figured out that there is a possible flow and
// constructed levels. This function called multiple
// times for a single call of BFS.
// flow : Current flow send by parent function call
// start[] : To keep track of next edge to be explored.
//           start[i] stores  count of edges explored
//           from i.
//  u : Current vertex
//  t : Sink
template<typename vtype, typename itype>
double graph<vtype,itype>::sendFlow(vtype u, double flow, vtype t, vtype start[])
{
    // Sink reached
    if (u == t)
        return flow;
 
    // Traverse all adjacent edges one -by - one.
    for (  ; start[u] < adj[u].size(); start[u]++) {
        // Pick next edge from adjacency list of u
        Edge<vtype,itype> &e = adj[u][start[u]]; 
                                     
        if (level[e.v] == level[u]+1 && e.flow < e.C) {
            // find minimum flow from u to t
            double curr_flow = min(flow, e.C - e.flow);
 
            double temp_flow = sendFlow(e.v, curr_flow, t, start);
 
            // flow is greater than zero
            if (temp_flow > 0) {
                // add flow  to current edge
                e.flow += temp_flow;
 
                // subtract flow from reverse edge
                // of current edge
                adj[e.v][e.rev].flow -= temp_flow;
                return temp_flow;
            }
        }
    }
 
    return 0;
}
 

template<typename vtype, typename itype>
void graph<vtype,itype>::find_cut(vtype u, vector<bool>& mincut, vtype& length)
{
    mincut[u] = true;
    length ++;
    //cout << "current len: " << length << endl;
    for (vtype i = 0 ; i < adj[u].size(); i ++) {
        Edge<vtype,itype> e = adj[u][i];
        if (e.flow < e.C && mincut[e.v] == false) {
            find_cut(e.v,mincut,length);
        }
    }
}


// Returns maximum flow in graph
template<typename vtype, typename itype>
pair<double,vtype> graph<vtype,itype>::DinicMaxflow(vtype s, vtype t, vtype V, vector<bool>& mincut)
{
    // Corner case
    if (s == t)
        return make_pair(-1,0);
 
    double total = 0;  // Initialize result
 
    // Augment the flow while there is path
    // from source to sink
    vtype *start = NULL;
    //cout << INT_MAX << endl;
    while (BFS(s, t, V) == true){
        // store how many edges are visited
        // from V { 0 to V }
        if (start != NULL) {
            delete[] start;
        }
        start = new vtype[V+1];
        fill(start,start+V+1,0);
 
        // while flow is not zero in graph from S to D
        double flow = sendFlow(s, INT_MAX, t, start);
        //cout << flow << endl;
        while (flow > 0) {
        	// Add path flow to overall flow
            total += flow;
            flow = sendFlow(s, INT_MAX, t, start);
        }
        //cout << "BFS" << endl;
    }
    //cout << "out" << endl;
    vtype length = 0;
    fill(mincut.begin(),mincut.end(),false);
    find_cut(s,mincut,length);
    //cout << "length " << length << endl;
 
    // return maximum flow
    return make_pair(total,length);
}

#endif