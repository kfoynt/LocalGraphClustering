#ifndef MAXFLOW_cpp
#define MAXFLOW_cpp

#include "routines.hpp"
#include <list>
#include <climits>
#include <utility>
#include <iostream>
#include <stack> 
#include <algorithm> 

/*
Based on http://www.geeksforgeeks.org/dinics-algorithm-maximum-flow/
*/

using namespace std;

// Finds if more flow can be sent from s to t.
// Also assigns levels to nodes.
template<typename vtype, typename itype>
bool graph<vtype,itype>::BFS(vtype s, vtype t, vtype V)
{
    //cout << "start BFS" << endl;
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
// UPDATE: convert original recursion approach to a iteration approach
// https://www.codeproject.com/Articles/418776/How-to-replace-recursive-functions-using-stack-and

template<typename vtype, typename itype>
double graph<vtype,itype>::sendFlow(vtype init_u, double init_flow, vtype t, vector<vtype>& start, vector<pair<int,double>>& SnapShots)
{   

    //pair<int,double> SnapShots[n];

    double retVal = 0;

    stack<vtype> SnapShotStack;

    //SnapShot currentSnapShot = SnapShot(init_u,init_flow);
    //SnapShot* currentPtr;
    SnapShots[init_u].first = 0;
    SnapShots[init_u].second = init_flow;

    SnapShotStack.push(init_u);
    while (!SnapShotStack.empty()) {
        //cout << SnapShotStack.size() << endl;
        vtype u=SnapShotStack.top();

        if (u == t) {
            retVal = SnapShots[u].second;
            SnapShotStack.pop();
            continue;
        }
        Edge<vtype,itype> &e = adj[u][start[u]];
        double flow = SnapShots[u].second;
        switch (SnapShots[u].first)
        {
        case 0:
            //cout << "a" << endl;
            //cout << u << " " << start[u] << " " << adj[u].size() << " " << e.v << endl;
            SnapShots[u].first = 1;
            //SnapShotStack.push(u);
            if (level[e.v] == level[u]+1 && e.flow < e.C) {
                // find minimum flow from u to t
                double curr_flow = min(flow, e.C - e.flow);
                SnapShots[e.v].first = 0;
                SnapShots[e.v].second = curr_flow;
                //SnapShot newSnapShot = SnapShot(e.v,curr_flow);
                SnapShotStack.push(e.v);
            }
            break;
        case 1:
            //cout << "b" << endl;
            if (retVal > 0) {
                // add flow to current edge
                e.flow += retVal;
                // subtract flow from reverse edge
                // of current edge
                adj[e.v][e.rev].flow -= retVal;
            }

            if (retVal <= 0 && (start[u]+1) < adj[u].size()) {
                start[u] ++;
                //cout << u << " " << start[u] << " " << adj[u].size() << endl;
                Edge<vtype,itype> &new_e = adj[u][start[u]];
                //SnapShotStack.push(u);
                //SnapShotStack.push(currentPtr);
                if (level[new_e.v] == level[u]+1 && new_e.flow < new_e.C) {
                    double curr_flow = min(flow, new_e.C - new_e.flow);
                    SnapShots[new_e.v].first = 0;
                    SnapShots[new_e.v].second = curr_flow;
                    SnapShotStack.push(new_e.v);
                    //SnapShot newSnapShot = SnapShot(new_e.v,curr_flow);
                    //SnapShotStack.push(&newSnapShot);
                }
            }
            else {
                SnapShotStack.pop();
            }
            break;
        }
    }

    return retVal;
}


// template<typename vtype, typename itype>
// double graph<vtype,itype>::sendFlow(vtype u, double flow, vtype t, vector<vtype>& start)
// {
//     //cout << u << " " << start[u] << endl;
//     // Sink reached
//     if (u == t)
//         return flow;
 
//     // Traverse all adjacent edges one -by - one.
//     for (  ; start[u] < adj[u].size(); start[u]++) {
//         // Pick next edge from adjacency list of u
//         Edge<vtype,itype> &e = adj[u][start[u]]; 
                                     
//         if (level[e.v] == level[u]+1 && e.flow < e.C) {
//             // find minimum flow from u to t
//             double curr_flow = min(flow, e.C - e.flow);
 
//             double temp_flow = sendFlow(e.v, curr_flow, t, start);
 
//             // flow is greater than zero
//             if (temp_flow > 0) {
//                 // add flow  to current edge
//                 e.flow += temp_flow;
 
//                 // subtract flow from reverse edge
//                 // of current edge
//                 adj[e.v][e.rev].flow -= temp_flow;
//                 //cout << u << " " << temp_flow << endl;
//                 return temp_flow;
//             }
//         }
//     }
 
//     return 0;
// }
 

template<typename vtype, typename itype>
void graph<vtype,itype>::find_cut(vtype u_init, vector<bool>& mincut, vtype& length)
{
    stack <vtype> stk;
    stk.push(u_init);
    while (!stk.empty()) {
        vtype u = stk.top();
        //cout << u << " " << stk.size() << endl;
        stk.pop();
        if (mincut[u] == true) {
            continue;
        }
        mincut[u] = true;
        length ++;
        for (int i = 0 ; i < adj[u].size(); i ++) {
            int k = adj[u].size() - 1 - i;
            //cout << k << " " << adj[u].size() << endl;
            Edge<vtype,itype> e = adj[u][k];
            if (e.flow < e.C && mincut[e.v] == false) {
                stk.push(e.v);
            }
        }
    }
}

/*
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
*/


// Returns maximum flow in graph
// s: source node
// t: taget node
// V: total number of nodes
// mincut: store the result
template<typename vtype, typename itype>
pair<double,vtype> graph<vtype,itype>::DinicMaxflow(vtype s, vtype t, vtype V, vector<bool>& mincut)
{
    for(int i = 0; i < V; i ++){
        for(int j = 0; j < adj[i].size(); j ++){
            Edge<vtype,itype> &e = adj[i][j]; 
            //cout << i << " " << e.v << " " << e.C << " " << e.rev << " " << adj[e.v][e.rev].v << " " << adj[e.v][e.rev].C << " " << adj[e.v][e.rev].rev << endl;
        }
    }
    // Corner case
    if (s == t)
        return make_pair(-1,0);
 
    double total = 0;  // Initialize result
 
    // Augment the flow while there is path
    // from source to sink
    vector<vtype> start(V+1);
    vector<pair<int,double>> SnapShots(V+1);

    while (BFS(s, t, V) == true){
        // store how many edges are visited
        // from V { 0 to V }
        fill(start.begin(),start.end(),0);
        // while flow is not zero in graph from S to D
        double flow = sendFlow(s, numeric_limits<double>::max(), t, start, SnapShots);
        while (flow > 0) {
        	// Add path flow to overall flow
            total += flow;
            flow = sendFlow(s, numeric_limits<double>::max(), t, start, SnapShots);
        }
        //cout << "curr flow: " << total << endl;
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
