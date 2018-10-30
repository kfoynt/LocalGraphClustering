#ifdef CRD_H

#include "include/routines.hpp"
#include <queue>
#include <tuple>
#include <algorithm>
#include <utility>

using namespace std;

template<typename vtype, typename itype>
bool push(itype* ai, vtype* aj, unordered_map<vtype,double>& f, 
    unordered_map<vtype,double>& f_v, vtype v, vtype u, unordered_map<vtype,vtype>& l,
    unordered_map<vtype,double>& ex, vtype U, vtype n, vtype w)
{
    bool pushed = false;
    vtype idx;
    int same_dir;
    if (v < u){
        idx = v*n - v*(v+1)/2 + (n-1-(n-u))-v;
        same_dir = 1;
    }
    else {
        idx = u*n - u*(u+1)/2 + (n-1-(n-v))-u;
        same_dir = -1;
    }
    if (f.count(idx) == 0) {
        f[idx] = 0;
    }
    /*
    if (1) {
        cout << "f is: ";
        print_map<vtype,double>(f);
        cout << "idx: " << idx << endl;
    }
    */
    double r = min(l[v],U) - same_dir*f[idx];
    if ((r > 0) && (l[v] > l[u])) {
        if (f_v.count(u) == 0) {
            f_v[u] = 0;
        }
        double degree_val = ai[u+1] - ai[u];
        double tmp[] = {ex[v],r,w*degree_val-f_v[u]};
        double psi = *min_element(tmp,tmp+3);
        f[idx] += same_dir*psi;
        //cout << "psi: " << psi << " f[idx]: " << f[idx] << " same_dir: " << same_dir << endl;
        f_v[v] -= psi;
        f_v[u] += psi;
        pushed = true;
    }
    return pushed;
}

template<typename vtype, typename itype>
void relabel(unordered_map<vtype,vtype>& l, vtype v)
{
    l[v] += 1;
}

template<typename vtype, typename itype>
tuple<bool,bool,vtype> push_relabel(itype* ai, vtype* aj, unordered_map<vtype,double>& f,
    unordered_map<vtype,double>& f_v, vtype U, vtype v, unordered_map<vtype,vtype>& current_v,
    unordered_map<vtype,double>& ex, unordered_map<vtype,vtype>& l, vtype n, vtype w)
{
    vtype index = current_v[v];
    //cout << "index: " << index << " v " << v << endl;
    vtype u = aj[ai[v]+current_v[v]];
    if (l.count(u) == 0) {
        l[u] = 0;
    }
    //cout << l.size() << endl;
    bool pushed = push<vtype,itype>(ai,aj,f,f_v,v,u,l,ex,U,n,w);
    bool relabelled = false;
    if (!pushed) {
        if (index < ai[v+1]-ai[v]-1) {
            current_v[v] = index+1;
        }
        else {
            relabel<vtype,itype>(l,v);
            relabelled = true;
            current_v[v] = 0;
        }
    }
    return make_tuple(pushed,relabelled,u);
}

template<typename vtype, typename itype>
void update_excess(itype* ai, vtype* aj, unordered_map<vtype,double>& f_v,
    vtype v, unordered_map<vtype,double>& ex)
{
    double degree_val = ai[v+1] - ai[v];
    double ex_ = max(f_v[v]-degree_val,0.0);
    if (ex.count(v) == 0 && ex_ == 0) {
        return;
    }
    ex[v] = ex_;
}


template<typename vtype, typename itype>
void add_in_Q(vtype v, unordered_map<vtype,vtype>& l, 
    priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> &Q,
    unordered_map<vtype,vtype>& current_v)
{
    Q.push(make_pair(l[v],v));
    current_v[v] = 0;
}


template<typename vtype, typename itype>
void remove_from_Q(priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> &Q)
{
    Q.pop();
}


template<typename vtype, typename itype>
void shift_from_Q(vtype v, unordered_map<vtype,vtype>& l,
    priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> &Q)
{
    Q.pop();
    Q.push(make_pair(l[v],v));
}

/*This is the implementation of crd inner process*/
template<typename vtype, typename itype>
void graph<vtype,itype>::unit_flow(unordered_map<vtype,double>& Delta, 
    vtype U, vtype h, vtype w, unordered_map<vtype,double>& f_v, 
    unordered_map<vtype,double>& ex, unordered_map<vtype,vtype>& l)
{
    priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> Q;
    Q = priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> ();
    unordered_map<vtype,vtype> current_v;
    unordered_map<vtype,double> f;
    current_v.clear();
    f.clear();
    for (auto iter = Delta.begin(); iter != Delta.end(); ++iter) {
        f_v[iter->first] = iter->second;
        l[iter->first] = 0;
        double degree_val = get_degree_unweighted(iter->first);
        //cout << "degree: " << degree_val << endl;
        if (iter->second > degree_val) {
            l[iter->first] = 1;
            Q.push(make_pair(1,iter->first));
            ex[iter->first] = iter->second - degree_val;
            current_v[iter->first] = 0;
        }
    }
    while (Q.size() > 0) {
        //cout << "iter: " << which_iter << " " << Q.size() << endl;
        
        vtype v = (Q.top()).second;
        tuple<bool,bool,vtype> tmp_result = push_relabel<vtype,itype>(ai,aj,f,f_v,U,v,
            current_v,ex,l,n,w);
        vtype u = get<2>(tmp_result);
        /*
        if (which_iter <= 154) {
            print_map<vtype,vtype>(l);
            print_map<vtype,double>(f_v);
            print_map<vtype,double>(ex);
            priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> Q1;
            Q1 = priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> ();
            cout << "v: " << v << " u: " << u << endl;
            int qsize = Q.size();
            for (int i = 0; i < qsize; i ++) {
                pair<vtype,vtype> tmp = Q.top();
                cout << "(" << tmp.first << "," << tmp.second << "),";
                Q.pop();
                Q1.push(tmp);
            }
            cout << endl;
            Q = Q1;
            cout << Q.size() << endl;
        }
        */
        /*if pushed*/
        if (get<0>(tmp_result)) {
            update_excess<vtype,itype>(ai,aj,f_v,u,ex);
            update_excess<vtype,itype>(ai,aj,f_v,v,ex);
            if (ex[v] == 0) {
                remove_from_Q<vtype,itype>(Q);
            }
            if (ex.count(u) > 0 && ex[u] > 0){
                add_in_Q<vtype,itype>(u,l,Q,current_v);
            }
        }
        /*if relabelled*/
        if (get<1>(tmp_result)) {
            if (l[v] < h) {
                /*since the label has been changed, we need to remove and reinsert to maintain priority queue*/
                shift_from_Q<vtype,itype>(v,l,Q);
            }
            else {
                /*new label is larger than threshold, just remove it*/
                remove_from_Q<vtype,itype>(Q);
            }
        }
    }
}

#endif