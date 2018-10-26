#ifndef CRD_H
#define CRD_H

#include <utility>
#include <unordered_map>
#include <tuple>
#include <queue>

using namespace std;

template<typename vtype, typename itype>
class my_cmp
{
public:
	bool operator()(pair<vtype,vtype> p1, pair<vtype,vtype> p2) {
		if (p1.first != p2.first) {
			return p1.first > p2.first;
		}
		else {
			return p1.second >= p2.second;
		}
		
	}
};

template<typename vtype, typename itype>
void reset(unordered_map<vtype,double>& f_v, unordered_map<vtype,double>& ex, 
	unordered_map<vtype,double>& cond_temp, unordered_map<vtype,vtype>& l, 
	unordered_map<vtype,vector<vtype>>& labels_temp);

template<typename vtype, typename itype>
void mat_vec_dot(itype* ai, vtype* aj, vtype n, unordered_map<vtype,double>& vec, 
	unordered_map<vtype,double>& result);

template<typename vtype, typename itype>
double vec_vec_dot(unordered_map<vtype,double>& vec1, unordered_map<vtype,double>& vec2);

template<typename vtype, typename itype>
void vec_vec_add(unordered_map<vtype,double>& vec1, unordered_map<vtype,double>& vec2);

template<typename vtype, typename itype>
bool push(itype* ai, vtype* aj, unordered_map<vtype,double>& f, 
	unordered_map<vtype,double>& f_v, vtype v, vtype u, unordered_map<vtype,vtype>& l,
	unordered_map<vtype,vtype>& ex, vtype U, vtype n, vtype w);

template<typename vtype, typename itype>
void relabel(unordered_map<vtype,vtype>& l, vtype v);

template<typename vtype, typename itype>
tuple<bool,bool,vtype> push_relabel(itype* ai, vtype* aj, unordered_map<vtype,double>& f,
	unordered_map<vtype,double>& f_v, vtype U, vtype v, unordered_map<vtype,vtype>& current_v,
	unordered_map<vtype,double>& ex, unordered_map<vtype,vtype>& l, vtype n, vtype w);

template<typename vtype, typename itype>
void update_excess(itype* ai, vtype* aj, unordered_map<vtype,double>& f_v,
	vtype v, unordered_map<vtype,double>& ex);

template<typename vtype, typename itype>
void add_in_Q(vtype v, unordered_map<vtype,vtype>& l, 
	priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> &Q,
	unordered_map<vtype,vtype>& current_v);


template<typename vtype, typename itype>
void remove_from_Q(priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> &Q);


template<typename vtype, typename itype>
void shift_from_Q(vtype v, unordered_map<vtype,vtype>& l,
	priority_queue<pair<vtype,vtype>,vector<pair<vtype,vtype>>,my_cmp<vtype,itype>> &Q);


#include "../capacity_releasing_diffusion.cpp"
#include "../unit_flow.cpp"
#endif