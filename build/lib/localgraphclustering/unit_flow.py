import Queue as queue

def push(A,f,f_v,v,u,l,ex,U,n,w,degree):
    
    pushed = 0
    
    if v < u:
        idx = v*n - v*(v+1)/2 + (n-1 - (n-u))-v
        same_dir = 1
    else:
        idx = u*n - u*(u+1)/2 + (n-1 - (n-v))-u
        same_dir = -1
    
    if not(f.has_key(idx)):
        f.update({idx:0})
    
    r = min(l[v],U) - same_dir*f[idx]
    
    if (r > 0) & (l[v] > l[u]):
        if not(f_v.has_key(u)):
            f_v.update({u:0}) 
        if not(degree.has_key(u)):
            degree.update({u:A[u,:].sum(axis=1)[0,0]})
        degree_val = degree[u]
        psi = min(ex[v],r,w*degree_val - f_v[u])
        f[idx] += same_dir*psi
        f_v[v] -= psi
        f_v[u] += psi
        pushed = 1
        
    return pushed

def relabel(v,l):
    l[v] += 1
    
def push_relabel(A,f,f_v,U,v,current_v,ex,l,n,w,degree):
    
    neighbors = (current_v[v])[2]
    index = (current_v[v])[0]
    num_neigh = (current_v[v])[1]
    u = neighbors[index]
    if not(l.has_key(u)):
        l.update({u:0})
        
    pushed = push(A,f,f_v,v,u,l,ex,U,n,w,degree)
    
    relabelled = 0
    
    if 1-pushed:
        if index < num_neigh-1:
            current_v[v] = [index+1,num_neigh,neighbors]
        else:
            relabel(v,l)
            relabelled = 1
            current_v[v] = [0,num_neigh,neighbors]
            
    return pushed,relabelled,u 
            
def update_excess(A,f_v,v,ex,degree):
    if not(degree.has_key(v)):
        degree.update({v:A[v,:].sum(axis=1)[0,0]})
    degree_val = degree[v]
    
    ex_ = max(f_v[v] - degree_val,0)
    if (not(ex.has_key(v))) and (ex_ == 0):
        return
    ex.update({v:ex_})
    
    
def add_in_Q(v,l,Q,A,current_v):
    Q.put([l[v],v])
    neighbors = A[v,:].nonzero()[1]
    current_v[v] = [0,len(neighbors),neighbors]
    
def remove_from_Q(v,Q):
    Q.get()
    
def shift_from_Q(v,l,Q): 
    Q.get()
    Q.put([l[v],v])
    
def unit_flow(A, Delta, U, h, w, degree):
    
    # Assumption on edge directions: based on two loops to read A.
    # Outer loop is rows and inner loop is columns. The variables for the algorithm
    # correspond to edges based on the direction imposed by reading A this way.
    
    # Dimensions
    n = A.shape[0]
    N = n*(n-1)/2
    
    # Variables and parameters
    f = {}
    l = {}
    ex = {}
    f_v = {}
    current_v = {}
    
    Q = queue.PriorityQueue()
    
    for i in Delta:
        f_v.update({i:Delta[i]})
        l[i]=0
        if not(degree.has_key(i)):
            degree.update({i:A[i,:].sum(axis=1)[0,0]})
        degree_val = degree[i]
        if Delta[i] > degree_val:
            l[i]=1
            Q.put([1,i])
            ex.update({i:Delta[i] - degree_val})
            neighbors = A[i,:].nonzero()[1]
            current_v.update({i: [0,len(neighbors),neighbors]})
    
    while Q.qsize() > 0:
        
        v = (Q.queue[0])[1]
            
        pushed,relabelled,u = push_relabel(A,f,f_v,U,v,current_v,ex,l,n,w,degree)
        
        if pushed:
            update_excess(A,f_v,u,ex,degree)
            update_excess(A,f_v,v,ex,degree)
                
            if ex[v] == 0:
                remove_from_Q(v,Q)
            if (ex.has_key(u)) and (ex[u] > 0):
                add_in_Q(u,l,Q,A,current_v)
                
        if relabelled:
            if l[v] < h:
                shift_from_Q(v,l,Q)
            else:
                remove_from_Q(v,Q)  
               
    return l,f_v,ex