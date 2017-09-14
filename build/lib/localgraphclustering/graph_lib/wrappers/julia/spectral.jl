# for issymmetric and checksquare 
using Compat 
import Compat.LinAlg.checksquare # updates for v0.5

## Todo
# 1. Add method for partial sweep cut
# 2. Add a "SymmetricMatrix" wrapper to avoid
#    constant symmetry checks.
# 3. Avoid creating the Laplacian matrix directly

"""
`_symeigs_smallest_arpack`
---
Compute eigenvalues and vectors using direct calls to the ARPACK wrappers
to get type-stability. This function works for symmetric matrices.
This function works on Float32 and Float64-valued sparse matrices.
It returns the smallest set of eigenvalues and vectors.
It only works on matrices with more than 21 rows and columns.
(Use a dense eigensolver for smaller problems.)
Functions
---------
- `(evals,evecs) = _symeigs_smallest_arpack(A::SparseMatrixCSC{V,Int},
                        nev::Int,tol::V,maxiter::Int, v0::Vector{V})`
Inputs
------
- `A`: the sparse matrix, must be symmetric
- `nev`: the number of eigenvectors requested
- `tol`: the relative tolerance of the eigenvalue computation
- `maxiter`: the maximum number of restarts
- `v0`: the initial vector for the Lanczos process
Example
-------
This is an internal function.
"""

function _symeigs_smallest_arpack{V}(
            A::SparseMatrixCSC{V,Int},nev::Int,tol::V,maxiter::Int,
            v0::Vector{V})

    n::Int = checksquare(A) # get the size
    @assert n >= 21

    # setup options
    mode = 1
    sym = true
    iscmplx = false
    bmat = Compat.String("I") # ByteString on Julia 0.4, String on 0.5
    ncv = min(max(2*nev,20),n-1)

    whichstr = Compat.String("SA") # ByteString on Julia 0.4, String on 0.5
    ritzvec = true
    sigma = 0.

    TOL = Array(V,1)
    TOL[1] = tol
    lworkl = ncv*(ncv + 8)
    v = Array(V, n, ncv)
    workd = Array(V, 3*n)
    workl = Array(V, lworkl)
    resid = Array(V, n)

    resid[:] = v0[:]

    info = zeros(Base.LinAlg.BlasInt, 1)
    info[1] = 1

    iparam = zeros(Base.LinAlg.BlasInt, 11)
    ipntr = zeros(Base.LinAlg.BlasInt, 11)
    ido = zeros(Base.LinAlg.BlasInt, 1)

    iparam[1] = Base.LinAlg.BlasInt(1)
    iparam[3] = Base.LinAlg.BlasInt(maxiter)
    iparam[7] = Base.LinAlg.BlasInt(mode)

    # this is a helpful indexing vector
    zernm1 = 0:(n-1)

    while true
        # This is the reverse communication step that ARPACK does
        # we need to extract the desired vector and multiply it by A
        # unless the code says to stop
        Base.LinAlg.ARPACK.saupd(
            ido, bmat, n, whichstr, nev, TOL, resid, ncv, v, n,
            iparam, ipntr, workd, workl, lworkl, info)

        load_idx = ipntr[1] + zernm1
        store_idx = ipntr[2] + zernm1

        x = workd[load_idx]

        if ido[1] == 1
            workd[store_idx] = A*x
        elseif ido[1] == 99
            break
        else
            error("unexpected ARPACK behavior")
        end
    end

    # Once the process terminates, we need to extract the
    # eigenvectors.

    # calls to eupd
    howmny = Compat.String("A") # ByteString on Julia 0.4, String on 0.5
    select = Array(Base.LinAlg.BlasInt, ncv)

    d = Array(V, nev)
    sigmar = ones(V,1)*sigma
    ldv = n
    Base.LinAlg.ARPACK.seupd(ritzvec, howmny, select, d, v, ldv, sigmar,
        bmat, n, whichstr, nev, TOL, resid, ncv, v, ldv,
        iparam, ipntr, workd, workl, lworkl, info)
    if info[1] != 0
        error("unexpected ARPACK exception")
    end

    # Now we want to return them in sorted order (smallest first)
    p = sortperm(d)

    d = d[p]
    vectors = v[1:n,p]

    return (d,vectors,iparam[5])
end

"""
`fiedler_vector`
----------------
Compute the Fiedler vector associated with the normalized Laplacian
of the graph with adjacency matrix A.
This function works with Float32 and Float64-valued sparse
matrices and vectors.
It requires a symmetric input matrix representing
the adjacency matrix of an undirected graph. If the input
is a disconnected network then the result may not
The return vector is signed so that the number of positive
entries is at least the number of negative entries. This will
always give a unique, deterministic output
in the case of a non-repeated second eigenvalue.
Functions
---------
- `(v,lam2) = fiedler_vector(A::SparseMatrixCSC{V,Int})`
- `(v,lam2) = fiedler_vector(A::MatrixNetwork{V,Int})`
Inputs
------
- `A`: the adjacency matrix
- Optional Keywoard Inputs
  - `checksym`: ensure the matrix is symmetric
  - `tol`: the residual tolerance in the eigenvalue, eigenvector estimate.
    (This is an absolute error)
  - `maxiter`: the maximum iteration for ARPACK
  - `nev`: the number of eigenvectors estimated by ARPACK
  - `dense`: the threshold for a dense (LAPACK) computation
    instead of a sparse (ARPACK) computation. If the size of
    the matrix is less than `dense` then maxiter and nev are
    ignored and LAPACK computes all the eigenvalues.
Returns
-------
- `v`: The Fiedler vector as an array
- `lam2`: the eigenvalue itself
Example
-------
~~~
# construct an n-node path graph
n = 25
A = sparse(1:n-1,2:n,1.,n,n)
A = A + A'
(v,lam2) = fiedler_vector(A) # returns a cosine-like fiedler vector
# using UnicodePlots; lineplot(v)
~~~
"""
function fiedler_vector{V}(A::SparseMatrixCSC{V,Int};
    tol=1e-12,maxiter=300,dense=96,nev=2,checksym=true)
    n = checksquare(A)
    if checksym
        if !issymmetric(A)
            throw(ArgumentError("The input matrix must be symmetric."))
        end
    end

    d = vec(sum(A,1))
    d = sqrt(d)

    if n == 1
        X = zeros(V,1,2)
        lam2 = 0.
    elseif n <= dense
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = full(L) + 2*eye(n)
        F = eigfact!(Symmetric(L))
        lam2 = F.values[2]-1.
        X = F.vectors
    else
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = L + 2.*speye(n)

        (lams,X,nconv) = _symeigs_smallest_arpack(L,nev,tol,maxiter,d)
        lam2 = lams[2]-1.
    end
    x1err = norm(X[:,1]*sign(X[1,1]) - d/norm(d))
    if x1err >= sqrt(tol)
        warn(@sprintf("""
        the null-space vector associated with the normalized Laplacian
        was computed inaccurately (diff=%.3e); the Fiedler vector is
        probably wrong or the graph is disconnected""",x1err))
    end

    x = vec(X[:,2])
    if n > 1
        x = x./d # applied sqrt above
    end

    # flip the sign if the number of pos. entries is less than the num of neg. entries
    nneg = sum(x .< 0.)
    if n-nneg < nneg
      x = -x;
    end

    return (x,lam2)
end

fiedler_vector{T}(A::MatrixNetwork{T};kwargs...) =
    fiedler_vector(sparse_transpose(A);kwargs...) # sparse_transpose converts directly

"""
RankedArray
-----------
This is a data-type that functions as a union between
a sparse and dense array. It allows us to check if
an element is not in the array (i.e. it's too long)
by adding a getindex function to the array type.
It's called a RankedArray because the idea is that
the elements of the array are the ranks of the
nodes in a sorted vector.
This should not be used externally.
Example
-------
v = rand(10)
r = RankedArray(sortperm(v))
haskey(r,11) # returns false
haskey(r,1) # returns true
haskey(r,0) # returns false
"""

type RankedArray{S}
    data::S
end

import Base: getindex, haskey

haskey{S}(A::RankedArray{S}, x::Int) = x >= 1 && x <= length(A.data)
getindex{S}(A::RankedArray{S}, i) = A.data[i]

"""
SweepcutProfile
---------------
This type is the result of a sweepcut operation. It stores
a number of vectors associated with a sweep cut including
the cut and volume associated with each.
Methods
-------
- `bestset(P::SweepcutProfile)` return the best set identified
  in the sweepcut. This is usually what you want.
Example
-------
See `sweepcut`
See also
--------
* `spectral_cut`
"""

immutable SweepcutProfile{V,F}
    p::Vector{Int}
    conductance::Vector{F}
    cut::Vector{V}
    volume::Vector{V}
    total_volume::V
    total_nodes::Int

    function SweepcutProfile(p::Vector{Int},nnodes::Int,totalvol::V)
        n = length(p)
        new(p,Array(F,n-1),Array(V,n-1),Array(V,n-1),totalvol,nnodes)
    end
end

"""
`sweepcut`
--------
A sweepcut is an operation that takes an order to
the vertices of a graph and evaluates the cut and
volume of every partition induced by a prefix of
that ordering. That is, if the order of vertices
is
    v1, v2, v3, ...
the the sweep cut evaluates
    cut({v1}), vol({v1})
    cut({v1,v2}), vol({v1,v2})
    cut({v1,v2,v3}), vol({v1,v2,v3})
    ...
where cut is the total edge weight that connects
vertices in S to the graph. And vol is the
total edge weight connecting
Functions
---------
- `profile = sweepcut(A,x::Vector{T})`
  A standard sweepcut call that will get a sweepcut
  for a vector where x is sorted to get the order
  of the vertices (decreasing order).
  This is useful if you have a vector that should
  indicate good cuts in the graph, such as
  the Fiedler vector.
- `profile = sweepcut(A,p,r,totalvol,maxvol)`
  the in-depth call that all others are converted into.
  We strongly recommend against calling this yourself
  unless you understand the sweepcut code.
Inputs
------
- `A`: the sparse matrix representing the symmetric graph
- `x`: A vector scoring each vertex. This will be sorted and
       turned into one of the other inputs.
- `p`: a partial permutation vector over the vertices of the graph
        This vector needs to list every vertex at most once.
        It could be shorter and need not list every vertex.
- `r`: A general data structure that gives
        the rank of an item in the permutation vector
        p should be sorted in decreasing order so that
        i < j means that r[p[i]] < r[p[j]]
- `totalvol`: the total volume of the graph
- `maxvol`: the maximum volume of any set considered
Returns
-------
A `SweepcutProfile` type with all the information computed
in the sweepcut. Most likely, you want the result `bestset`
as indicated below.
Example
-------
~~~~
A = load_matrix_network("minnesota")
v = fiedler_vector(A)[1] # get the
p = sweepcut(A,v)
S = bestset(p) # get the bestset from the profile
T = spectral_cut(A).set # should give you the same set
# using UnicodePlots; lineplot(p.conductance) # show the conductance
~~~~
"""
function sweepcut{V,T}(A::SparseMatrixCSC{V,Int}, p::Vector{Int}, r, totalvol::V, maxvol::T)

    n = checksquare(A)
    nlast = length(p)

    if n < nlast
        throw(DimensionMismatch(
            "the permutation vector is too long and should have fewer entries than the graph"))
    end

    F = typeof(one(V)/one(V)) # find the conductance type
    output = SweepcutProfile{V,F}(p,n,totalvol)

    cut = zero(V)
    vol = zero(V)
    colptr = A.colptr
    rowval = rowvals(A)
    nzval = A.nzval

    for (i,v) in enumerate(p)
        deltain = zero(V) # V might be pos. only... so delay subtraction
        deg = zero(V)
        rankv = getindex(r,v)
        for nzi in colptr[v]:(colptr[v+1] - 1)
            nbr = rowval[nzi]
            deg += nzval[nzi]
            if haskey(r,nbr) # our neighbor is ranked
                if getindex(r,nbr) <= rankv # nbr is ranked lower, decrease cut
                    deltain += v == nbr ? nzval[nzi] : 2*nzval[nzi]
                end
            end
        end
        cut += deg
        cut -= deltain
        vol += deg

        # don't assign final values because they are unhelpful
        if i==nlast
            break
        end

        cond = cut/min(vol,totalvol-vol)
        output.conductance[i] = cond
        output.cut[i] = cut
        output.volume[i] = vol
    end

    if nlast == size(A,1)
        @assert abs(cut) <= 1e-12*totalvol
    end
    return output
end

function sweepcut{V,T}(A::SparseMatrixCSC{V,Int}, x::Vector{T}, vol::V)
    p = sortperm(x,rev=true)
    ranks = Array(Int, length(x))
    for (i,v) in enumerate(p)
        ranks[v] = i
    end
    r = RankedArray(ranks)
    return sweepcut(A, p, r, vol, Inf)
end

sweepcut{V,T}(A::SparseMatrixCSC{V,Int}, x::Vector{T}) =
    sweepcut(A, x, sum(A))

function bestset{V,F}(prof::SweepcutProfile{V,F})
    nnodes = 0
    if !isempty(prof.conductance)
        bsetind = indmin(prof.conductance)
        bsetvol = prof.volume[bsetind]
        nnodes = bsetind
        if bsetvol > prof.total_volume - bsetvol
            # we want the complement set
            nnodes = prof.total_nodes - bsetind
        end
    end

    bset = Vector{Int}(nnodes)
    if isempty(prof.conductance)
    elseif bsetvol > prof.total_volume - bsetvol
        # ick, we need the complement
        bset[:] = setdiff(IntSet(Int(1):Int(prof.total_nodes)),prof.p[1:bsetind])
    else
        # easy
        bset[:] = prof.p[1:bsetind]
    end
    return bset
end

"""
`SpectralCut`
-------------
The return type from the `spectral_cut`
Fields
------
- `set`: the small side of the spectral cut
- `A`: the network of the largest strong component of the network
- `lam2`: the eigenvalue of the normalized Laplacian
- `x`: the Fiedler vector for spectral partitioning
- `sweepcut_profile`: the sweepcut profile output
- `comps`: the output from the strong_components function
    check comps.number to get the number of components
- `largest_component`: the index of the largest strong component
The most useful outputs are `set` and `lam2`; the others are provided
for experts who wish to use some of the diagonstics provided.
"""

immutable SpectralCut{V,F}
    set::Vector{Int}
    A::SparseMatrixCSC{V,Int}
    lam2::F
    x::Vector{Float64}
    sweepcut_profile::SweepcutProfile{V,F}
    comps::MatrixNetworks.Strong_components_output
    largest_component::Int
end

function show{V,F}(io::IO,obj::SpectralCut{V,F})
    println(io,"Spectral cut on adjacency matrix with $(size(obj.A,1)) nodes and $(nnz(obj.A)) non-zeros")
    if obj.comps.number > 1
        println(io, "    formed from the largest connected component of the input matrix")
    end
    println(io,"  conductance = $(minimum(obj.sweepcut_profile.conductance))")
    println(io,"  size = $(length(obj.set))")
end

"""
`spectral_cut`
--------------
`spectral_cut` will produce a spectral cut partition of the graph into
two pieces.
Special cases
* if your graph is disconnected, then we will partition it into the
largest connected component (chosen arbitrary if there are multiple)
and produce a cut of just the largest connected component.
* if your graph is a single node, we will partition it into the empty
cut.
Output
------
The output has type SpectralCut
We always return the smaller side of the cut in terms of total volume.
Inputs
------
- `A`: The sparse matrix or martrix network that you want to
  find the spectral cut for.
- `checksym`: Should we check symmetry?
  Don't set this to false unless you have checked symmetry
  in another call. *This may go away in future interfaces*
- `ccwarn`: Turn off the warning for disconnected graphs.
  This useful in larger subroutines where this is handled
  through another mechanism.
"""

function spectral_cut{V}(A::SparseMatrixCSC{V,Int},checksym::Bool,ccwarn::Bool)
    n = checksquare(A)
    if checksym
        if !issymmetric(A)
            throw(ArgumentError("The input matrix must be symmetric."))
        end
    end

    # test for non-negativity
    if !all(nonzeros(A) .>= 0)
        throw(ArgumentError("The input matrix must be non-negative"))
    end

    # need to test for components
    G = MatrixNetwork(n,A.colptr,A.rowval,A.nzval)
    cc = scomponents(G)
    lccind = indmax(cc.sizes)
    B = A
    if cc.number > 1
        if ccwarn
            warn("The input matrix had $(cc.number) components, running on largest...")
        end
        f = cc.map .== lccind
        B = A[f,f] # using logical indexing is faster than find for large sets (2015/11/14)
        # n=10^5;A=sprand(n,n,25/n);f=rand(n).>0.01;s=find(f);@time sum(A);@time A[f,f]; @time A[s,s];
        subset = find(f)
    end

    # run the partition
    x,lam2 = fiedler_vector(B;checksym=false)

    totalvol = sum(B)
    sweep = sweepcut(B,x,totalvol)
    bset = bestset(sweep)

    if cc.number > 1
        # map the set back to the original index set
        bset = subset[bset]
    end

    return SpectralCut(bset,B,lam2,x,sweep,cc,lccind)
end

function spectral_cut{V}(A::SparseMatrixCSC{V,Int})
    return spectral_cut(A,true,true)
end

function spectral_cut{V}(A::MatrixNetwork{V})
    return spectral_cut(sparse_transpose(A),true,true)
end
