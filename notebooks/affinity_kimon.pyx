#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.math cimport abs
from libc.math cimport sqrt

def affinity(int n, int p, int q, double r, double var_I, double var_X, double threshold, double[:,:,:] sub_img):
    
    cdef int i, j, k, l
    cdef double wkl, d, nF, temp1, temp2, temp3

    cdef list row = [], col = [], data = []
    
    cdef double[:,:] F
    cdef double[:,:] X
    
    F = np.zeros([n, 3])  
    X = np.zeros([n, 2])
    
    for i in range(n):
        F[i,0] = 0
        F[i,1] = 0
        F[i,2] = 0
        X[i,0] = 0
        X[i,1] = 0
    
    # RGB values
    for i in range(p):
        for j in range(q):
            k = i*q + j
            F[k,0] = sub_img[i, j, 0]
            F[k,1] = sub_img[i, j, 1]
            F[k,2] = sub_img[i, j, 2]
            X[k,0] = i
            X[k,1] = j

    for k in range(n):
        #if k%100 == 0:
            #print k
        for l in range(k+1, n):
            temp1 = abs(X[k,0] - X[l,0])
            temp1 = temp1*temp1
            temp2 = abs(X[k,1] - X[l,1])
            temp2 = temp2*temp2
            d = temp1 + temp2
            if d < r:
                temp1 = abs(F[k,0] - F[l,0])
                temp1 = temp1*temp1
                temp2 = abs(F[k,1] - F[l,1])
                temp2 = temp2*temp2
                temp3 = abs(F[k,2] - F[l,2])
                temp3 = temp3*temp3
                nF = temp1 + temp2 + temp3
                wkl = exp(-nF/var_I - d/var_X)
                if wkl > threshold:
                    row.extend([k, l])
                    col.extend([l, k])
                    data.extend([wkl, wkl])
    return data, row, col

def affinity_gray(int n, int p, int q, double r, double var_I, double var_X, double threshold, double[:,:] sub_img):

    cdef int i, j, k, l
    cdef double wkl, d, nF, temp1

    cdef list row = [], col = [], data = []

    cdef double[:] F
    cdef double[:,:] X

    F = np.zeros(n)
    X = np.zeros([n, 2])

    for i in range(n):
        F[i] = 0
        X[i,0] = 0
        X[i,1] = 0

    # RGB values
    for i in range(p):
        for j in range(q):
            k = i*q + j
            F[k] = sub_img[i, j]
            X[k,0] = i
            X[k,1] = j

    for k in range(n):
        #if k%100 == 0:
            #print k
        for l in range(k+1, n):
            temp1 = abs(X[k,0] - X[l,0])
            temp1 = temp1*temp1
            temp2 = abs(X[k,1] - X[l,1])
            temp2 = temp2*temp2
            d = temp1 + temp2
            if d < r:
                temp1 = abs(F[k] - F[l])
                nF = temp1*temp1
                wkl = exp(-nF/var_I - d/var_X)
                if wkl > threshold:
                    row.extend([k, l])
                    col.extend([l, k])
                    data.extend([wkl, wkl])
    return data, row, col

def affinity_multiscale(int n, int p, int q, double r, double var_I, double var_X, double threshold, double[:,:,:] sub_img):

    cdef int i, j, k, l
    cdef double wkl, d, nF, temp1, temp2, temp3

    cdef list row = [], col = [], data = []

    cdef double[:,:] F
    cdef double[:,:] X

    F = np.zeros([n, 8])
    X = np.zeros([n, 2])

    for i in range(n):
        F[i,0] = 0
        F[i,1] = 0
        F[i,2] = 0
        F[i,3] = 0
        F[i,4] = 0
        F[i,5] = 0
        F[i,6] = 0
        F[i,7] = 0
        X[i,0] = 0
        X[i,1] = 0

    # RGB values
    for i in range(p):
        for j in range(q):
            k = i*q + j
            F[k,0] = sub_img[i, j, 0]
            F[k,1] = sub_img[i, j, 1]
            F[k,2] = sub_img[i, j, 2]
            F[k,3] = sub_img[i, j, 3]
            F[k,4] = sub_img[i, j, 4]
            F[k,5] = sub_img[i, j, 5]
            F[k,6] = sub_img[i, j, 6]
            F[k,7] = sub_img[i, j, 7]
            X[k,0] = i
            X[k,1] = j

    for k in range(n):
        #if k%100 == 0:
        #print k
        for l in range(k+1, n):
            temp1 = abs(X[k,0] - X[l,0])
            temp1 = temp1*temp1
            temp2 = abs(X[k,1] - X[l,1])
            temp2 = temp2*temp2
            d = temp1 + temp2
            if d < r:
                temp1 = abs(F[k,0] - F[l,0])
                temp1 = temp1*temp1
                temp2 = abs(F[k,1] - F[l,1])
                temp2 = temp2*temp2
                temp3 = abs(F[k,2] - F[l,2])
                temp3 = temp3*temp3
                temp4 = abs(F[k,3] - F[l,3])
                temp4 = temp4*temp4
                temp5 = abs(F[k,4] - F[l,4])
                temp5 = temp5*temp5
                temp6 = abs(F[k,5] - F[l,5])
                temp6 = temp6*temp6
                temp7 = abs(F[k,6] - F[l,6])
                temp7 = temp7*temp7
                temp8 = abs(F[k,7] - F[l,7])
                temp8 = temp8*temp8
                nF = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8
                wkl = exp(-nF/var_I - d/var_X)
                if wkl > threshold:
                    row.extend([k, l])
                    col.extend([l, k])
                    data.extend([wkl, wkl])
    return data, row, col
