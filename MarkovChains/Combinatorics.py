import numpy as np
import math
from scipy import linalg
from decimal import *

def make_vector_0(m,k):
    vector = []
    getcontext().prec += 60
    for d in range(1, m + 1 ):
        #vector.append( ( math.factorial ( 2 * d * (m - k + 1) ) ) / ( math.factorial( d * (m - k + 1)) **2   ))
        vector.append( Decimal(math.factorial(2 * d * k)) / (math.factorial(d * k) ** 2) )
    return vector


def make_vector_1(m,k):
    vector = []
    for d in range(m + 3, 2*m +3 ):
        vector.append( Decimal( math.factorial ( 2 * d * k ) ) / ( math.factorial( d * k) * math.factorial(d* k - m)))
    return vector

def make_vector_2(m,k):
    vector = []
    for d in range(3, m + 3):
        vector.append( ( 4 ** ( d * k)))
    return vector


def make_matrix_2(m,p):

    vectors = []

    for k in range(1, m+1):
        vectors.append( make_vector_2(m,k))

    return vectors

def make_matrix_0(m,p):

    vectors = []

    for k in range(1, m+1):
        vectors.append( make_vector_0(m,k))

    return vectors

def make_matrix_1(m,p):

    vectors = []

    for k in range(1 , m - 2):
        vectors.append( make_vector_1(m,k))

    return vectors

def choose(n,p):

    return float( Decimal( math.factorial(n) )/ ( Decimal(math.factorial(p)) * Decimal(math.factorial(n - p))))

def make_full_vector(r,k, D):
    vector = []
    for d in D:
        vector.append( choose(2 * d * k , d * k + r))
    return vector

def make_full_matrix(R, K):
    matrix = []

    D = list( range(R, 20*R* K))
    #??? Why does changing from 20 to 4 *increase* the rank. This makes no sense, and makes me dubious ...
    for k in range(1,K+1):
        for r in range(R):
            matrix.append( make_full_vector( r,k,D))


    return matrix

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4, linewidth=1000)
M = make_full_matrix(3,3)

print(len(M))
print(np.linalg.matrix_rank(M))
#print(np.matrix(M))


def clear_down(A):
    getcontext().prec += 60
    m = len(A)

    for i in range(m):

        a = A[i][i]
        for k in range(i+1, m):
            q = A[i][k]/a
            for s in range(m):
                A[s][k] = A[s][k] - q * A[s][i]

    for i in range(m):
        for j in range(m):
            A[i][j] = float(A[i][j])
    return A

def check_clear_down_conjecture(A,B):
    m = len(A)
    for i in range(m):
        if A[i][i] > 2 * B[i][i]:
            print(i)
            print( A[i][i], 2 * B[i][i])
            return False
    return True






np.set_printoptions(suppress=True)
np.set_printoptions(precision=2, linewidth=1000)
for k in range(20,40):
    p = 5
    A = make_matrix_1(k, p )
    print( "k: ", k)
    #print(np.matrix(A))
    B = clear_down(A)
    print( check_clear_down_conjecture(A,B))

    m = len(A)

    for i in range(m):
        for j in range(m):
            B[i][j] = float(B[i][j])

    #print( np.matrix(B))
    #print ( linalg.lu(A)[1])
    #print ( linalg.lu(A)[2])
    #print(np.linalg.eigvals(A))

    #print( (np.linalg.det(A)))

