import numpy as np
import networkx as nx


for size in range(50,70):
    grid = nx.grid_graph ( [size, size])
    Adj = nx.to_numpy_array(grid)

    z = .2

    eigen = np.linalg.eigvals(Adj)

    print (np.max(eigen))

    print("compute right and left eigenvectors")
    #This is the stationary distribution, and the 1 vector.
    #So the weight here is the stationary probability.
    #This doesn't make sense, as it's not giving something symmetric. No -- its not the transition matrix
    #It's not proportional to it either.


    A = np.kron(z, Adj)

    T = (np.identity(size**2) - A)


#X = np.linalg.inv(T)

#down = X[0][size]