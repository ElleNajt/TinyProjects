import numpy as np
import random
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.optimize import linprog

class LPMatroid():
    def __init__(self, A, b):
        self.A = A
        self.num_variables = A.shape[1]
        self.num_equations = A.shape[0]
        if len(b) != self.num_equations:
            print ( "Dimensions do not agree")

        self.rank = np.linalg.matrix_rank(A)
        self.current_basis = self.pick_basis()
        self.b = b
        self.basic_feasible_solution = 0
        self.tries = 1000
        self.estimated_relaxation_time = 5000
        self.list_of_BFS = []

    def independent(self, subset):
        submatrix = self.A[:, subset]
        rank = np.linalg.matrix_rank(submatrix)
        size = len(subset)
        return rank == size

    def reject_sample_basis(self):
        #Good if $A$ is a random
        m = self.rank
        candidate = np.random.choice(list(range(self.num_variables)), m, replace = False)
        while not self.independent(candidate):
            candidate = np.random.choice(list(range(self.num_variables)), m, replace=False)
        return candidate
    
    def pick_basis(self):
        independent_set = []
        ordering = np.random.permutation(list(range(self.num_variables)))
        for x in ordering:
            new_set = independent_set + [x]
            if self.independent(new_set):
                independent_set = new_set
        return independent_set

    def step_on_BFS(self):
        '''This does a basis exchange step constrained to remain on the polytope'''
        element_out = random.choice(self.current_basis)

        element_in = random.choice([ x for x in range(self.num_variables) if x not in self.current_basis])

        self.current_basis.remove(element_out)
        self.current_basis.append(element_in)

        if self.independent(self.current_basis):
            self.find_basic_feasible_solution()
            if self.bfs_nonnegative():
                return
        self.current_basis.remove(element_in)
        self.current_basis.append(element_out)

    def basis_exchange(self):
        element_out = random.choice(self.current_basis)

        element_in = random.choice(  [ x for x in range(self.num_variables) if x not in self.current_basis])

        self.current_basis.remove(element_out)
        self.current_basis.append(element_in)

        if self.independent(self.current_basis):
            return
        self.current_basis.remove(element_in)
        self.current_basis.append(element_out)
        return

    def bfs_nonnegative(self):
        for t in self.basic_feasible_solution:
            if t < 0:
                return False
        return True

    def do_basis_exchange(self):
        for i in range(self.tries):
            for j in range(self.estimated_relaxation_time):
                self.basis_exchange()
            self.find_basic_feasible_solution()
            if self.bfs_nonnegative():
                #print(np.sort(self.current_basis))
                #print(self.basic_feasible_solution)
                self.list_of_BFS.append(self.basic_feasible_solution)


    def jumper(self):
        #This *ONLY* works if the matrix A is in general position, because then it always takes the first m columns
        #in the permutation. OW it is closer to the MST distribution.
        for i in range(self.tries):
            self.current_basis = self.reject_sample_basis()
            self.find_basic_feasible_solution()
            if self.bfs_nonnegative():
                #print(np.sort(self.current_basis))
                #print(self.basic_feasible_solution)
                self.list_of_BFS.append(self.basic_feasible_solution)

    def do_constrained_basis_exchange(self):
        self.find_basic_feasible_solution()
        while not self.bfs_nonnegative():
            self.basis_exchange()
            self.find_basic_feasible_solution()
        print("started")
        for i in range(self.tries):
            for j in range(self.estimated_relaxation_time):
                self.step_on_BFS()
            self.find_basic_feasible_solution()
            if self.bfs_nonnegative():
                #print(np.sort(self.current_basis))
                #print(self.basic_feasible_solution)
                self.list_of_BFS.append(self.basic_feasible_solution)

    def find_basic_feasible_solution(self):
        subset = np.sort(self.current_basis)

        #This is just because we will want to know the order
        inverse = {}
        j = 0
        for i in subset:
            inverse[i] = j
            j += 1
        #TODO -- use index instead

        submatrix = self.A[:, subset]
        x_B = np.linalg.solve(submatrix, self.b)
        x = [0 for t in range(self.num_variables)]
        for i in range(self.num_variables):
            if i in subset:
                x[i] = x_B[inverse[i]]
        self.basic_feasible_solution = x

def build_degenerate_cycle_polytope(graph):
    '''
    This takes a graph an builds the equational form of the system (2.2) from Coullard / Pulleyblank
    We need x[e] for all edges e, and also slack variables s[e,v], for x( \delta(v)) - x(e) = s[e,v] --
    SO each edge of G appears in exactly 3 variables
    We also add a constraint that cuts out the vertex figure of the cone at the origin.
    There may be a better way to choose that cutting plane?

    Remark: It's not surprising that the rejection rate is so high -- think of it this way:
    each basis gives a set of column vectors. We get a BFS iff b is in the cone generated by those column vectors.
    The matrix A has

    '''
    list_of_edges = graph.graph["edgelist"]
    num_edges = len(graph.edges())

    num_variables = 3*num_edges
    num_equations = 2 * num_edges +1

    #Each edge will contribute 2 constraints

    A = np.zeros([ num_equations , num_variables])

    for edge_number in range(len(graph.edges())):
        edge = list_of_edges[edge_number]
        q = 0
        for u in edge:
            row = np.zeros(num_variables)
            neighbors = list(graph[u])
            adjacent_edges = [(u, v) for v in neighbors if (u, v) in list_of_edges and (u, v) != edge] \
                             + [(v, u) for v in neighbors if (v, u) in list_of_edges and (v, u) != edge]
            indices = [list_of_edges.index(e) for e in adjacent_edges]
            row[edge_number] = -1
            for i in indices:
                row[i] = 1

            #slack variable entry:
            row[(q + 1)*num_edges + edge_number] = -1
            A[2*edge_number +q] = row
            q += 1

    last_row = np.zeros(num_variables)
    '''
    I think it's ok to cut with the edge weights, because lemma 2.1 in "On Cycle Cones and Polyhedra" tells us
    that 
    
    
    for i in range(num_edges):
    '''
    for i in range(num_variables):
        last_row[i] = 1
    A[-1] = last_row
    b = np.zeros(num_equations)
    b[-1] = 2*len(graph.nodes())
    return [A,b]


def build_degenerate_cycle_box_polytope(graph):
    '''
    This takes a graph an builds the equational form of the system (2.2) from Coullard / Pulleyblank
    We need x[e] for all edges e, and also slack variables s[e,v], for x( \delta(v)) - x(e) = s[e,v] --
    SO each edge of G appears in exactly 3 variables
    We also add a constraint that cuts out the vertex figure of the cone at the origin.
    There may be a better way to choose that cutting plane?

    Remark: It's not surprising that the rejection rate is so high -- think of it this way:
    each basis gives a set of column vectors. We get a BFS iff b is in the cone generated by those column vectors.
    The matrix A has

    '''
    list_of_edges = graph.graph["edgelist"]
    num_edges = len(graph.edges())

    num_variables = 4 * num_edges
    num_equations = 3 * num_edges

    # Each edge will contribute 2 constraints
    #Plug we have the upper bound constraints
    A = np.zeros([num_equations, num_variables])

    for edge_number in range(len(graph.edges())):
        edge = list_of_edges[edge_number]
        q = 0
        for u in edge:
            row = np.zeros(num_variables)
            neighbors = list(graph[u])
            adjacent_edges = [(u, v) for v in neighbors if (u, v) in list_of_edges and (u, v) != edge] \
                             + [(v, u) for v in neighbors if (v, u) in list_of_edges and (v, u) != edge]
            indices = [list_of_edges.index(e) for e in adjacent_edges]
            row[edge_number] = -1
            for i in indices:
                row[i] = 1

            # slack variable entry:
            row[(q + 1) * num_edges + edge_number] = -1
            A[2 * edge_number + q] = row
            q += 1

    for i in range(num_edges):
        new_row = np.zeros(num_variables)
        #Add other slack variables
        new_row[i] = 1
        new_row[3* num_edges + i] = 1
        A[-i - 1] = new_row
    b = np.zeros(num_equations)

    for i in range(num_edges):
        b[-i - 1] = 1
    return [A, b]

def convert_bfs_to_edges(graph, x):

    list_of_edges = graph.graph["edgelist"]
    edge_set = []
    colors = {}
    for edge_number in range(len(graph.edges())):
        if x[edge_number] > .0001:
            edge_set.append(list_of_edges[edge_number])
            colors[list_of_edges[edge_number]] = x[edge_number]
    return [edge_set, colors]

def viz_edge(T, set_of_cycles):
    i = 0
    for edge_path in set_of_cycles:
        k = 20

        values = [1 - int((x in edge_path) or ((x[1], x[0]) in edge_path)) for x in T.edges()]

        f = plt.figure()

        nx.draw(T, ax=f.add_subplot(111), pos=nx.get_node_attributes(T, 'coord'), node_size = 1, width =2, cmap=plt.get_cmap('jet'),  edge_color=values)
        f.savefig("graph" + str(len(T.nodes())) + "sample_cycle" + str(i))
        #plt.close(f)
        i += 1

def viz_edge_gradient(graph, edge_path, colors):

    k = 20

    for e in graph.edges():
        if e not in edge_path or (e[1], e[0]) not in edge_path:
            if e  in edge_path:
                colors[(e[1], e[0])] = colors[e]
            if (e[1], e[0])  in edge_path:
                colors[e] = colors[(e[1], e[0]) ]
            if e not in edge_path and  (e[1], e[0]) not in edge_path:
                colors[e] = 0
    values = [colors[x] for x in graph.edges()]

    f = plt.figure()

    nx.draw(graph, ax=f.add_subplot(111), pos=nx.get_node_attributes(graph, 'coord'), node_size = 1, width =2, edge_cmap=plt.cm.Greys,  edge_color=values)
    f.savefig("graph" + str(len(graph.nodes())) + "sample_cycle" + str(i))
    #plt.close(f)
    np.min ( list(colors.values()))


def test_rejection_random_polytope():


    means  = np.zeros([1000,1000])
    variances = np.zeros([1000,1000])

    tries = 1000
    lower_dimension = 40
    upper_dimension = 41
    lower_size_bonus = 0
    upper_size_bonus = 1
    tally = []
    sphere = False
    for size_bonus in range(lower_size_bonus, upper_size_bonus):

        for d in range(lower_dimension, upper_dimension):
            rate = []
            for i in range(10):
                dimension = 3 + d
                size = d + size_bonus + 5

                A = np.random.uniform(0, 1, [dimension,  size])

                if sphere == True:
                    A = []
                    for i in range(size):
                        a = np.random.multivariate_normal(np.zeros(dimension),np.identity(dimension))
                        a = a / np.linalg.norm(a)
                        a = np.abs(a)
                        A.append(a)
                    A = np.array(A).T

                b = np.ones(dimension)
                b = np.mean(A,1)
                
                res = linprog(c=np.zeros(A.shape[1]), A_eq=A, b_eq=b)
                #Make sure we have a BFS
                while not res["success"]:
                    A = np.random.uniform(0, 1, [dimension, size])

                    if sphere == True:
                        A = []
                        for i in range(size):
                            a = np.random.multivariate_normal(np.zeros(dimension), np.identity(dimension))
                            a = a / np.linalg.norm(a)
                            a = np.abs(a)
                            A.append(a)
                        A = np.array(A).T

                    b = np.ones(dimension)
                    b = np.mean(A, 1)
                    res = linprog(c=np.zeros(A.shape[1]), A_eq=A, b_eq=b)
                #print("success")
                M = LPMatroid(A,b)

                M.tries = tries
                #M.do_basis_exchange()
                M.jumper()
                rate.append(len(M.list_of_BFS)/M.tries)
                #print(len(M.list_of_BFS)/M.tries)
            tally.append(rate)
            print("Dimension: ", dimension, "Mean: ", np.mean(rate), "Var:", np.var(rate) )
            means[dimension - 6, size_bonus] = np.mean(rate)
            variances[dimension, size_bonus] = np.var(rate)

    print(means[0:20, 0:20])


def exploring_01intersection():
    #matplotlib.use("Agg")

    graph = nx.grid_graph([10,10])
    for v in graph.nodes():
        graph.node[v]['coord'] = v

    #graph = nx.complete_graph(4)
    list_of_edges = list(graph.edges())
    graph.graph["edgelist"] = list_of_edges
    A, b = build_degenerate_cycle_box_polytope(graph)

    m = A.shape[1]
    num_edges = len(graph.edges())
    c = np.zeros(m)

    for i in range(num_edges):
        c[i] = -1 * np.random.binomial(2,.5)

    x = linprog(c, A_eq=A, b_eq=b,  method = 'interior-point')
    edge_path, colors = convert_bfs_to_edges(graph, x['x'])
    viz_edge_gradient(graph, edge_path, colors)
    print(edge_path)

def test_random_direction(A, b):

    list_of_solutions = []
    num_samples = 1
    m = A.shape[1]
    for i in range(num_samples):
        c = np.random.normal(0, 1, m)
        c  = np.ones(m)
        d = np.zeros(m)
        for i in range(m):
            d[i] = -1
        for i in range(int(m/3)):
            d[i] = 1
        x = linprog(d, A_eq = A, b_eq = b)
        list_of_solutions.append(x['x'])

    print("found:", len(list_of_solutions))
    if len(list_of_solutions) > 0:

        viz_edge(graph, list_of_solutions)

def test_basis_walk(A,b):

    M = LPMatroid(A, b)
    M.estimated_relaxation_time = 10000

    M.tries = 100

    M.do_constrained_basis_exchange()

    list_of_solutions = [convert_bfs_to_edges(graph, x) for x in M.list_of_BFS]

    #Why do we keep getting the same cycle?

    print("found:", len(list_of_solutions))
    if len(list_of_solutions) > 0:

        viz_edge(graph, list_of_solutions)
