import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class LPMatroid():
    def __init__(self, A, b):
        self.A = A
        self.num_variables = A.shape[1]
        self.num_equations = A.shape[0]
        if len(b) != self.num_equations:
            print ( "Dimensions do not agree")
            raise
        self.rank = np.linalg.matrix_rank(A)
        self.current_basis = self.pick_basis()
        self.b = b
        self.basic_feasible_solution = 0
        self.tries = 1000
        self.list_of_BFS = []
    def independent(self, subset):
        submatrix = self.A[:,subset]
        rank = np.linalg.matrix_rank(submatrix)
        size = len(subset)
        return rank == size

    def pick_basis(self):
        independent_set = []
        ordering = np.random.permutation(list(range(self.num_variables)))
        for x in ordering:
            new_set = independent_set + [x]
            if self.independent(new_set):
                independent_set = new_set
        return independent_set

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

    def show_basis_exchange(self):
        for i in range(self.tries):
            self.basis_exchange()
            self.find_basic_feasible_solution()
            if self.bfs_nonnegative():
                print(np.sort(self.current_basis))
                print(self.basic_feasible_solution)
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

        submatrix = self.A[:,subset]
        x_B = np.linalg.solve(submatrix, self.b)
        x = [0 for t in range(self.num_variables)]
        for i in range(self.num_variables):
            if i in subset:
                x[i] = x_B[inverse[i]]
        self.basic_feasible_solution = x

def build_degenerate_cycle_polytope(graph):
    #This takes a graph an builds the equational form of the system (2.2) from Coullard / Pulleyblank
    #We need x[e] for all edges e, and also slack variables s[e,v], for x( \delta(v)) - x(e) = s[e,v] --
    #SO each edge of G appears in exactly 3 variables

    #Let's put an arbitrary direction on the graph
    #Also, I want

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
            adjacent_edges = [(u,v) for v in neighbors if (u,v) in list_of_edges and (u,v) != edge]  + [(v,u) for v in neighbors if (v,u) in list_of_edges and (v,u) != edge]
            indices = [list_of_edges.index(e) for e in adjacent_edges]
            row[edge_number] = -1
            for i in indices:
                row[i] = 1

            #slack variable entry:
            row[(q + 1)*num_edges + edge_number] = -1
            A[2*edge_number +q] = row
            q += 1

    last_row = np.zeros(num_variables)
    for i in range(num_edges):
        last_row[i] = 1
    A[-1] = last_row
    b = np.zeros(num_equations)
    b[-1] = len(graph.nodes())
    return [A,b]

def convert_bfs_to_edges(graph, x):

    list_of_edges = graph.graph["edgelist"]
    edge_set = []
    for edge_number in range(len(graph.edges())):
        if x[edge_number] > .01:
            edge_set.append(list_of_edges[edge_number])

    return edge_set

def viz_edge(T, set_of_cycles):
    i = 0
    for edge_path in set_of_cycles:
        k = 20

        values = [1 - int((x in edge_path) or ((x[1], x[0]) in edge_path)) for x in T.edges()]
        plt.figure(i)
        nx.draw(T, pos=nx.get_node_attributes(T, 'coord'), node_size = 1, width =2, cmap=plt.get_cmap('jet'),  edge_color=values)
        plt.savefig("sample_cycle" + str(i))
        i += 1

graph = nx.grid_graph([5,5])
for v in graph.nodes():
    graph.node[v]['coord'] = v

#graph = nx.complete_graph(4)
list_of_edges = list(graph.edges())
graph.graph["edgelist"] = list_of_edges
A, b = build_degenerate_cycle_polytope(graph)


#A = np.random.uniform(0,1, [4,10])

M = LPMatroid(A,b)
M.show_basis_exchange()

M.tries = 10*16
M.find_basic_feasible_solution()

list_of_solutions = [convert_bfs_to_edges(graph, x) for x in M.list_of_BFS]

print("found:", len(list_of_solutions))
if 5 > len(list_of_solutions)> 0 :

    viz_edge(graph, list_of_solutions)

#I'm confused -- there's a subtle thing: the basis exchange graph mixes rapidly, but constrain it to lie on the polytope
#may not.
#On the other hand, we can try to constrain the steps to lie on the polytope.