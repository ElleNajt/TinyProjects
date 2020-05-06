import networkx as nx
import numpy as np

graph = nx.grid_graph([5,5])
matroid = nx.incidence_matrix(graph, oriented = True).A
n = len(G.nodes())

node_list = list(G.nodes())

for e in graph.edges():
    graph.edges[e]["vector"] = [v in e for v in node_list]

initial_vectors = np.identity(n)

graph.graph["node_list"] = node_list

def construct_means(graph, vectors):
    for e in graph.edges():
        graph.edges[e]["weight"] = np.sum([ np.linalg.matmul(x, graph.edges[e]["vector"]) for x in vectors])
    graph.nodes[node_list[0]]["mean"] = np.random.normal(0,1)
    
    


def produce_sample(graph, vectors):
    ##Construct means:
    construct_means(graph, vectors)
    