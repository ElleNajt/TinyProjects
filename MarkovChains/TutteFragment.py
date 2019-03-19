import networkx as nx
import copy

#For checking that \tilde{F} is 3 connected

G = nx.Graph()

nodes = list(range(16))

G.add_nodes_from(nodes)

edges = [
    [0,1],[0,5],[0,13],
    [1,2],[1,8],
    [2,3],[2,9],
    [3,4],[3,7],
    [4,5],[4,6],
    [5,14],
    [6,7],[6,11],
    [7,10],
    [8,9],[8,12],
    [9,10],
    [10,15],
    [11,14],[11,15],
    [12,15],[12,13],
    [13,14]
]
G.add_edges_from(edges)

#Sanity checking:

print(dict(G.degree()).values())


for x in nodes:
    for y in nodes:
        H = copy.deepcopy(G)
        H.remove_nodes_from([x,y])
        if nx.is_connected(H) == False:
            print(x,y)