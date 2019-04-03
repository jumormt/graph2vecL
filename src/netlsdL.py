import netlsd
import networkx as nx

# g = nx.erdos_renyi_graph(100, 0.01) # create a random graph with 100 nodes
g = nx.Graph()
g.add_nodes_from([1,2,3])
g.add_edges_from([(1, 2), (1, 3)])
g.add_node("spam")
g.add_nodes_from("spam")
g.add_edge(3,'m')

descriptor = netlsd.heat(g) # compute the signature

print("end")