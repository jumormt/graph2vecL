import json
import networkx as nx
import matplotlib.pyplot as plt

precutroot = "/Users/chengxiao/Downloads/SARD.2019-02-28-22-07-31/noerror/result_sym/"
cutroot = "/Users/chengxiao/Downloads/SARD.2019-02-28-22-07-31/noerror/cut/result_sym/"
# graphPath = precutroot+"ahscpy1-good.c.exm_1_sym.json"
graphPath = "/Users/chengxiao/Downloads/CWE-691/424/raw_result/"+"CWE366_Race_Condition_Within_Thread__global_int_04.c.exm_0_1.json"

name = graphPath.strip(".json").split("/")[-1]
data = json.load(open(graphPath, encoding="utf8"))
gra = nx.from_edgelist(data["edges"])
G = nx.DiGraph()
G.add_edges_from(data["edges"])
nodes = data['nodes']
for i in range(len(nodes)):
    G.nodes[i]["code"] = nodes[i]
node_labels = nx.get_node_attributes(G,'code')
options = {
    'node_size': 500,
    'width': 2,
    'arrows': True,
    'alpha':0.5,
    # 'node_color':'white',
    # 'labels':node_labels
}
plt.subplot(111)
pos = nx.spring_layout(G)
# nx.draw_networkx_labels(G, labels = node_labels)
nx.draw(G, with_labels=True, font_weight='bold', **options)
for i in range(len(nodes)):
    x,y = pos[i]
    # plt.text(x, y + 0.1, s=G.nodes[i]["code"], bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center')
    print("node{}: ".format(i))
    print(G.nodes[i]["code"])
# plt.subplot(122)
# nx.draw_shell(graph, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold',**options)
plt.show()
print()
# plt.savefig("../resources/images/image.png",bbox_inches='tight',dpi=100)