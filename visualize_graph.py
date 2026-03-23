import matplotlib.pyplot as plt
import networkx as nx
import os


def visualize_graph(G, output_path="graph.png"):

    if G.number_of_nodes() == 0:
        return "Graph is empty"

    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(G, k=0.5)

    # Node colors by type
    color_map = []
    for node in G.nodes(data=True):
        ntype = node[1].get("type", "unknown")

        if ntype == "gene":
            color_map.append("red")
        elif ntype == "drug":
            color_map.append("green")
        elif ntype == "protein":
            color_map.append("blue")
        elif ntype == "pathway":
            color_map.append("orange")
        else:
            color_map.append("gray")

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=500,
        font_size=8
    )

    plt.title("Knowledge Graph")
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path