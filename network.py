import networkx as nx
import matplotlib.pyplot as plt

class Network:
    def __init__(self, rng, params: dict):
        self.rng = rng
        self.params = params
        self.size = self.params["size"]
        self.nodes = list(range(self.size))
        self.defined_cliques: 'list[list[int]]' = []
        self.node_cliques: 'dict[int, list[int]]' = {}

        self.graph = self.generate_clique_based_graph(
            self.params["clique_size"], 
            self.params["node_out_degree"], 
            self.params["propinquity"]
        )
        self.last_communication_graph = nx.DiGraph()

        self.severed_edges = set()


    def get_shuffled_nodes(self):
        copied = self.nodes.copy()
        self.rng.shuffle(copied)
        return copied

  
    def generate_clique_based_graph(self, 
                                  clique_size: int,
                                  node_out_degree: int,
                                  propinquity: float):
        """
        target_graph: the graph to generate this topology on
        clique_size: the size of the cliques in the graph
        node_out_degree: the number of outbound edges to give each node
        propinquity: the probability that a new edge is within a node's clique
        """

        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)

        # first, sort nodes into cliques
        assert self.size % clique_size == 0, "Size must be divisible by clique size"
        for i in range(0, self.size, clique_size):
            clique = list(range(i, i + clique_size))
            for node in clique:
                self.node_cliques[node] = clique
            self.defined_cliques.append(clique)
        
        # second, generate edges

        for u in self.nodes:
            clique = self.node_cliques[u].copy()
            clique.remove(u)
            excluded = clique.copy()
            for i in range(node_out_degree):
                if self.rng.random() <= propinquity:
                    v = self.rng.choice(clique)
                else:
                    v = self.get_random_node(excluded)
                    
                excluded.append(v)
                G.add_edge(u, v)
        
        return G

  
    def get_random_node(self, excluded=[]) -> int:
        node = self.rng.choice(self.nodes)
        while node in excluded:
            node = self.rng.choice(self.nodes)
        return node


    def sever_connection(self, node_a: int, node_b: int) -> None:
        if self.graph.has_edge(node_a, node_b):
            self.graph.remove_edge(node_a, node_b)
            self.graph.add_edge(node_a, self.get_random_node([node_a, node_b]))
            self.severed_edges.add((node_a, node_b))


    def has_been_severed(self, node_a: int, node_b: int) -> bool:
        return (node_a, node_b) in self.severed_edges


    def relocate(self, node: int, node_out_degree: int) -> None:
        # TODO: break edges
        current_clique = self.node_cliques[node]
        clique_options = self.defined_cliques.copy() # note: shallow copy
        clique_options.remove(current_clique)
        new_clique = clique_options[self.rng.integers(0, len(clique_options))]
        vs = self.rng.choice(
            new_clique, 
            size=min(node_out_degree, len(new_clique)),
            replace=True
        )
        for v in vs:
            if not self.graph.has_edge(node, v):
                self.graph.add_edge(node, v)
            if not self.graph.has_edge(v, node):
                self.graph.add_edge(v, node)

        new_clique.append(node)
        self.node_cliques[node] = new_clique


    def get_influencers(self, node: int):
        return self.graph.neighbors(node)


    def reset_communication_graph(self):
        self.last_communication_graph = nx.DiGraph()


    def log_communication(self, to_node: int, from_node: int):
        self.last_communication_graph.add_edge(from_node, to_node)


    def get_last_observers(self, node: int):
        return self.last_communication_graph[node]


    def display_connection_graph(self, title, node_colors, edge_colors, pos=None, save_path=None):
        if pos == None:
            pos = nx.spring_layout(self.graph)

        fig = plt.figure()
        plt.title(title)

        nx.draw(
            self.graph, 
            pos=pos, 
            node_size=25, 
            node_color=node_colors, 
            edge_color=edge_colors
        )
        if save_path == None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close(fig)
        return pos