import numpy as np
import math
import pandas as pd

class Graph(object):

    def __init__(self, size, with_fixed_edges=False):
        self.size = size
        self.nodes = list(range(size))
        if with_fixed_edges:
            self.adj_matrix = np.zeros((self.size, self.size), dtype=int)
        else:
            self.adj_matrix = np.full((self.size, self.size), 2, dtype=int)
            for i in range(self.size):
                self.adj_matrix[i][i] = 0
        self.first_partition_size = 0

    def add_edge(self, edge):
        n1, n2 = edge
        self.adj_matrix[n1][n2] = 1
        self.adj_matrix[n2][n1] = 1

    def get_edge_label(self, edge):
        n1, n2 = edge
        return self.adj_matrix[n1][n2]

    def __len__(self):
        return len(self.graph.edges())

    def add_edges_from(self, edges):
        for edge in edges:
            n1, n2 = edge
            self.adj_matrix[n1][n2] = 1
            self.adj_matrix[n2][n1] = 1

    def remove_edge(self, edge):
        n1, n2 = edge
        self.adj_matrix[n1][n2] = 0
        self.adj_matrix[n2][n1] = 0

    def neighbors(self, n1):
        return np.where(self.adj_matrix[n1] == 1)[0]


    def common_neighbors_index(self, edge=None):
        if edge:
            n1, n2 = edge
            return len(set(self.neighbors(n1)).intersection(set(self.neighbors(n2))))
        else:
            common_neighbors = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(i, self.size):
                    common_neighbors[i][j] = len(set(self.neighbors(i)).intersection(set(self.neighbors(j))))
                    common_neighbors[j][i] = common_neighbors[i][j]
            return common_neighbors


    def jaccard_index(self, edge=None):

        if edge:
            n1, n2 = edge
            common_neighbors = self.common_neighbors_index(edge)
            union_neighbors = len(set(self.neighbors(n1)).union(set(self.neighbors(n2))))
            if union_neighbors == 0:
                return 0
            return common_neighbors / union_neighbors
        else:
            jaccard = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(i+1, self.size):
                    common_neighbors = self.common_neighbors_index((i, j))
                    union_neighbors = len(set(self.neighbors(i)).union(set(self.neighbors(j))))
                    if union_neighbors == 0:
                        jaccard[i][j] = 0
                    else:
                        jaccard[i][j] = common_neighbors / union_neighbors
            return jaccard


    def preferential_attachment_index(self, edge=None):
        if edge:
            n1, n2 = edge
            return len(self.neighbors(n1)) * len(self.neighbors(n2))
        else:
            pref_attach = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(i+1, self.size):
                    pref_attach[i][j] = len(self.neighbors(i)) * len(self.neighbors(j))
            return pref_attach

    def adamic_adar_index(self, edge=None):
        if edge:
            n1, n2 = edge
            common_neighbors = set(self.neighbors(n1)).intersection(set(self.neighbors(n2)))
            return sum([1 / math.log(len(self.neighbors(k))) for k in common_neighbors])
        adamic_adar = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(i+1, self.size):
                common_neighbors = set(self.neighbors(i)).intersection(set(self.neighbors(j)))
                adamic_adar[i][j] = sum([1 / math.log(len(self.neighbors(k))) for k in common_neighbors])
        return adamic_adar

    def resource_allocation_index(self, edge=None):
        if edge:
            n1, n2 = edge
            common_neighbors = set(self.neighbors(n1)).intersection(set(self.neighbors(n2)))
            return sum([1 / len(self.neighbors(k)) for k in common_neighbors])

        else:
            resource_allocation = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(i+1, self.size):
                    common_neighbors = set(self.neighbors(i)).intersection(set(self.neighbors(j)))
                    resource_allocation[i][j] = sum([1 / len(self.neighbors(k)) for k in common_neighbors])
            return resource_allocation

    
    def link_prediction_scores(self, method, edge=None):
        if method == "jaccard":
            return self.jaccard_index(edge=edge)
        elif method == "preferential_attachment":
            return self.preferential_attachment_index(edge=edge)
        elif method == "adamic_adar":
            return self.adamic_adar_index(edge=edge)
        elif method == "resource_allocation":
            return self.resource_allocation_index(edge=edge)
        elif method == "common_neighbors":
            return self.common_neighbors_index(edge=edge)

        else:
            print("Non supported link prediction method ", method)
            return None


    def has_edge(self, edge):
        n1, n2 = edge
        return self.adj_matrix[n1][n2] == 1

    def does_not_know_edge(self, edge):
        n1, n2 = edge
        return self.adj_matrix[n1][n2] == 2

    def does_not_have_edge(self, edge):
        n1, n2 = edge
        return self.adj_matrix[n1][n2] == 0

    def edges(self):
        return [(i, j) for i in range(self.size) for j in range(i, self.size) if self.adj_matrix[i][j] == 1]

    def degree(self, node):
        return len(self.neighbors(node))

    def common_neighbors(self, edge):
        n1, n2 = edge
        return set(self.neighbors(n1)).intersection(set(self.neighbors(n2)))

    def stats(self):
        num_present = len(np.where(self.adj_matrix == 1)[0])
        num_absent = len(np.where(self.adj_matrix == 0)[0])
        num_unknown = len(np.where(self.adj_matrix == 2)[0])

        return num_absent, num_present, num_unknown

    def copy(self):
        copy =  Graph(self.size)
        copy.adj_matrix = np.copy(self.adj_matrix)
        return copy

    @staticmethod
    def from_nodes_and_edges(nodes, edges):
        graph = Graph(nodes, with_fixed_edges=True)
        
        for edge in edges:
            graph.add_edge(edge)

        return graph

    @staticmethod
    def from_adj_matrix(adj_matrix, with_fixed_edges=False):
        nodes = list(range(len(adj_matrix)))
        graph = Graph(len(nodes), with_fixed_edges=with_fixed_edges)
        graph.adj_matrix = adj_matrix
        return graph

    @staticmethod
    def from_npy(filepath):
        adj_matrix = np.load(filepath, allow_pickle=True)
        return Graph.from_adj_matrix(adj_matrix, with_fixed_edges=True)


    @staticmethod
    def from_txt(filepath):
        edges = pd.read_csv(filepath, sep='\t', header=None)
        nodes = list(range(0, max(edges[0].tolist() + edges[1].tolist() ) ))
        graph = Graph(len(nodes), with_fixed_edges=True)
        for i in range(len(edges)):
            graph.add_edge((edges[0][i]-1, edges[1][i]-1))

        return graph

    @staticmethod
    def block_from_txt(filepath):
        edges = pd.read_csv(filepath, sep='\t', header=None)
        max_partition_1 = max(edges[0].tolist())
        nodes_partition_1 = list(range(0, max_partition_1 ) )
        nodes_partition_2 = list(range(max_partition_1, max(edges[1].tolist()) + max_partition_1 ))
        nodes = nodes_partition_1 + nodes_partition_2
        graph = Graph(nodes, with_fixed_edges=True)
        for i in range(len(edges)):
            n1 = edges[0][i]-1
            n2 = edges[1][i]-1 + max_partition_1
            graph.add_edge((n1, n2))

        graph.first_partition_size = max_partition_1
        return graph

    @staticmethod
    def random_graph(nodes, edge_prob=0.5):
        graph = Graph(nodes)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if np.random.random() < edge_prob:
                    graph.add_edge((i, j))
        return graph

    @staticmethod
    def barabasi_albert_graph(size, m=1, m0=1):
        graph = Graph(size, with_fixed_edges=True)
        for i in range(m0, size):
            probs = [graph.degree(j) for j in range(i)]
            probs = [p / sum(probs) for p in probs] if sum(probs) > 0 else [1 / i for _ in range(i)]
            targets = np.random.choice(range(i), size=m, p=probs, replace=False)
            for target in targets:
                graph.add_edge((i, target))

        return graph




    def unknown_edges(self):
        return [(i, j) for i in range(self.size) for j in range(i+1, self.size) if self.adj_matrix[i][j] == 2]

    def split_dataset(self, common_prop=0, graph1_prop=0.5):
        # sample edges and non edges to create two graphs
        graph1 = self.copy()
        graph2 = self.copy()

        graph1_threshold = common_prop + graph1_prop

        for i in range(self.size):
            for j in range(i+1, self.size):
                prob = np.random.random()
                if prob < common_prop:
                    graph1.adj_matrix[i][j] = self.adj_matrix[i][j]
                    graph1.adj_matrix[j][i] = self.adj_matrix[j][i]
                    graph2.adj_matrix[i][j] = self.adj_matrix[i][j]
                    graph2.adj_matrix[j][i] = self.adj_matrix[j][i]
                elif common_prop < prob < graph1_threshold:
                    graph1.adj_matrix[i][j] = self.adj_matrix[i][j]
                    graph1.adj_matrix[j][i] = self.adj_matrix[j][i]
                    graph2.adj_matrix[i][j] = 2
                    graph2.adj_matrix[j][i] = 2
                else:
                    graph1.adj_matrix[i][j] = 2
                    graph1.adj_matrix[j][i] = 2
                    graph2.adj_matrix[i][j] = self.adj_matrix[i][j]
                    graph2.adj_matrix[j][i] = self.adj_matrix[j][i]
                    
        return graph1, graph2

    def fix_edges(self, value=0):
        for i in range(self.size):
            for j in range(i+1, self.size):
                if self.adj_matrix[i][j] == 2:
                    self.adj_matrix[i][j] = value
                    self.adj_matrix[j][i] = value


    def global_clustering_coefficient(self):
        adj_matrix = self.adj_matrix
        adj_matrix_3 = np.dot(adj_matrix, np.dot(adj_matrix, adj_matrix))
        adj_matrix_2 = np.dot(adj_matrix, adj_matrix)
        numerator = 1/6 * np.trace(adj_matrix_3)
        denominator = 1/2 * np.sum(adj_matrix_2[i, j] for i in range(self.size) for j in range(self.size) if i != j)
        return numerator / denominator


    def to_networkx(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for i in range(self.size):
            for j in range(i+1, self.size):
                if self.adj_matrix[i][j] == 1:
                    G.add_edge(i, j)
        return G

    def to_txt(self, filepath):
        with open(filepath, 'w') as f:
            for (n1, n2) in self.edges():
                f.write(f"{n1+1}\t{n2+1 - self.first_partition_size}\t1\n")


                    


class BipartiteGraph:

    def __init__(self, part1_nodes, part2_nodes, adj_matrix=None):
        self.part1_size = part1_nodes
        self.part2_size = part2_nodes
        self.total_size = self.part1_size + self.part2_size
        if adj_matrix is not None:
            self.adj_matrix = adj_matrix
        else:
            self.adj_matrix = np.full((self.part1_size, self.part2_size), 2, dtype=int)


    def edges(self):
        edges = []
        for i in range(self.part1_size):
            for j in range(self.part2_size):
                if self.adj_matrix[i][j] == 1:
                    edges.append((i, j + self.part1_size))
        return edges

    def add_edge(self, edge):
        n1, n2 = edge
        if n1 < self.part1_size and n2 >= self.part1_size:
            self.adj_matrix[n1][n2 - self.part1_size] = 1
        elif n2 < self.part1_size and n1 >= self.part1_size:
            self.adj_matrix[n2][n1 - self.part1_size] = 1
        else:
            raise ValueError("Edges can only exist between partitions in a bipartite graph.")

    def get_edge_label(self, edge):
        n1, n2 = edge
        if n1 < self.part1_size and n2 >= self.part1_size:
            return self.adj_matrix[n1][n2 - self.part1_size]
        elif n2 < self.part1_size and n1 >= self.part1_size:
            return self.adj_matrix[n2][n1 - self.part1_size]
        else:
            raise ValueError("Invalid edge in bipartite graph.")

    def add_edges_from(self, edges):
        for edge in edges:
            self.add_edge(edge)

    def remove_edge(self, edge):
        n1, n2 = edge
        if n1 < self.part1_size and n2 >= self.part1_size:
            self.adj_matrix[n1][n2 - self.part1_size] = 0
        elif n2 < self.part1_size and n1 >= self.part1_size:
            self.adj_matrix[n2][n1 - self.part1_size] = 0
        else:
            raise ValueError("Edges can only be removed between partitions in a bipartite graph.")

    def neighbors(self, node):
        if node < self.part1_size:
            return np.where(self.adj_matrix[node] == 1)[0] + self.part1_size
        elif node >= self.part1_size:
            return np.where(self.adj_matrix[:, node - self.part1_size] == 1)[0]
        else:
            raise ValueError("Node out of bounds in bipartite graph.")

    def common_neighbors(self, edge):
        n1, n2 = edge
        return set(self.neighbors(n1)).intersection(set(self.neighbors(n2)))

    def common_neighbors_index(self, edge=None):
        if edge:
            n1, n2 = edge
            return len(set(self.neighbors(n1)).intersection(set(self.neighbors(n2))))
        else:
            common_neighbors = np.zeros((self.total_size, self.total_size))
            for i in range(self.part1_size):
                for j in range(self.part2_size):
                    common_neighbors[i][j] = len(set(self.neighbors(i)).intersection(set(self.neighbors(j + self.part1_size))))
            return common_neighbors

    # Similar methods can be written for jaccard_index, adamic_adar_index, etc.
    # Add the other methods based on bipartite graph structure.

    @staticmethod
    def from_adj_matrix(adj_matrix):
        part1_nodes = adj_matrix.shape[0]
        part2_nodes = adj_matrix.shape[1]
        return BipartiteGraph(part1_nodes, part2_nodes, adj_matrix)

    @staticmethod
    def from_txt(filepath):
        edges = pd.read_csv(filepath, sep='\t', header=None)
        part1_nodes = max(edges[0].tolist())
        part2_nodes = max(edges[1].tolist())
        graph = BipartiteGraph(part1_nodes, part2_nodes)
        for i in range(len(edges)):
            graph.add_edge((edges[0][i]-1, edges[1][i]-1 + part1_nodes))
        return graph

    def unknown_edges(self):
        return [(i, j) for i in range(self.part1_size) for j in range(self.part2_size) if self.adj_matrix[i][j] == 2]

    def to_txt(self, filepath):
        with open(filepath, 'w') as f:
            for i in range(self.part1_size):
                for j in range(self.part2_size):
                    if self.adj_matrix[i][j] == 1:
                        f.write(f"{i+1}\t{j+1}\n")

    def split_dataset(self, common_prop=0, graph1_prop=0.5):
        graph1 = self.copy()
        graph2 = self.copy()

        graph1_threshold = common_prop + graph1_prop

        for i in range(self.part1_size):
            for j in range(self.part2_size):
                prob = np.random.random()
                if prob < common_prop:
                    graph1.add_edge((i, j + self.part1_size))
                    graph2.add_edge((i, j + self.part1_size))
                elif common_prop < prob < graph1_threshold:
                    graph1.add_edge((i, j + self.part1_size))
                    graph2.remove_edge((i, j + self.part1_size))
                else:
                    graph1.remove_edge((i, j + self.part1_size))
                    graph2.add_edge((i, j + self.part1_size))

        return graph1, graph2

    def copy(self):
        return BipartiteGraph(self.part1_size, self.part2_size, adj_matrix=np.copy(self.adj_matrix))

    def stats(self):
        num_present = len(np.where(self.adj_matrix == 1)[0])
        num_absent = len(np.where(self.adj_matrix == 0)[0])
        num_unknown = len(np.where(self.adj_matrix == 2)[0])

        return num_absent, num_present, num_unknown

    def fix_edges(self):
        for i in range(self.part1_size):
            for j in range(self.part2_size):
                if self.adj_matrix[i][j] == 2:
                    self.adj_matrix[i][j] = 0