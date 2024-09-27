from Graph import Graph
import numpy as np
from tqdm import tqdm


class RevisitedSpectral:

    def __init__(self, reconstructed_graph, A):
        self.last_G = reconstructed_graph
        self.G = reconstructed_graph
        self.A = A


    def run(self, alpha=1, beta=1, threshold=0.5):
        M_star = np.zeros_like(self.G.adj_matrix)
        U_A, S_A, V_A = np.linalg.svd(self.A, compute_uv=True)

        Gstar_edges = np.argwhere(self.G.adj_matrix == 1)
        number_of_edges = np.sum(np.diag(self.A))

        for i in tqdm(range(S_A.shape[0]), desc="Revisited spectral attack"):

            S_Gi_plus = np.sqrt(S_A[i])
            S_Gi_minus =  - np.sqrt(S_A[i])

            M_star_plus = M_star + U_A[:, i].reshape(-1, 1) * S_Gi_plus * V_A[i, :].reshape(1, -1)
            M_star_minus = M_star + U_A[:, i].reshape(-1, 1) * S_Gi_minus * V_A[i, :].reshape(1, -1)

            binary_M_star_plus = np.where(M_star_plus >= threshold, 1, 0)
            binary_M_star_minus = np.where(M_star_minus >= threshold, 1, 0)

            Gstar_edges = np.argwhere(self.G.adj_matrix == 1)
            Gstar_non_edges = np.argwhere(self.G.adj_matrix == 0)

            distance_plus_M_G = np.linalg.norm(M_star_plus[Gstar_edges[:, 0], Gstar_edges[:, 1]] - self.G.adj_matrix[Gstar_edges[:, 0], Gstar_edges[:, 1]])
            distance_minus_M_G = np.linalg.norm(M_star_minus[Gstar_edges[:, 0], Gstar_edges[:, 1]] - self.G.adj_matrix[Gstar_edges[:, 0], Gstar_edges[:, 1]])

            distance_plus_M_G += np.linalg.norm(M_star_plus[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]])
            distance_minus_M_G += np.linalg.norm(M_star_minus[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]])

            distance_plus_M_binary = np.linalg.norm(M_star_plus - binary_M_star_plus, ord='fro')
            distance_minus_M_binary = np.linalg.norm(M_star_minus - binary_M_star_minus, ord='fro')

            distance_plus = alpha * distance_plus_M_G + beta * distance_plus_M_binary #+ distance_plus_M_non_edges
            distance_minus = alpha * distance_minus_M_G + beta * distance_minus_M_binary #+ distance_minus_M_non_edges

            if distance_plus < distance_minus:
                M_star = M_star_plus
            elif distance_minus < distance_plus:  
                M_star = M_star_minus
            else:
                M_star = M_star_plus


        M_star = np.where(M_star >= threshold, 1, 0)

        for edge in Gstar_edges:
            M_star[edge[0], edge[1]] = 1

        for non_edge in Gstar_non_edges:
            M_star[non_edge[0], non_edge[1]] = 0

        self.reconstructed_graph = Graph.from_adj_matrix(M_star)
        
    def sanity_check(self):
        modifs = 0
        A_prime = np.dot(self.reconstructed_graph.adj_matrix, self.reconstructed_graph.adj_matrix.T)
        for i in range(self.A.shape[0]):
            for j in range(i, self.A.shape[1]):
                if self.A[i, j] != A_prime[i, j] and self.A[i, i] == A_prime[i, i]:
                    for k in range(self.A.shape[0]):
                        if self.last_G.adj_matrix[i, k] == 2:
                            self.reconstructed_graph.adj_matrix[j, k] = 2
                            self.reconstructed_graph.adj_matrix[k, j] = 2
                            modifs += 1
                if self.A[i, j] != A_prime[i, j] and self.A[j, j] == A_prime[j, j]:
                    for k in range(self.A.shape[0]):
                        if self.last_G.adj_matrix[j, k] == 2:
                            self.reconstructed_graph.adj_matrix[i, k] = 2
                            self.reconstructed_graph.adj_matrix[k, i] = 2
                            modifs += 1

        print(f"Updated {modifs} edges in G*")


    def get_reconstructed_graph(self):
        return self.reconstructed_graph


