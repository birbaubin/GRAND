from Graph import Graph
import numpy as np
from tqdm import tqdm


class SpectralAttack:
    def __init__(self, A, threshold=0.5):
        self.A = A
        self.G = Graph(range(A.shape[0]))
        self.threshold = threshold

    def run(self):
        M_star = np.zeros_like(self.A)
        U_A, S_A, V_A = np.linalg.svd(self.A, compute_uv=True)
        for i in tqdm(range(min(self.A.shape)), desc="Spectral attack"):

            S_Gi_plus = np.sqrt(S_A[i])
            S_Gi_minus =  - np.sqrt(S_A[i])

            add_plus = U_A[:, i].reshape(-1, 1) * S_Gi_plus * V_A[i, :].reshape(1, -1)
            add_minus = U_A[:, i].reshape(-1, 1) * S_Gi_minus * V_A[i, :].reshape(1, -1)

            M_star_plus = M_star + add_plus
            M_star_minus = M_star + add_minus

            binary_M_star_plus = np.where(M_star_plus > self.threshold, 1, 0)
            binary_M_star_minus = np.where(M_star_minus > self.threshold, 1, 0)

            distance_plus = np.linalg.norm(M_star_plus - binary_M_star_plus, ord='fro')
            distance_minus = np.linalg.norm(M_star_minus - binary_M_star_minus, ord='fro')        

            # print(distance_plus, distance_minus)
            
            if distance_plus < distance_minus:
                M_star = M_star_plus
            else:
                M_star = M_star_minus

            # print("Iteration ", i, M_star[:5, :5])

        M_star = np.where(M_star > self.threshold, 1, 0)
        self.G = Graph.from_adj_matrix(M_star)

    def get_Gstar(self):
        return self.G


    