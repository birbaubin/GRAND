import numpy as np
from tqdm import tqdm

def GRAND(A, Gstar, threshold):
    M_star = np.zeros_like(Gstar.adj_matrix)
    U_A, S_A, V_A = np.linalg.svd(A, compute_uv=True)

    number_of_edges = np.sum(np.diag(A))

    Gstar_edges = np.argwhere(Gstar.adj_matrix == 1)
    Gstar_non_edges = np.argwhere(Gstar.adj_matrix == 0)

    choices = []

    for i in tqdm(range(S_A.shape[0])):


        S_Gi_plus = np.sqrt(S_A[i])
        S_Gi_minus =  - np.sqrt(S_A[i])

        M_star_plus = M_star + U_A[:, i].reshape(-1, 1) * S_Gi_plus * V_A[i, :].reshape(1, -1)
        M_star_minus = M_star + U_A[:, i].reshape(-1, 1) * S_Gi_minus * V_A[i, :].reshape(1, -1)

        binary_M_star_plus = np.where(M_star_plus >= threshold, 1, 0)
        binary_M_star_minus = np.where(M_star_minus >= threshold, 1, 0)

        # distance_plus_M_G = np.linalg.norm(M_star_plus[Gstar_edges[:, 0], Gstar_edges[:, 1]] - Gstar.adj_matrix[Gstar_edges[:, 0], Gstar_edges[:, 1]])
        # distance_minus_M_G = np.linalg.norm(M_star_minus[Gstar_edges[:, 0], Gstar_edges[:, 1]] - Gstar.adj_matrix[Gstar_edges[:, 0], Gstar_edges[:, 1]])

        distance_plus_M_G = np.linalg.norm(M_star_plus[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]] - Gstar.adj_matrix[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]])
        distance_minus_M_G = np.linalg.norm(M_star_minus[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]] - Gstar.adj_matrix[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]])

        distance_plus_M_binary = np.linalg.norm(M_star_plus - binary_M_star_plus, ord='fro')
        distance_minus_M_binary = np.linalg.norm(M_star_minus - binary_M_star_minus, ord='fro')

        print("Distance plus M_G = ", distance_plus_M_G)
        print("Distance minus M_G = ", distance_minus_M_G)
        print("Distance plus M_binary = ", distance_plus_M_binary)
        print("Distance minus M_binary = ", distance_minus_M_binary)

        distance_plus = distance_plus_M_G + distance_plus_M_binary #+ distance_plus_M_non_edges
        distance_minus = distance_minus_M_G + distance_minus_M_binary #+ distance_minus_M_non_edges

        if distance_plus < distance_minus:
            M_star = M_star_plus
            choices.append(1)
        else:
            M_star = M_star_minus
            choices.append(-1)

    M_star = np.where(M_star >= threshold, 1, 0)

    for edge in Gstar_edges:
        M_star[edge[0], edge[1]] = 1

    print("Choices = ", choices)
    return M_star

