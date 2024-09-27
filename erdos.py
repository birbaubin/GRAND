def erdos(G_square, threshold=0.5):
    M_star = np.zeros_like(G_square)
    U_A, S_A, V_A = np.linalg.svd(G_square, compute_uv=True)
    for i in tqdm(range(S_A.shape[0])):

        S_Gi_plus = np.sqrt(S_A[i])
        S_Gi_minus =  - np.sqrt(S_A[i])

        add_plus = U_A[:, i].reshape(-1, 1) * S_Gi_plus * V_A[i, :].reshape(1, -1)
        add_minus = U_A[:, i].reshape(-1, 1) * S_Gi_minus * V_A[i, :].reshape(1, -1)

        M_star_plus = M_star + add_plus
        M_star_minus = M_star + add_minus

        binary_M_star_plus = np.where(M_star_plus > threshold, 1, 0)
        binary_M_star_minus = np.where(M_star_minus > threshold, 1, 0)

        distance_plus = np.linalg.norm(M_star_plus - binary_M_star_plus, ord='fro')
        distance_minus = np.linalg.norm(M_star_minus - binary_M_star_minus, ord='fro')        

        if distance_plus < distance_minus:
            M_star = M_star_plus
        else:
            M_star = M_star_minus

    M_star = np.where(M_star > threshold, 1, 0)
    return M_star


    