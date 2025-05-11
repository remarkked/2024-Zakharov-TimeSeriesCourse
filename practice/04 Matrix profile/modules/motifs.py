import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []
    
    # Copy matrix profile and its indices
    mp = np.array(matrix_profile['mp'])
    mpi = np.array(matrix_profile['mpi']).astype(int)
    exclusion_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        if is_nan_inf(mp):
            break

        min_idx = np.argmin(mp)
        min_distance = mp[min_idx]

        motifs_dist.append([0, min_distance])
        motifs_idx.append(np.sort([min_idx, mpi[min_idx]]))

        mp = apply_exclusion_zone(mp, min_idx, exclusion_zone, np.inf)
        mp = apply_exclusion_zone(mp, mpi[min_idx], exclusion_zone, np.inf)

    return {
        "distances" : motifs_dist,
        "indices" : motifs_idx,
        }
