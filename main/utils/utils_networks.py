import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import sparse

def prepare_lp_matrices(network,
                        n_nodes,
                        gamma,
                        franchise,
                        leftover_prev_level=None):
    """
    Helper function to generat matrices needed in the linear programming
    exercises. 
    
    PARAMETER
    ---------
    network: pd.DataFrame
        Edgelist of the network (without reciprocals)
        
    n_nodes: int or float or similar
        Amount of nodes in the network
        
    gamma: int or float or similar
        Upper bound on gamma parameter
        
    franchise: int or float or similar
        Amount of the franchise
        
    leftover_prev_level: array, dataframe or None, optional (default=None)
        maximal leftover amount sharable (s - z vector from paper). If
        the argument is None a vector with the maximum amount of the gamma
        parameter is used
        
        
    returns: sparse matrices to be used in optimization step 
    """
        
    matrix_X = network.to_numpy()
    nrow_X = matrix_X.shape[0]
    
    a = np.repeat(1, nrow_X)
    
    A = sparse.lil_matrix((n_nodes, nrow_X), 
                          dtype=np.int8)
    
    for i in np.arange(0, matrix_X.shape[0]):
        A[matrix_X[i, 0:2], i] = 1
        
    A = sparse.csr_matrix(A)
        
    B = sparse.identity(nrow_X)
    
    AB = sparse.vstack([B, A])
    
    if isinstance(leftover_prev_level, np.ndarray):
        print('Note: using previous results')
        ab = np.concatenate((np.repeat(gamma, B.shape[0]),
                             leftover_prev_level), 
                             axis=None)
    else:    
        ab = np.concatenate((np.repeat(gamma, B.shape[0]),
                             np.repeat(franchise, A.shape[0])), 
                             axis=None)
    
    return a, AB, ab, nrow_X

def edges2adjmat(e): 
    """Helper function"""
    ea = np.array(e)
    numverts = ea.max() + 1
    a = sparse.lil_matrix((numverts,numverts))

    for edge in e:
        a[edge[0].__int__(),edge[1].__int__()] = 1
        a[edge[1].__int__(),edge[0].__int__()] = 1

    return a


def p2p_lin_solver(a, AB, ab, nrow_X, max_iters=1000):
    """
    Wrapper for the linear programming exerciese
    """
    z = cp.Variable(nrow_X)

    prob = cp.Problem(cp.Maximize(a.T@z),
                     [AB @ z <= ab, 
                      z >= 0])

    prob.solve(solver=cp.CBC)
    
    return z

def recover_sharing_amount(network, 
                           lp_solution):
    """
    Function to recover the total amount shared between nodes given 
    a linear programming solution. 
    
    PARAMETER
    ---------
    network: pd.DataFrame
        Network dataframe without reciprocal connections
        
    lp_solution: array
        Solution from a linear programming exercise
    
    """
    df = network.copy()
    
    df['node_weight'] = np.round(lp_solution)
    
    df_reciprocal = (df
                     .append((df
                              .loc[:, ['id1', 'id2', 'node_weight']]
                              .rename(columns={'id1': 'id2',
                                               'id2': 'id1'})),
                                    ignore_index=True))
    
    risk_sharing = (df_reciprocal
                    .groupby('id2')
                    .agg({'node_weight': sum})
                    .reset_index())

    return risk_sharing


def prepare_friends_of_friends_sparse(full_network,
                               friend_level):
    """
    Sparse implementation to get "friends of friends"-network
    from original edgelist. 
    
    PARAMETER
    ---------
    
    full_network: pd.DataFrame
        Full edgelist (including reciprocal edges)
        
    friend_level: int
        Level for which the network should be calculated. 1 would be the original network.
        2 corresponds to friends-of-friends, 3 to friends-of-friends-of-friends etc. 
    """
    
    # Create adjacency matrix from dataframe
    sparse_adj_mat = edges2adjmat(full_network.to_numpy())
    
    # Work down the connection level
    squared_adj = sparse_adj_mat*sparse_adj_mat
    # Remove self connections
    squared_adj.setdiag(0)
    # Set to maximum one (as some nodes may be friends of multiple of our friends)
    squared_adj[squared_adj > 0] = 1
    
    # Recast to pandas for easier data manip
    friends_of_friends_df = (pd.melt((pd.DataFrame.sparse.from_spmatrix(squared_adj)
                                  .reset_index()
                                  .rename(columns={'index': 'id2'})),
                                 id_vars=['id2'],
                                 var_name='id1', 
                                 value_name='connection')
                        )
    friends_of_friends_df = friends_of_friends_df.loc[friends_of_friends_df.connection == 1, 
                                                      ['id2', 'id1']]
    
    return friends_of_friends_df


def return_table(series_):
    unique, counts = np.unique(series_, return_counts=True)

    return np.asarray((unique, counts)).T