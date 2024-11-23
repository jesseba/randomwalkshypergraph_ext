import numpy as np
from scipy import sparse
from typing import Tuple, List

class HypergraphLaplacian:
    """Base class for different hypergraph Laplacian constructions"""
    
    def __init__(self, universe: np.ndarray, pi_list: List[Tuple]):
        """
        Args:
            universe: Array of all vertices/players
            pi_list: List of (players, scores) tuples for each match/hyperedge
        """
        self.universe = universe
        self.pi_list = pi_list
        self.n = len(universe)  # number of vertices
        self.m = len(pi_list)   # number of hyperedges
        
        # Initialize basic matrices needed by most constructions
        self.R, self.W = self._construct_basic_matrices()
        
    def _construct_basic_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct incidence and weight matrices:
        R: |E| x |V| vertex-weight matrix
        W: |V| x |E| hyperedge weight matrix
        """
        R = np.zeros([self.m, self.n])  # |E| x |V|
        W = np.zeros([self.n, self.m])  # |V| x |E|
        
        for i, (players, scores) in enumerate(self.pi_list):
            if len(players) > 1:
                for j, p in enumerate(players):
                    v_idx = np.where(self.universe == p)[0][0]
                    R[i, v_idx] = np.exp(scores[j])  # Vertex weight
                    W[v_idx, i] = 1.0
                
                # Edge weight is std dev of scores + 1
                W[:, i] = (np.std(scores) + 1.0) * W[:, i]
                # Normalize R row to sum to 1
                R[i, :] = R[i,:] / sum(R[i,:])
                
        return R, W
    
    def compute_laplacian(self) -> np.ndarray:
        """To be implemented by specific Laplacian constructions"""
        raise NotImplementedError
        
    def compute_transition_matrix(self, L: np.ndarray) -> np.ndarray:
        """Convert Laplacian to transition matrix"""
        return np.eye(self.n) - L
        
    def compute_pagerank(self, P: np.ndarray, r: float = 0.4, 
                        eps: float = 1e-8) -> np.ndarray:
        """
        Compute PageRank scores
        Args:
            P: Transition matrix
            r: Restart probability
            eps: Convergence threshold
        """
        x = np.ones(self.n) / self.n
        converged = False
        t = 0
        
        while not converged:
            x_new = (1-r)*P.dot(x) + (r/self.n)*np.ones(self.n)
            if (np.linalg.norm(x_new - x, ord=1) < eps and t > 100):
                converged = True
            t += 1
            x = x_new
            
        return x

class RandomWalkLaplacian(HypergraphLaplacian):
    """Random walk-based Laplacian from original paper"""
    
    def compute_laplacian(self) -> np.ndarray:
        # Compute vertex degrees
        d_v = np.sum(self.W, axis=1)
        D_v = np.diag(d_v)
        D_v_inv = np.diag(1.0 / d_v)
        
        # Compute transition matrix
        P = D_v_inv.dot(self.W).dot(self.R)
        P = P.T  # Use column stochastic version
        
        # Return I - P
        return np.eye(self.n) - P

class ZhouLaplacian(HypergraphLaplacian):
    """Implementation of Zhou et al. 2006 Laplacian"""
    
    def compute_laplacian(self) -> np.ndarray:
        # Compute degree matrices
        d_v = np.sum(self.W, axis=1)
        d_e = np.sum(self.R, axis=1)
        D_v = np.diag(d_v)
        D_e = np.diag(d_e)
        D_v_inv = np.diag(1.0 / d_v)
        D_e_inv = np.diag(1.0 / d_e)
        
        # Compute normalized Laplacian
        temp = D_v_inv.dot(self.W)
        temp = temp.dot(D_e_inv)
        temp = temp.dot(self.R)
        
        return np.eye(self.n) - temp

class ChanLaplacian(HypergraphLaplacian):
    """Implementation of Chan et al. 2018 Laplacian with mediators"""
    
    def __init__(self, universe: np.ndarray, pi_list: List[Tuple], 
                 beta: float = 0.5):
        """
        Args:
            beta: Mediator weight parameter (0 to 1)
                 Higher values = more direct flow vs mediated flow
        """
        super().__init__(universe, pi_list)
        self.beta = beta
        
    def compute_laplacian(self) -> np.ndarray:
        # Add debug prints
        print(f"W shape: {self.W.shape}")
        print(f"R shape: {self.R.shape}")
        print(f"Number of vertices (n): {self.n}")
        print(f"Number of edges (m): {self.m}")
        
        # Compute vertex degrees
        d_v = np.sum(self.W, axis=1)
        D_v = np.diag(d_v)
        D_v_sqrt = np.diag(np.sqrt(d_v))
        D_v_sqrt_inv = np.diag(1.0 / np.sqrt(d_v))
        
        # Compute edge degrees
        d_e = np.sum(self.R, axis=1)
        D_e = np.diag(d_e)
        
        try:
            # Compute normalized adjacency matrix
            H = self.W.dot(self.R)  # Non-normalized adjacency
            print(f"Successfully computed H with shape: {H.shape}")
        except ValueError as e:
            print(f"Error computing H: {str(e)}")
            raise
        
        # Direct flow component
        direct_flow = self.beta * H
        
        # Mediated flow component 
        mediated_flow = (1 - self.beta) * (H.dot(H))
        
        # Combined normalized matrix
        A_norm = direct_flow + mediated_flow
        
        # Return normalized Laplacian
        return np.eye(self.n) - A_norm

def create_synthetic_hypergraph(n_vertices: int, 
                              n_edges: int,
                              k_clusters: int) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Create synthetic hypergraph with known clustering structure
    
    Args:
        n_vertices: Number of vertices
        n_edges: Number of hyperedges
        k_clusters: Number of ground truth clusters
        
    Returns:
        universe: Vertex array
        pi_list: List of (players, scores) tuples
    """
    universe = np.arange(n_vertices)
    pi_list = []
    
    # Assign vertices to clusters
    cluster_size = n_vertices // k_clusters
    clusters = np.repeat(range(k_clusters), cluster_size)
    
    # Generate hyperedges biased towards clusters
    for _ in range(n_edges):
        # Pick main cluster
        main_cluster = np.random.randint(k_clusters)
        cluster_vertices = universe[clusters == main_cluster]
        
        # Add some noise vertices
        other_vertices = universe[clusters != main_cluster]
        noise_vertices = np.random.choice(
            other_vertices,
            size=np.random.randint(1, 4),
            replace=False
        )
        
        # Combine vertices
        edge_vertices = np.concatenate([
            np.random.choice(
                cluster_vertices,
                size=np.random.randint(3, 7),
                replace=False
            ),
            noise_vertices
        ])
        
        # Generate random scores
        scores = np.random.rand(len(edge_vertices))
        
        pi_list.append((edge_vertices, scores))
        
    return universe, pi_list