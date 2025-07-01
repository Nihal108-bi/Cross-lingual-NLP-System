# src/embedding_aligner.py

"""
Implements cross-lingual embedding alignment using supervised mapping. 
"""
import numpy as np

class EmbeddingAligner:
    """
    Aligns two sets of embeddings using a linear transformation (Procrustes).
    """
    def __init__(self):
        self.mapping_matrix = None

    def learn_mapping(self, X: np.ndarray, Y: np.ndarray):
        """
        Learns a linear mapping from source embedding space (X) to target (Y).
        This is a supervised method based on Orthogonal Procrustes.

        Args:
            X (np.ndarray): Source language embeddings (n_samples, embed_dim).
            Y (np.ndarray): Target language embeddings (n_samples, embed_dim).
        """
        print("Learning embedding space mapping...")
        if X.shape != Y.shape:
            raise ValueError("Source and target embedding matrices must have the same shape.")

        # Compute the matrix product XY^T
        prod = np.dot(X.T, Y)

        # Perform Singular Value Decomposition (SVD)
        U, _, V_t = np.linalg.svd(prod)

        # The optimal orthogonal mapping matrix is W = U * V_t
        self.mapping_matrix = np.dot(U, V_t)
        print("Mapping matrix learned successfully.")

    def align_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Aligns source embeddings to the target space using the learned mapping.

        Args:
            X (np.ndarray): Source embeddings to align.

        Returns:
            np.ndarray: Aligned embeddings.
        """
        if self.mapping_matrix is None:
            raise RuntimeError("Mapping must be learned before alignment. Call 'learn_mapping' first.")
        return np.dot(X, self.mapping_matrix)