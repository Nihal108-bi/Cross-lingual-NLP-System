# src/evaluation.py

"""
Provides functions for evaluating translation and alignment quality. 
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sacrebleu.metrics import BLEU
from typing import List


def evaluate_translation_bleu(hypotheses: List[str], references: List[List[str]]) -> float:
    """
    Calculates BLEU score using sacrebleu, with proper reference formatting.
    """
    
    bleu = BLEU()
    
    # Transpose reference format: [[ref1], [ref2], ...] → [['ref1', 'ref2', ...]]
    transposed_refs = list(map(list, zip(*references)))
    
    return bleu.corpus_score(hypotheses, transposed_refs).score


def evaluate_alignment_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """
    Calculates the mean cosine similarity between two sets of embeddings. 

    Args:
        embeddings1 (np.ndarray): The first set of embeddings.
        embeddings2 (np.ndarray): The second set of embeddings.

    Returns:
        float: The average cosine similarity.
    """
    if embeddings1.shape != embeddings2.shape:
        raise ValueError("Embedding matrices must have the same shape.")

    # Calculate cosine similarity for each pair of vectors (row-wise)
    sims = cosine_similarity(embeddings1, embeddings2)
    
    # We are interested in the similarity of corresponding pairs, which is the diagonal
    mean_similarity = np.mean(np.diag(sims))
    return float(mean_similarity)