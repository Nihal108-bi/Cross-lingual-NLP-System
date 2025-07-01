# src/data_loader.py

"""
Handles loading and preprocessing of custom parallel corpora. 
"""
from typing import List, Tuple

def load_parallel_corpus(src_file: str, tgt_file: str) -> Tuple[List[str], List[str]]:
    """
    Loads a parallel corpus from two text files.

    Args:
        src_file (str): Path to the source language file.
        tgt_file (str): Path to the target language file.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing lists of
                                     source and target sentences.
    """
    print(f"Loading source corpus from: {src_file}")
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f.readlines()]

    print(f"Loading target corpus from: {tgt_file}")
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_sentences = [line.strip() for line in f.readlines()]

    if len(src_sentences) != len(tgt_sentences):
        raise ValueError("Source and target corpora must have the same number of sentences.")

    print(f"Successfully loaded {len(src_sentences)} parallel sentences.")
    return src_sentences, tgt_sentences