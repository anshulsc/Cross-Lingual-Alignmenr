import numpy as np
from scipy.linalg import orthogonal_procrustes
import os

def get_word_vector(model, word):
    """Get word vector from the model."""
    return model.get_word_vector(word)

def align_embeddings(source_emb, target_emb, src_words, tgt_words, lexicon):
    """Perform initial Procrustes alignment between source and target embeddings."""
    src_vectors = []
    tgt_vectors = []
    
    for src_word, tgt_word in tqdm(lexicon, desc="Aligning embeddings"):
        if src_word in src_words and tgt_word in tgt_words:
            src_vectors.append(get_word_vector(source_emb, src_word))
            tgt_vectors.append(get_word_vector(target_emb, tgt_word))
    
    src_aligned = np.array(src_vectors)
    tgt_aligned = np.array(tgt_vectors)
    R, _ = orthogonal_procrustes(src_aligned, tgt_aligned)
    return R

def apply_alignment(model, alignment_matrix):
    aligned_vectors = {}
    for word in tqdm(model.get_words(), desc="Applying alignment"): 
        original_vector = get_word_vector(model, word)
        aligned_vector = np.dot(original_vector, alignment_matrix)
        aligned_vectors[word] = aligned_vector
    return aligned_vectors

import numpy as np
from scipy.linalg import orthogonal_procrustes
import os
import logging
from tqdm import tqdm

def get_word_vector(model, word):
    return model.get_word_vector(word)

def align_embeddings(source_emb, target_emb, src_words, tgt_words, lexicon):
    src_vectors = []
    tgt_vectors = []
    
    for src_word, tgt_word in tqdm(lexicon, desc="Aligning embeddings"):
        if src_word in src_words and tgt_word in tgt_words:
            src_vectors.append(get_word_vector(source_emb, src_word))
            tgt_vectors.append(get_word_vector(target_emb, tgt_word))
    
    src_aligned = np.array(src_vectors)
    tgt_aligned = np.array(tgt_vectors)

    R, _ = orthogonal_procrustes(src_aligned, tgt_aligned)
    return R

def apply_alignment(model, alignment_matrix):

    aligned_vectors = {}
    for word in tqdm(model.get_words(), desc="Applying alignment"):
        original_vector = get_word_vector(model, word)
        aligned_vector = np.dot(original_vector, alignment_matrix)
        aligned_vectors[word] = aligned_vector
    return aligned_vectors

def iterative_procrustes_alignment(source_emb, target_emb, src_words, tgt_words, lexicon, num_iterations=10, tol=1e-5):
    logging.info("Starting iterative Procrustes alignment...")
    

    alignment_matrix = align_embeddings(source_emb, target_emb, src_words, tgt_words, lexicon)
    logging.info("Initial alignment matrix computed.")
    
    for iteration in range(num_iterations):
        logging.info(f"Iteration {iteration + 1}: Applying alignment to source embeddings...")
        
        # Apply the current alignment to the source embeddings
        aligned_source_embeddings = apply_alignment(source_emb, alignment_matrix)
        
        # Extract aligned vectors for the lexicon
        src_vectors = []
        tgt_vectors = []
        for src_word, tgt_word in tqdm(lexicon, desc=f"Preparing vectors for iteration {iteration + 1}"):
            if src_word in aligned_source_embeddings and tgt_word in tgt_words:
                src_vectors.append(aligned_source_embeddings[src_word])
                tgt_vectors.append(get_word_vector(target_emb, tgt_word))
        
        src_matrix = np.array(src_vectors)
        tgt_matrix = np.array(tgt_vectors)
        
        # Compute the new alignment matrix using orthogonal Procrustes
        new_alignment_matrix, _ = orthogonal_procrustes(src_matrix, tgt_matrix)
        logging.info(f"Iteration {iteration + 1}: New alignment matrix computed.")
        
        # Compute the difference between the new and previous alignment matrices
        matrix_diff = np.linalg.norm(new_alignment_matrix - alignment_matrix)
        logging.info(f"Iteration {iteration + 1}: Matrix difference = {matrix_diff:.6f}")
        
        # Check for convergence
        if matrix_diff < tol:
            logging.info("Convergence reached.")
            alignment_matrix = new_alignment_matrix
            break
        
        # Update the alignment matrix for the next iteration
        alignment_matrix = new_alignment_matrix
    
    logging.info("Iterative Procrustes alignment completed.")
    return alignment_matrix



def load_bilingual_lexicon_pait(source_lang, target_lang, muse_dir, max_pairs=None, train=True):
    file_suffix = 'train' if train else 'test'
    lexicon_path = os.path.join(muse_dir, f"{source_lang}-{target_lang}.{file_suffix}.txt")
    
    if not os.path.exists(lexicon_path):
        raise FileNotFoundError(f"Bilingual lexicon not found at {lexicon_path}")
    
    word_pairs = []
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            source_word, target_word = line.strip().split()
            word_pairs.append((source_word, target_word))
            if max_pairs and len(word_pairs) >= max_pairs:
                break
    
    return word_pairs