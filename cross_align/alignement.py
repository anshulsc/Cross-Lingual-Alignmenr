import numpy as np
from scipy.linalg import orthogonal_procrustes


def get_word_vector(model, word):
    return model.get_word_vector(word)

def align_embeddings(source_emb, target_emb, src_words, tgt_words, lexicon):
    src_vectors = []
    tgt_vectors = []
    
    for src_word, tgt_word in lexicon:
        if src_word in src_words and tgt_word in tgt_words:
            src_vectors.append(get_word_vector(source_emb, src_word))
            tgt_vectors.append(get_word_vector(target_emb, tgt_word))
    
    src_aligned = np.array(src_vectors)
    tgt_aligned = np.array(tgt_vectors)
    
    R, _ = orthogonal_procrustes(src_aligned, tgt_aligned)
    return R


def apply_alignment(model, alignment_matrix):
    aligned_vectors = {}
    for word in model.get_words():
        original_vector = get_word_vector(model, word)
        aligned_vector = np.dot(original_vector, alignment_matrix)
        aligned_vectors[word] = aligned_vector
    return aligned_vectors